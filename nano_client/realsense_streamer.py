"""Stream Intel RealSense RGB (e.g. D435i) to the GPU SLAM server with factory intrinsics.

Uses ``pyrealsense2`` at the configured color resolution, reads
``video_stream_profile.get_intrinsics()`` (fx, fy, ppx, ppy), and sends
multipart ``[utf8_json, jpeg]`` matching ``gpu_server/droideka_src/live_slam.py``.

Requires ``pip install pyrealsense2`` on the host (typically not available as a
wheel on Jetson aarch64 — use a dev PC or build librealsense from source).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import zmq

from camera_intrinsics import scale_pinhole_intrinsics


def _try_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> bytes:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not success:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buffer.tobytes()


def start_streaming(
    server_ip: str,
    port: int,
    resize: tuple[int, int],
    jpeg_quality: int,
) -> None:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: pyrealsense2 is not installed. Install with "
            "`pip install pyrealsense2` (Windows / x86_64 Linux wheels).",
            file=sys.stderr,
        )
        return

    out_w, out_h = resize
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{server_ip}:{port}")
    print(f"ZeroMQ PUSH connected to tcp://{server_ip}:{port}.")

    pipeline = rs.pipeline()
    candidates = [(out_w, out_h), (1280, 720), (848, 480), (640, 480)]
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for wh in candidates:
        if wh not in seen:
            ordered.append(wh)
            seen.add(wh)

    profile = None
    native_w, native_h = out_w, out_h
    last_err: Optional[BaseException] = None
    for nw, nh in ordered:
        try:
            try:
                pipeline.stop()
            except Exception:
                pass
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, nw, nh, rs.format.bgr8, 30)
            profile = pipeline.start(cfg)
            native_w, native_h = nw, nh
            if (nw, nh) != (out_w, out_h):
                print(
                    f"NOTE: color sensor has no {out_w}x{out_h} BGR8 mode; "
                    f"using {nw}x{nh} then cv2.resize to output."
                )
            break
        except RuntimeError as e:
            last_err = e
            profile = None

    if profile is None:
        print(
            f"ERROR: could not start a BGR8 color stream (tried {ordered}): {last_err}",
            file=sys.stderr,
        )
        socket.close(linger=0)
        return

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    if (native_w, native_h) != (out_w, out_h):
        fx, fy, cx, cy = scale_pinhole_intrinsics(
            fx, fy, cx, cy, (native_w, native_h), (out_w, out_h)
        )
    print(
        f"RealSense color native {native_w}x{native_h} -> output {out_w}x{out_h} BGR8 "
        f"@ {color_profile.fps()} FPS; intrinsics (for output) "
        f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}"
    )
    intrinsics_json = json.dumps(
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, separators=(",", ":")
    ).encode("utf-8")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frames = pipeline.wait_for_frames(timeout_ms=5000)
            if not ok or frames is None:
                print("WARNING: wait_for_frames timed out.")
                continue

            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            if frame is None or frame.size == 0:
                continue

            if (native_w, native_h) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h))

            payload = _encode_jpeg(frame, quality=jpeg_quality)
            socket.send_multipart([intrinsics_json, payload])

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-6)
                size_kb = len(payload) / 1024
                print(
                    f"Sent: {frame_count} frames | FPS: {fps:.1f} | "
                    f"Size: {size_kb:.1f} KB"
                )

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        socket.close(linger=0)
        print("Camera and network closed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream RealSense RGB to the GPU SLAM server with per-device intrinsics "
            "(multipart ZMQ: JSON + JPEG)."
        ),
    )
    parser.add_argument(
        "--server-ip",
        default=os.environ.get("DROIDEKA_SERVER_IP", "127.0.0.1"),
        help="Server IP / hostname (default: 127.0.0.1 or env DROIDEKA_SERVER_IP).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_try_int(os.environ.get("DROIDEKA_SERVER_PORT")) or 5555,
        help="Server port (default: 5555 or env DROIDEKA_SERVER_PORT).",
    )
    parser.add_argument(
        "--resize",
        default="960x288",
        help=(
            "Output JPEG size WIDTHxHEIGHT after optional downscale (default: 960x288). "
            "Must match live_slam --image_size order (H,W) = (HEIGHT, WIDTH). "
            "If the sensor has no native BGR8 mode at this size, a supported mode "
            "(e.g. 1280x720) is used and frames are resized."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (0-100, default: 95).",
    )
    args = parser.parse_args()

    if "x" not in args.resize.lower():
        raise ValueError("--resize must be WIDTHxHEIGHT (e.g. 960x288)")
    w_str, h_str = args.resize.lower().split("x", 1)
    resize = (int(w_str), int(h_str))

    start_streaming(
        server_ip=args.server_ip,
        port=args.port,
        resize=resize,
        jpeg_quality=args.jpeg_quality,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
