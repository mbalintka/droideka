"""Replay a RealSense ``.bag`` recording over ZeroMQ — same wire format as
``streamer.py`` / ``zmq_video_sender.py``.

Use this to feed the GPU server (``gpu_server/droideka_src/live_slam.py``) with
real on-car footage without needing the live Jetson + camera. Only the color
stream is needed; DROID-SLAM is monocular RGB here.

Example::

    # Terminal 1 (GPU box)
    python -m droideka_src.live_slam

    # Terminal 2 (dev machine with the .bag)
    python nano_client/bag_streamer.py \
        --server-ip <gpu> --bag "20260305_135542 (1).bag"

    # Terminal 3 (Nano, only needed to exercise the controller)
    python -m nano_client.controller.node --gpu-host <gpu>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple

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


def _parse_resize(spec: str) -> Optional[Tuple[int, int]]:
    if spec.lower() == "none":
        return None
    if "x" not in spec.lower():
        raise ValueError("--resize must be WIDTHxHEIGHT or 'none' (e.g. 960x288)")
    w_str, h_str = spec.lower().split("x", 1)
    return (int(w_str), int(h_str))


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> bytes:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not success:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buffer.tobytes()


def _to_bgr(raw: np.ndarray, fmt: "object") -> np.ndarray:
    """Convert a RealSense color frame buffer to OpenCV-friendly BGR8.

    ``fmt`` is the ``rs.format`` of the source frame. We only handle the color
    formats the D435i can actually emit; anything else raises so the caller
    sees a real error instead of a silently wrong frame.
    """
    # pyrealsense2 is imported lazily inside the caller, so do the same here.
    import pyrealsense2 as rs  # type: ignore[import-not-found]

    if fmt == rs.format.bgr8:
        return raw
    if fmt == rs.format.rgb8:
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    if fmt == rs.format.bgra8:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    if fmt == rs.format.rgba8:
        return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
    if fmt == rs.format.yuyv:
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == rs.format.uyvy:
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_UYVY)
    if fmt == rs.format.y8:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported RealSense color format for replay: {fmt}")


def stream_bag(
    bag_path: str,
    server_ip: str,
    port: int,
    resize: Optional[Tuple[int, int]],
    jpeg_quality: int,
    rate_hz: float,
    skip_frames: int,
    max_frames: Optional[int],
) -> int:
    """Push color frames from ``bag_path`` to ``tcp://server_ip:port``.

    ``rate_hz <= 0`` means "send as fast as possible". ``skip_frames`` drops
    every Nth frame at the source so we can downsample very high-FPS captures
    cheaply before the JPEG encode.
    """
    # pyrealsense2 is only needed for this script; import lazily so the rest
    # of nano_client/ stays usable on dev machines without the RealSense SDK.
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: pyrealsense2 is not installed. Install it with "
            "`pip install pyrealsense2` (Windows/Linux x86_64 wheels available).",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(bag_path):
        print(f"ERROR: bag file not found: {bag_path}", file=sys.stderr)
        return 2

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    # Keep linger short so Ctrl+C exits promptly even if the GPU isn't draining.
    socket.setsockopt(zmq.LINGER, 100)
    socket.connect(f"tcp://{server_ip}:{port}")
    print(f"ZeroMQ PUSH connected to tcp://{server_ip}:{port}.")

    # Non-real-time playback so we never silently drop frames while the GPU is
    # busy with a heavy DROID-SLAM iteration. We do our own pacing below.
    #
    # NOTE: we deliberately do NOT pin the color format here. The RealSense
    # device almost never records in BGR8 — it's typically YUYV or RGB8 — so
    # constraining the format upfront makes pipeline.start() fail with the
    # opaque "Couldn't resolve requests" error. We accept whatever the bag has
    # and convert to BGR after the fact (see _to_bgr below).
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path, repeat_playback=False)
    cfg.enable_all_streams()

    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        print(f"ERROR: failed to open RealSense bag {bag_path}: {e}", file=sys.stderr)
        socket.close(linger=0)
        return 2

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Find the color video stream in the bag (it may have depth / IR too).
    color_profile = None
    stream_names = []
    for sp in profile.get_streams():
        stream_names.append(sp.stream_name())
        if sp.stream_type() == rs.stream.color:
            color_profile = sp.as_video_stream_profile()

    if color_profile is None:
        print(
            f"ERROR: bag has no color stream. Streams in file: {stream_names}",
            file=sys.stderr,
        )
        try:
            pipeline.stop()
        finally:
            socket.close(linger=0)
        return 2

    src_w, src_h = color_profile.width(), color_profile.height()
    src_fmt = color_profile.format()
    print(
        f"Bag opened: color={src_w}x{src_h} @ {color_profile.fps()} FPS "
        f"({src_fmt}). Streams in file: {stream_names}"
    )

    intr = color_profile.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    if resize is not None:
        dst_w, dst_h = resize
        fx, fy, cx, cy = scale_pinhole_intrinsics(
            fx, fy, cx, cy, (src_w, src_h), (dst_w, dst_h)
        )
        print(
            f"Intrinsics scaled for resize {src_w}x{src_h} -> {dst_w}x{dst_h}: "
            f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}"
        )
    else:
        print(f"Intrinsics at bag resolution: fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")

    intrinsics_json = json.dumps(
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, separators=(",", ":")
    ).encode("utf-8")

    delay_s = 1.0 / max(rate_hz, 1e-6) if rate_hz > 0.0 else 0.0
    sent = 0
    read = 0
    start_time = time.time()
    last_log_t = start_time

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok or frames is None:
                # End of bag (or a long stall — same effect for an offline file).
                break

            color = frames.get_color_frame()
            if not color:
                continue

            read += 1
            if skip_frames > 0 and (read - 1) % (skip_frames + 1) != 0:
                continue

            raw = np.asanyarray(color.get_data())
            if raw is None or raw.size == 0:
                continue

            try:
                frame = _to_bgr(raw, src_fmt)
            except RuntimeError as e:
                # Surface this loudly and stop — the rest of the bag will fail
                # the same way and we don't want to flood the GPU with garbage.
                print(f"ERROR: {e}", file=sys.stderr)
                break

            if resize is not None:
                frame = cv2.resize(frame, resize)

            try:
                payload = _encode_jpeg(frame, quality=jpeg_quality)
            except RuntimeError as e:
                print(f"WARNING: JPEG encode failed on frame {read}: {e}")
                continue

            socket.send_multipart([intrinsics_json, payload])
            sent += 1

            now = time.time()
            if now - last_log_t >= 1.0:
                elapsed = now - start_time
                fps = sent / max(elapsed, 1e-6)
                print(
                    f"sent={sent} read={read} fps_out={fps:.1f} "
                    f"size={len(payload) / 1024:.1f} KB"
                )
                last_log_t = now

            if max_frames is not None and sent >= max_frames:
                print(f"Reached --max-frames={max_frames}, stopping.")
                break

            if delay_s > 0.0:
                time.sleep(delay_s)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        socket.close(linger=0)
        elapsed = time.time() - start_time
        print(
            f"Done. read={read} sent={sent} elapsed={elapsed:.1f}s "
            f"avg_fps={sent / max(elapsed, 1e-6):.1f}"
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the color stream of a RealSense .bag as JPEG frames over "
            "ZeroMQ PUSH. Sends multipart [intrinsics_json, jpeg] for "
            "gpu_server/droideka_src/live_slam.py."
        ),
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to the RealSense .bag file to replay.",
    )
    parser.add_argument(
        "--server-ip",
        default=os.environ.get("DROIDEKA_SERVER_IP", "127.0.0.1"),
        help="GPU server IP / hostname (default: 127.0.0.1 or env DROIDEKA_SERVER_IP).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_try_int(os.environ.get("DROIDEKA_SERVER_PORT")) or 5555,
        help="Server frames port (default: 5555 or env DROIDEKA_SERVER_PORT).",
    )
    parser.add_argument(
        "--resize",
        default="960x288",
        help=(
            "Resize WIDTHxHEIGHT before JPEG encoding (default: 960x288). "
            "Pinhole intrinsics are read from the bag and scaled to match. "
            "Use 'none' to keep the bag's native resolution."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help=(
            "Outgoing frame rate; 0 means 'as fast as possible' (useful for "
            "deterministic offline runs). Default: 30 Hz, matching streamer.py."
        ),
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help=(
            "Drop this many source frames between each sent frame "
            "(0 = send every frame, 1 = every other frame, ...)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after sending this many frames (for quick sanity runs).",
    )
    args = parser.parse_args()

    resize = _parse_resize(args.resize)

    return stream_bag(
        bag_path=args.bag,
        server_ip=args.server_ip,
        port=args.port,
        resize=resize,
        jpeg_quality=args.jpeg_quality,
        rate_hz=args.rate_hz,
        skip_frames=max(0, int(args.skip_frames)),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
