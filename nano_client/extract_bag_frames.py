"""Decode the color stream of a RealSense ``.bag`` into a folder of JPEG frames.

Companion to :mod:`nano_client.bag_streamer`. Use this when the host running
the bag (a dev box) cannot reach the GPU server directly, but the Nano can —
decode here, copy the JPEG folder to the Nano, replay there with the existing
``zmq_video_sender.py``. The Nano never needs ``pyrealsense2`` (which has no
prebuilt aarch64 wheel on PyPI).

Wire format on the Nano stays identical to live ``streamer.py`` — JPEG bytes
on ZMQ PUSH port 5555 — so DROID-SLAM on the GPU sees the same input as it
would from the on-car camera.

Example::

    # 1. On the host with the .bag and pyrealsense2 installed:
    python nano_client/extract_bag_frames.py \
        --bag "20260305_135542 (1).bag" --out bag_frames

    # 2. Copy the folder to the Nano (any of scp / rsync / shared FS).
    scp -r bag_frames/ nano:~/droideka/

    # 3. On the Nano, replay over the existing video-sender:
    python nano_client/zmq_video_sender.py \
        --server-ip <gpu> --source 'bag_frames/*.jpg' --resize none --fps 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def _parse_resize(spec: str) -> Optional[Tuple[int, int]]:
    if spec.lower() == "none":
        return None
    if "x" not in spec.lower():
        raise ValueError("--resize must be WIDTHxHEIGHT or 'none' (e.g. 960x288)")
    w_str, h_str = spec.lower().split("x", 1)
    return (int(w_str), int(h_str))


def _to_bgr(raw: np.ndarray, fmt: "object") -> np.ndarray:
    """Convert a RealSense color frame buffer to OpenCV-friendly BGR8.

    Mirrors :func:`nano_client.bag_streamer._to_bgr` — we keep the two copies
    in sync rather than introducing a shared util for two ~10-line helpers.
    """
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
    raise RuntimeError(f"Unsupported RealSense color format for extraction: {fmt}")


def extract(
    bag_path: str,
    out_dir: str,
    resize: Optional[Tuple[int, int]],
    jpeg_quality: int,
    max_frames: Optional[int],
) -> int:
    """Decode color frames from ``bag_path`` into ``out_dir/frame_NNNNNN.jpg``.

    ``resize`` matches the default in ``streamer.py`` / ``bag_streamer.py``
    (960×288) so the resulting folder is a drop-in source for the existing
    SLAM intrinsics on the GPU. Pass ``None`` to keep original resolution.
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: pyrealsense2 is not installed. Install it with "
            "`pip install pyrealsense2` (Windows / x86_64 Linux wheels available).",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(bag_path):
        print(f"ERROR: bag file not found: {bag_path}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path, repeat_playback=False)
    cfg.enable_all_streams()

    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        print(f"ERROR: failed to open RealSense bag {bag_path}: {e}", file=sys.stderr)
        return 2

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

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
        except Exception:
            pass
        return 2

    src_fmt = color_profile.format()
    src_w, src_h = color_profile.width(), color_profile.height()
    print(
        f"Bag opened: color={src_w}x{src_h} @ {color_profile.fps()} FPS "
        f"({src_fmt}). Streams in file: {stream_names}"
    )
    out_abs = os.path.abspath(out_dir)
    print(
        f"Writing JPEG frames (quality={jpeg_quality}) to {out_abs}"
        + (f", resized to {resize[0]}x{resize[1]}." if resize else ", original size.")
    )

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    written = 0
    read = 0
    bytes_written = 0
    start = time.time()
    last_log = start

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=2000)
            if not ok or frames is None:
                break

            color = frames.get_color_frame()
            if not color:
                continue

            read += 1
            raw = np.asanyarray(color.get_data())
            if raw is None or raw.size == 0:
                continue

            try:
                bgr = _to_bgr(raw, src_fmt)
            except RuntimeError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                break

            if resize is not None:
                bgr = cv2.resize(bgr, resize)

            out_path = os.path.join(out_dir, f"frame_{written:06d}.jpg")
            if not cv2.imwrite(out_path, bgr, encode_param):
                print(f"WARNING: failed to write {out_path}")
                continue

            try:
                bytes_written += os.path.getsize(out_path)
            except OSError:
                pass

            written += 1

            now = time.time()
            if now - last_log >= 1.0:
                elapsed = now - start
                fps = written / max(elapsed, 1e-6)
                print(
                    f"written={written} fps={fps:.1f} "
                    f"total={bytes_written / (1024 * 1024):.1f} MiB"
                )
                last_log = now

            if max_frames is not None and written >= max_frames:
                print(f"Reached --max-frames={max_frames}, stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    elapsed = time.time() - start
    print(
        f"Done. read={read} written={written} "
        f"size={bytes_written / (1024 * 1024):.1f} MiB elapsed={elapsed:.1f}s"
    )
    print()
    print("Next steps:")
    print(f"  scp -r {out_dir}/ nano:~/droideka/")
    print(
        f"  python nano_client/zmq_video_sender.py --server-ip <gpu> "
        f"--source '{out_dir}/*.jpg' --resize none --fps 30"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the color stream of a RealSense .bag into a folder of JPEG "
            "frames so it can be replayed on the Nano with zmq_video_sender.py "
            "(no pyrealsense2 needed on the Nano)."
        ),
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to the RealSense .bag file.",
    )
    parser.add_argument(
        "--out",
        default="bag_frames",
        help="Output directory (created if missing). Default: bag_frames",
    )
    parser.add_argument(
        "--resize",
        default="960x288",
        help=(
            "Resize WIDTHxHEIGHT before encoding (default: 960x288 to match "
            "the intrinsics in live_slam.py). Use 'none' to keep the source "
            "resolution and resize on the Nano instead."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95, same as streamer.py).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after writing this many frames (sanity runs).",
    )
    args = parser.parse_args()

    resize = _parse_resize(args.resize)

    return extract(
        bag_path=args.bag,
        out_dir=args.out,
        resize=resize,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
