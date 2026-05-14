"""Grab one frame from a USB webcam and save it (Jetson / Linux V4L2 or Windows DirectShow)."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from typing import Optional

import cv2


def _try_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _open_capture(camera_index: int) -> cv2.VideoCapture:
    system = platform.system()
    if system == "Linux":
        return cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if system == "Windows":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a single frame from a USB camera (matches streamer backends).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=_try_int(os.environ.get("DROIDEKA_CAMERA_INDEX")) or 2,
        help="Camera index (default: 2 or env DROIDEKA_CAMERA_INDEX, same as streamer.py).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="snapshot.jpg",
        help="Output image path (.jpg or .png). Default: snapshot.jpg",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Frames to discard before capture (lets exposure settle). Default: 10.",
    )
    args = parser.parse_args()

    cap = _open_capture(args.camera_index)
    if not cap.isOpened():
        print(
            f"ERROR: Could not open camera index {args.camera_index}. "
            "Try another index (0–2) or run find_usb_camera.py on Linux.",
            file=sys.stderr,
        )
        return 1

    frame = None
    try:
        for _ in range(max(0, args.warmup)):
            cap.read()
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        print("ERROR: Failed to read a frame.", file=sys.stderr)
        return 1

    ok = cv2.imwrite(args.output, frame)
    if not ok:
        print(f"ERROR: cv2.imwrite failed for {args.output!r}.", file=sys.stderr)
        return 1

    h, w = frame.shape[:2]
    print(f"Saved {w}x{h} BGR frame to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
