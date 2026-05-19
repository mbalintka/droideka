"""Offline JPEG-sequence → MP4 stitcher.

Usage:
    python -m droideka_src.make_video \\
        --frames-dir runs/<stamp>/frames \\
        --output     runs/<stamp>/teach_run.mp4 \\
        [--fps 10]

The frames directory must contain *.jpg files recorded by live_slam.py
with --record-dir. Frames are stitched in sorted filename order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def make_video(frames_dir: Path, output: Path, fps: float = 10.0) -> int:
    """Stitch sorted JPEGs in frames_dir into an MP4 at output.

    Returns the number of frames written. Calls sys.exit(1) if frames_dir
    contains no .jpg files or the first frame cannot be decoded.
    """
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        print(f"ERROR: no .jpg files found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    first = cv2.imread(str(files[0]))
    if first is None:
        print(f"ERROR: could not read first frame {files[0]}", file=sys.stderr)
        sys.exit(1)

    h, w = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )

    count = 0
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[make_video] warning: skipping unreadable frame {f}")
            continue
        writer.write(frame)
        count += 1

    writer.release()
    print(f"[make_video] wrote {count} frames -> {output}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch recorded JPEG frames into an MP4 for visual review."
    )
    parser.add_argument(
        "--frames-dir", required=True, type=Path,
        help="Directory of *.jpg frames saved by live_slam.py --record-dir."
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output .mp4 path."
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Playback frame rate (default: 10, matches typical SLAM throughput)."
    )
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        print(f"ERROR: {args.frames_dir} is not a directory", file=sys.stderr)
        return 1

    make_video(args.frames_dir, args.output, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
