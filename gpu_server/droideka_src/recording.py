"""Frame recording helpers — pure file I/O, no DROID-SLAM dependency.

Kept in its own module so tests can import it without triggering the heavy
torch/droid imports at the top of live_slam.py.
"""
from __future__ import annotations

from pathlib import Path


def record_frame(
    record_dir: Path,
    frame_id: int,
    jpeg_bytes: bytes,
    active: bool,
) -> None:
    """Write jpeg_bytes to record_dir/<frame_id:06d>.jpg when active.

    Errors are caught and printed; recording is best-effort and must never
    interrupt the SLAM loop.
    """
    if not active:
        return
    try:
        (record_dir / f"{frame_id:06d}.jpg").write_bytes(jpeg_bytes)
        if frame_id % 50 == 0:
            print(f"[record] frame {frame_id:06d} -> {record_dir}")
    except OSError as e:
        print(f"[record] warning: could not write frame {frame_id}: {e}")
