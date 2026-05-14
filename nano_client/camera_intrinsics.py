"""Pinhole camera intrinsics helpers (shared by streamers and tools)."""

from __future__ import annotations


def scale_pinhole_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale (fx, fy, cx, cy) when resizing from src_wh to dst_wh (width, height).

    Matches independent axis scaling used by ``cv2.resize(..., dst_wh)``.
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    if sw <= 0 or sh <= 0:
        raise ValueError(f"invalid source size: {src_wh}")
    sx = dw / float(sw)
    sy = dh / float(sh)
    return fx * sx, fy * sy, cx * sx, cy * sy
