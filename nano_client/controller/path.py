"""Reference-path utilities for Pure Pursuit.

The on-disk format for a taught path is a NumPy .npy file containing an
(N, 3) float32/float64 array, with columns [x, y, theta] in meters/radians,
expressed in the SLAM world frame.

Public surface:
    - load_path(path) -> np.ndarray (N, 3)
    - save_path(out, points) -> Path
    - arc_length_resample(points, spacing) -> np.ndarray
    - find_lookahead_point(pose, path, lookahead, last_idx=0)
        -> (target_xy, segment_idx, distance_to_goal, cross_track_m)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def load_path(path: PathLike) -> np.ndarray:
    """Load a taught path as an (N, 3) float64 array of [x, y, theta]."""
    arr = np.load(str(path))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Path file {path} must be shape (N, 3) [x, y, theta]; got {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError(f"Path file {path} needs at least 2 waypoints.")
    return arr.astype(np.float64, copy=False)


def save_path(out: PathLike, points: np.ndarray) -> Path:
    """Persist an (N, 3) array of [x, y, theta] waypoints to .npy."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Points must be shape (N, 3) [x, y, theta]; got {points.shape}."
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), points)
    return out_path


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def arc_length_resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline so consecutive samples are ~`spacing` meters apart.

    `theta` is linearly interpolated (caller should keep `spacing` small enough
    that wrap-around between adjacent samples is not an issue, which is the
    case for any reasonable taught path).
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()

    # Cumulative arc length along the XY polyline.
    diffs = np.diff(pts[:, :2], axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= 0.0:
        return pts[:1].copy()

    n_samples = max(2, int(np.ceil(total / spacing)) + 1)
    s = np.linspace(0.0, total, n_samples)

    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    th = np.interp(s, cum, pts[:, 2])
    return np.stack([x, y, th], axis=1)


def _closest_point_on_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Project p onto segment (a, b). Returns (closest_point, t) with t in [0, 1]."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    return a + t * ab, t


def find_lookahead_point(
    pose_xy: np.ndarray,
    path: np.ndarray,
    lookahead: float,
    last_idx: int = 0,
) -> Tuple[np.ndarray, int, float, float]:
    """Find the lookahead target on `path`.

    Strategy:
        1) Find the closest point on the path (only searching segments at or
           after `last_idx`, which monotonically advances along the path to
           avoid back-tracking when the path crosses itself).
        2) Walk forward from there until the cumulative distance from the
           closest point exceeds `lookahead`, then linearly interpolate the
           exact target on the crossing segment.
        3) If the end of the path is reached before exceeding `lookahead`,
           return the last waypoint as the target.

    Args:
        pose_xy: shape (2,) current vehicle position [x, y].
        path:    shape (N, 3) reference path.
        lookahead: lookahead distance in meters.
        last_idx: lower bound for the closest-segment search (advances each
                  call to prevent latching onto an earlier loop of the path).

    Returns:
        target_xy:        shape (2,) lookahead point in world coords.
        segment_idx:      index of the segment whose start is the new
                          monotonic lower-bound for the next call.
        distance_to_goal: straight-line distance from pose to the final
                          waypoint of the path (used by the goal-reached check).
        cross_track_m:    perpendicular distance from the pose to the closest
                          point on the path (the cross-track error, useful for
                          telemetry / quality scoring of a run).
    """
    pose_xy = np.asarray(pose_xy, dtype=np.float64).reshape(2)
    xy = path[:, :2]
    n = xy.shape[0]
    last_idx = max(0, min(int(last_idx), n - 2))

    # ---- 1) closest segment (monotonic) ----
    best_seg = last_idx
    best_pt = xy[last_idx].copy()
    best_d2 = float(np.sum((pose_xy - best_pt) ** 2))
    for i in range(last_idx, n - 1):
        cp, _ = _closest_point_on_segment(pose_xy, xy[i], xy[i + 1])
        d2 = float(np.sum((pose_xy - cp) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_pt = cp
            best_seg = i

    cross_track_m = float(np.sqrt(best_d2))

    # ---- 2) walk forward by `lookahead` ----
    remaining = float(lookahead)
    cur = best_pt
    target = xy[-1].copy()
    found = False
    for i in range(best_seg, n - 1):
        seg_end = xy[i + 1]
        seg_vec = seg_end - cur
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= 1e-9:
            cur = seg_end
            continue
        if seg_len >= remaining:
            target = cur + (remaining / seg_len) * seg_vec
            found = True
            break
        remaining -= seg_len
        cur = seg_end

    if not found:
        target = xy[-1].copy()

    distance_to_goal = float(np.linalg.norm(pose_xy - xy[-1]))
    return target, best_seg, distance_to_goal, cross_track_m
