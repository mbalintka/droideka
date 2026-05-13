"""GPU-side Pure Pursuit utilities (offline / replay use only).

The autonomous loop runs the Nano-side controller in
``nano_client/controller/pure_pursuit.py``. This module is kept on the GPU
side for two reasons:

1. ``live_slam.py`` imports :data:`CAMERA_OFFSET_FORWARD_M` and
   :func:`camera_pose_to_vehicle_pose` to transform SLAM camera poses into
   rear-axle poses before they go on the wire.
2. The :class:`PurePursuitController` class is useful for offline
   path-tracking experiments (replay a recorded ``.pth`` against a
   reference path without the Nano in the loop).

Do not wire the :class:`PurePursuitController` into the live control loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants for the RC buggy
# ---------------------------------------------------------------------------

# Wheelbase L (rear axle → front axle), adjustable on the car; use nominal.
BUGGY_WHEELBASE_M: float = 0.320

# Camera is mounted 200 mm behind the front axle.
# → camera is (wheelbase − 0.200) = 0.120 m ahead of the rear axle.
# The bicycle model's reference point is the REAR axle, so we must subtract
# this offset when converting a SLAM camera pose to a vehicle control pose.
CAMERA_OFFSET_FORWARD_M: float = BUGGY_WHEELBASE_M - 0.200   # = 0.120 m

# Camera height above ground (informational; not used in 2D control).
CAMERA_HEIGHT_M: float = 0.200

# Maximum Ackermann steering angle at the front wheels (conservative).
BUGGY_MAX_STEER_RAD: float = math.radians(30.0)

# "Close enough to goal" threshold — roughly one car-length.
BUGGY_GOAL_TOLERANCE_M: float = 0.50


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PurePursuitConfig:
    """Pure Pursuit controller parameters (kinematic bicycle model).

    Units:
      - distances in meters
      - angles in radians
      - speed in m/s

    Default values are tuned for the RC buggy (wheelbase 320 mm).
    Lookahead is speed-adaptive:
      L_d = clamp(lookahead_min_m + lookahead_gain_s * v,
                  lookahead_min_m, lookahead_max_m)
    A reasonable outdoor walking-pace range is 0.5 m/s … 2 m/s,
    giving L_d ≈ 0.50 … 1.10 m with the default gains.
    """

    wheelbase_m: float = BUGGY_WHEELBASE_M

    # Speed-adaptive lookahead
    lookahead_min_m: float = 0.50    # minimum lookahead (standing still)
    lookahead_max_m: float = 2.00    # cap at moderate RC speed
    lookahead_gain_s: float = 0.60   # seconds of look-ahead time (Ld += gain*v)

    # Safety / stability
    max_steer_rad: float = BUGGY_MAX_STEER_RAD
    goal_tolerance_m: float = BUGGY_GOAL_TOLERANCE_M


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _as_path_xy(path_xy: np.ndarray) -> np.ndarray:
    """Validate and return path as float64 Nx2 array."""
    arr = np.asarray(path_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"path_xy must have shape (N,2), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("path_xy must contain at least 2 points")
    return arr


def _segment_projection(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Project point p onto segment ab.

    Returns:
      - proj: closest point on the segment
      - t: segment interpolation in [0,1]
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return proj, t


def _closest_point_on_polyline(
    path_xy: np.ndarray, position_xy: np.ndarray
) -> Tuple[np.ndarray, int, float, float]:
    """Find closest point on a polyline to a position.

    Returns:
      - closest_xy: closest point on the polyline
      - seg_idx: segment index i for segment [i, i+1]
      - seg_t: interpolation in [0,1] on that segment
      - dist_m: Euclidean distance to closest point
    """
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    best_d2 = float("inf")
    best_xy = None
    best_i = 0
    best_t = 0.0
    for i in range(path_xy.shape[0] - 1):
        a = path_xy[i]
        b = path_xy[i + 1]
        proj, t = _segment_projection(p, a, b)
        d2 = float(np.dot(p - proj, p - proj))
        if d2 < best_d2:
            best_d2 = d2
            best_xy = proj
            best_i = i
            best_t = t
    assert best_xy is not None
    return best_xy, best_i, best_t, float(math.sqrt(best_d2))


def _advance_along_polyline(
    path_xy: np.ndarray, seg_idx: int, seg_t: float, distance_m: float
) -> Tuple[np.ndarray, int, float]:
    """Move forward along the polyline by distance_m starting from (seg_idx, seg_t).

    Returns:
      - target_xy: point on polyline after advancing
      - out_seg_idx, out_seg_t: location on polyline for subsequent calls
    """
    if distance_m <= 0.0:
        a = path_xy[seg_idx]
        b = path_xy[seg_idx + 1]
        return a + float(seg_t) * (b - a), seg_idx, float(seg_t)

    i = int(seg_idx)
    t = float(seg_t)

    # Current point on segment
    a = path_xy[i]
    b = path_xy[i + 1]
    cur = a + t * (b - a)
    remaining = float(distance_m)

    while True:
        a = path_xy[i]
        b = path_xy[i + 1]
        seg_vec = b - a
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= 1e-12:
            if i >= path_xy.shape[0] - 2:
                return b.copy(), i, 1.0
            i += 1
            t = 0.0
            cur = path_xy[i].copy()
            continue

        # Distance from current point to end of segment
        end = b
        dist_to_end = float(np.linalg.norm(end - cur))
        if remaining <= dist_to_end + 1e-12:
            # Advance within this segment
            dir_vec = (end - cur) / max(dist_to_end, 1e-12)
            target = cur + remaining * dir_vec
            # Convert back to (i, t) within segment
            # t = dot(target-a, seg_vec) / |seg_vec|^2
            t_new = float(np.dot(target - a, seg_vec) / max(seg_len * seg_len, 1e-12))
            t_new = float(np.clip(t_new, 0.0, 1.0))
            return target, i, t_new

        # Consume segment and move to next
        remaining -= dist_to_end
        if i >= path_xy.shape[0] - 2:
            return end.copy(), i, 1.0
        i += 1
        t = 0.0
        cur = path_xy[i].copy()


def camera_pose_to_vehicle_pose(
    cam_x: float,
    cam_y: float,
    yaw_rad: float,
    cam_offset_fwd_m: float = CAMERA_OFFSET_FORWARD_M,
) -> Tuple[float, float]:
    """Convert 2D SLAM camera position to the rear-axle position.

    The bicycle model's reference point is the rear axle.  The camera sits
    cam_offset_fwd_m ahead of the rear axle along the vehicle's heading.
    To recover the rear axle we simply walk backwards:

        rear_axle = camera_pos − cam_offset_fwd_m * [cos(yaw), sin(yaw)]

    Args:
      cam_x, cam_y:       Camera position in the SLAM world frame.
      yaw_rad:            Vehicle heading (0 = +x, CCW positive).
      cam_offset_fwd_m:   How far the camera is ahead of the rear axle (m).

    Returns:
      (rear_x, rear_y): Rear-axle position in the same world frame.
    """
    rear_x = cam_x - cam_offset_fwd_m * math.cos(yaw_rad)
    rear_y = cam_y - cam_offset_fwd_m * math.sin(yaw_rad)
    return float(rear_x), float(rear_y)


class PurePursuitController:
    def __init__(self, cfg: PurePursuitConfig):
        if cfg.wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be > 0")
        if cfg.lookahead_min_m <= 0.0 or cfg.lookahead_max_m <= 0.0:
            raise ValueError("lookahead distances must be > 0")
        if cfg.lookahead_min_m > cfg.lookahead_max_m:
            raise ValueError("lookahead_min_m must be <= lookahead_max_m")
        self.cfg = cfg

        # Internal "progress" state to avoid jumping backwards on noisy poses.
        self._seg_idx: int = 0
        self._seg_t: float = 0.0

    def reset(self) -> None:
        self._seg_idx = 0
        self._seg_t = 0.0

    def compute_lookahead_m(self, speed_mps: float) -> float:
        ld = self.cfg.lookahead_min_m + self.cfg.lookahead_gain_s * max(0.0, float(speed_mps))
        return float(np.clip(ld, self.cfg.lookahead_min_m, self.cfg.lookahead_max_m))

    def compute_control(
        self,
        *,
        path_xy: np.ndarray,
        position_xy: np.ndarray,
        yaw_rad: float,
        speed_mps: float,
    ) -> Tuple[float, np.ndarray, bool]:
        """Compute steering command to follow a 2D path.

        Args:
          path_xy: Nx2 polyline in world frame.
          position_xy: vehicle position [x,y] in world frame.
          yaw_rad: vehicle heading in world frame (0 along +x, CCW positive).
          speed_mps: current speed (used only for lookahead selection).

        Returns:
          steer_rad: steering angle command (clipped to cfg.max_steer_rad).
          target_xy: the lookahead target point on the path in world frame.
          reached_goal: True if within cfg.goal_tolerance_m of final waypoint.
        """
        path = _as_path_xy(path_xy)
        pos = np.asarray(position_xy, dtype=np.float64).reshape(2)
        yaw = float(yaw_rad)

        goal_xy = path[-1]
        reached_goal = float(np.linalg.norm(goal_xy - pos)) <= self.cfg.goal_tolerance_m

        # Find closest point on path, but prevent "going backwards" by starting search
        # near our current progress (simple and robust for real-time tracking).
        # If path is short, this still searches all segments.
        search_start = max(0, min(self._seg_idx - 5, path.shape[0] - 2))
        search_end = min(path.shape[0] - 2, self._seg_idx + 25)
        if search_end <= search_start:
            search_start = 0
            search_end = path.shape[0] - 2

        # Local closest search
        best_d2 = float("inf")
        best_xy = None
        best_i = self._seg_idx
        best_t = self._seg_t
        p = pos
        for i in range(search_start, search_end + 1):
            a = path[i]
            b = path[i + 1]
            proj, t = _segment_projection(p, a, b)
            d2 = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_xy = proj
                best_i = i
                best_t = t
        if best_xy is None:
            # Fallback to global search (should be rare).
            best_xy, best_i, best_t, _ = _closest_point_on_polyline(path, pos)

        # Update progress state
        self._seg_idx = int(best_i)
        self._seg_t = float(best_t)

        # Choose lookahead target along the path
        lookahead_m = self.compute_lookahead_m(speed_mps)
        target_xy, self._seg_idx, self._seg_t = _advance_along_polyline(
            path, self._seg_idx, self._seg_t, lookahead_m
        )

        # Pure Pursuit geometry:
        # alpha = angle between vehicle heading and vector to target (in world frame)
        # curvature kappa = 2*sin(alpha) / Ld
        # steering delta = atan(L * kappa)
        dx = float(target_xy[0] - pos[0])
        dy = float(target_xy[1] - pos[1])
        target_angle = math.atan2(dy, dx)
        alpha = _wrap_to_pi(target_angle - yaw)

        ld = max(lookahead_m, 1e-3)
        curvature = (2.0 * math.sin(alpha)) / ld
        steer = math.atan(self.cfg.wheelbase_m * curvature)
        steer = float(np.clip(steer, -self.cfg.max_steer_rad, self.cfg.max_steer_rad))
        return steer, target_xy, bool(reached_goal)

