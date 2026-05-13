"""Pure Pursuit path-tracking controller (kinematic bicycle model).

I/O free: this module operates only on a current pose and a reference path
(NumPy array). Network/ROS plumbing lives in the sibling modules.

References
----------
Coulter, R. C. (1992). "Implementation of the Pure Pursuit Path Tracking
Algorithm." Carnegie Mellon University, Tech. Report CMU-RI-TR-92-01.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import VehicleConfig
from .path import find_lookahead_point


@dataclass
class Pose:
    """Planar vehicle pose in the SLAM world frame (SI units)."""

    x: float
    y: float
    theta: float  # yaw, radians, right-hand rule (z-up)


@dataclass
class ControlCommand:
    """Output of one Pure Pursuit step."""

    v: float          # commanded forward velocity (m/s)
    delta: float      # commanded steering angle (rad), positive = left
    done: bool        # True once the goal tolerance has been reached
    target_xy: np.ndarray  # the lookahead point in world frame (debug/log)
    distance_to_goal: float
    seg_idx: int = -1            # closest-segment index on the path (debug/log)
    cross_track_m: float = 0.0   # perpendicular distance to the path (debug/log)


class PurePursuit:
    """Pure Pursuit controller with a velocity-scheduled lookahead.

    The controller is stateful only in `_last_seg_idx`, which monotonically
    advances along the path so we don't re-latch onto an earlier loop when
    the path crosses itself.
    """

    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg
        self._last_seg_idx: int = 0
        self._done: bool = False

    def reset(self) -> None:
        self._last_seg_idx = 0
        self._done = False

    def _lookahead_for(self, v: float) -> float:
        cfg = self.cfg
        ld = cfg.lookahead_gain * abs(v) + cfg.lookahead_min
        return float(min(max(ld, cfg.lookahead_min), cfg.lookahead_max))

    def compute(self, pose: Pose, path: np.ndarray) -> ControlCommand:
        """Compute the next (v, delta) command.

        Args:
            pose: current vehicle pose in the SLAM world frame.
            path: (N, 3) array of [x, y, theta] waypoints.

        Returns:
            ControlCommand. When the goal is reached, v=0, delta=0, done=True
            and the controller latches in the done state until reset() is called.
        """
        cfg = self.cfg

        if self._done:
            return ControlCommand(
                v=0.0,
                delta=0.0,
                done=True,
                target_xy=path[-1, :2].copy(),
                distance_to_goal=float(
                    np.linalg.norm(np.array([pose.x, pose.y]) - path[-1, :2])
                ),
                seg_idx=self._last_seg_idx,
                cross_track_m=0.0,
            )

        v_cmd = cfg.target_v
        ld = self._lookahead_for(v_cmd)

        pose_xy = np.array([pose.x, pose.y], dtype=np.float64)
        target_xy, seg_idx, dist_to_goal, cross_track_m = find_lookahead_point(
            pose_xy, path, lookahead=ld, last_idx=self._last_seg_idx
        )
        self._last_seg_idx = seg_idx

        # Goal reached: latch a stop.
        if dist_to_goal <= cfg.goal_tolerance_m:
            self._done = True
            return ControlCommand(
                v=0.0, delta=0.0, done=True,
                target_xy=target_xy, distance_to_goal=dist_to_goal,
                seg_idx=seg_idx, cross_track_m=cross_track_m,
            )

        # Vector from rear axle (~vehicle origin) to lookahead point, expressed
        # in the vehicle's body frame (x forward, y left).
        dx = target_xy[0] - pose.x
        dy = target_xy[1] - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy

        # Effective lookahead distance (use actual chord length, not the
        # scheduled value, in case we clipped at the end of the path).
        ld_eff = math.hypot(local_x, local_y)
        if ld_eff < 1e-6:
            # Already at the lookahead point; coast straight at low gain.
            return ControlCommand(
                v=v_cmd, delta=0.0, done=False,
                target_xy=target_xy, distance_to_goal=dist_to_goal,
                seg_idx=seg_idx, cross_track_m=cross_track_m,
            )

        # Pure Pursuit curvature: kappa = 2 * y_local / Ld^2
        # (equivalent to 2 * sin(alpha) / Ld with alpha = atan2(y_local, x_local)).
        kappa = 2.0 * local_y / (ld_eff * ld_eff)

        # Ackermann steering for the kinematic bicycle: delta = atan(L * kappa).
        delta = math.atan(self.cfg.wheelbase * kappa)
        delta = max(-cfg.max_steer_rad, min(cfg.max_steer_rad, delta))

        # If the lookahead point is behind us (local_x < 0), reduce velocity to
        # avoid lunging past the start of a sharp turn.
        if local_x < 0.0:
            v_cmd *= 0.5

        return ControlCommand(
            v=v_cmd, delta=delta, done=False,
            target_xy=target_xy, distance_to_goal=dist_to_goal,
            seg_idx=seg_idx, cross_track_m=cross_track_m,
        )
