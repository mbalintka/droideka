"""Thread-safe JSONL telemetry logger for autonomy runs.

Writes one JSON record per line to ``<run_dir>/autonomy.jsonl``. Each record
has ``t_wall_ns`` (epoch nanoseconds) and ``t_mono_ns`` (process-monotonic
nanoseconds) so post-run analysis can both align with wall clock and measure
intervals robustly across clock skew.

Records produced
----------------
* ``start``           — emitted once at the top of ``main()`` with cfg / path stats.
* ``pose``            — emitted by the pose subscriber thread on every received pose.
* ``tick``            — emitted by the control loop on every Pure Pursuit step.
* ``goal_reached``    — emitted once when the controller latches done.
* ``pose_stale``      — emitted when a tick drops to zero Twist because of pose age.
* ``pose_unavailable``— emitted when a tick has no pose at all yet.
* ``error``           — emitted on caught exceptions from the autonomy loop.
* ``shutdown``        — emitted exactly once from ``finalize()`` with aggregate stats.

The pose subscriber thread and the ROS timer thread can both call into the
logger concurrently, so all writes go through a single ``threading.Lock``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


class RunLogger:
    """Append-only JSONL logger. Cheap, thread-safe, no external deps.

    Aggregate counters (``n_ticks``, ``n_poses``, min/max pose age, last
    distance-to-goal) are maintained on every call so :meth:`finalize` can
    emit a one-shot summary without re-reading the file.
    """

    def __init__(self, run_dir: Path):
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._run_dir / "autonomy.jsonl"
        self._lock = threading.Lock()
        self._fh = self._path.open("a", encoding="utf-8")
        self._closed = False

        # Aggregates for the shutdown summary.
        self._t0_mono_ns = time.monotonic_ns()
        self._t0_wall_ns = time.time_ns()
        self._n_ticks = 0
        self._n_poses = 0
        self._min_pose_age_s: Optional[float] = None
        self._max_pose_age_s: Optional[float] = None
        self._last_dist_to_goal_m: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        record.setdefault("t_wall_ns", time.time_ns())
        record.setdefault("t_mono_ns", time.monotonic_ns())
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            if self._closed:
                return
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()

    @staticmethod
    def _pose_dict(pose: Any) -> dict:
        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the underlying ``autonomy.jsonl`` file."""
        return self._path

    def log_event(self, kind: str, **fields: Any) -> None:
        """Emit a freeform event record (``kind`` is required)."""
        record = {"kind": str(kind)}
        record.update(fields)
        self._write(record)

    def log_start(
        self,
        cfg_dict: dict,
        path_meta: dict,
        n_waypoints: int,
        gpu_host: str,
        pose_port: int,
        cmd_port: int,
        rate_hz: float,
    ) -> None:
        self.log_event(
            "start",
            cfg=cfg_dict,
            path_meta=path_meta,
            n_waypoints=int(n_waypoints),
            gpu_host=str(gpu_host),
            pose_port=int(pose_port),
            cmd_port=int(cmd_port),
            rate_hz=float(rate_hz),
        )

    def log_pose(self, pose: Any, t_recv_ns: int) -> None:
        """Emit a ``pose`` record. Safe to call from the subscriber thread."""
        self._n_poses += 1
        self._write(
            {
                "kind": "pose",
                "t_recv_ns": int(t_recv_ns),
                "pose": self._pose_dict(pose),
            }
        )

    def log_tick(
        self,
        pose: Any,
        cmd: Any,
        pose_age_s: float,
    ) -> None:
        """Emit a ``tick`` record with the controller's full output."""
        self._n_ticks += 1
        age = float(pose_age_s)
        if self._min_pose_age_s is None or age < self._min_pose_age_s:
            self._min_pose_age_s = age
        if self._max_pose_age_s is None or age > self._max_pose_age_s:
            self._max_pose_age_s = age
        dist = float(getattr(cmd, "distance_to_goal", float("nan")))
        if dist == dist:  # not NaN
            self._last_dist_to_goal_m = dist

        target = getattr(cmd, "target_xy", None)
        if target is not None:
            target_xy = [float(target[0]), float(target[1])]
        else:
            target_xy = None

        self._write(
            {
                "kind": "tick",
                "pose": self._pose_dict(pose),
                "pose_age_s": age,
                "v": float(getattr(cmd, "v", 0.0)),
                "delta": float(getattr(cmd, "delta", 0.0)),
                "done": bool(getattr(cmd, "done", False)),
                "target_xy": target_xy,
                "seg_idx": int(getattr(cmd, "seg_idx", -1)),
                "cross_track_m": float(getattr(cmd, "cross_track_m", float("nan"))),
                "dist_to_goal_m": dist,
            }
        )

    def finalize(self, reason: str, extra: Optional[dict] = None) -> None:
        """Emit a final ``shutdown`` record and close the file.

        Idempotent: subsequent calls are no-ops, which makes it safe to invoke
        from a ``finally:`` block even after an explicit shutdown elsewhere.
        """
        with self._lock:
            if self._closed:
                return
        elapsed_ns = time.monotonic_ns() - self._t0_mono_ns
        summary = {
            "kind": "shutdown",
            "reason": str(reason),
            "n_ticks": int(self._n_ticks),
            "n_poses": int(self._n_poses),
            "min_pose_age_s": self._min_pose_age_s,
            "max_pose_age_s": self._max_pose_age_s,
            "final_dist_to_goal_m": self._last_dist_to_goal_m,
            "wall_duration_s": elapsed_ns / 1e9,
        }
        if extra:
            summary.update(extra)
        self._write(summary)

        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
