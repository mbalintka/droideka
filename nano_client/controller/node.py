"""ROS 2 autonomy node for the droideka project.

This is the Nano-side glue between:

* the GPU server (DROID-SLAM live pose stream over ZeroMQ), and
* the ``auto_control_ws/car_control_node`` which already subscribes to
  ``/cmd_vel`` (``geometry_msgs/msg/Twist``: ``linear.x`` = m/s,
  ``angular.z`` = steering [rad]).

Boot order
----------
1. REQ on ``tcp://<gpu>:<cmd_port>`` with ``{"cmd": "stop_teach"}`` -> receive
   a multipart reply ``[meta_json, path_bytes]`` packing an ``(N, 3)`` float64
   ``[x, y, theta]`` reference path.
2. Background thread: ZMQ SUB on ``tcp://<gpu>:<pose_port>`` receives live
   pose dicts ``{x, y, theta, frame_id, t_ns}`` and stores the latest one
   under a lock.
3. Create publisher on ``/cmd_vel`` (depth 10).
4. On a fixed timer (``--rate-hz``): snapshot the latest pose; if it is
   older than ``cfg.pose_timeout_s`` publish a zero Twist (the downstream
   ``car_control_node`` watchdog also catches this after ~1 s), otherwise
   run :class:`PurePursuit.compute` and publish the resulting Twist.
5. On goal-reached or shutdown: publish a zero Twist, close sockets,
   shutdown ``rclpy``.

The ``rclpy`` import is scoped inside :func:`main` so this module remains
importable on dev machines (Windows / CI) that do not have ROS 2 installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

from .config import VehicleConfig
from .pure_pursuit import ControlCommand, Pose, PurePursuit


# ---------------------------------------------------------------------------
# Path retrieval (REQ to the GPU server)
# ---------------------------------------------------------------------------


def request_path_from_gpu(
    gpu_host: str,
    cmd_port: int,
    min_step_m: Optional[float] = None,
    smooth_window: Optional[int] = None,
    timeout_s: float = 30.0,
) -> np.ndarray:
    """Ask the GPU server to switch to autonomous mode and return the path.

    Returns:
        (N, 3) float64 array ``[x, y, theta]`` in the SLAM world frame.
    """
    payload: dict = {"cmd": "stop_teach"}
    if min_step_m is not None:
        payload["min_step_m"] = float(min_step_m)
    if smooth_window is not None:
        payload["smooth_window"] = int(smooth_window)

    context = zmq.Context.instance()
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    sock.connect(f"tcp://{gpu_host}:{cmd_port}")

    frames: list[bytes] = []
    try:
        sock.send_string(json.dumps(payload))
        frames = sock.recv_multipart()
    except zmq.Again as e:
        raise RuntimeError(
            f"No reply from GPU on tcp://{gpu_host}:{cmd_port} within {timeout_s}s. "
            "Is live_slam running there, and is port open (same checks as frames :5555)? "
            "If the server was updated recently, restart live_slam so REQ/REP framing matches."
        ) from e
    finally:
        sock.close(linger=0)

    if len(frames) < 2:
        raise RuntimeError(
            f"stop_teach reply had {len(frames)} frame(s), expected >= 2"
        )

    meta = json.loads(frames[0].decode("utf-8"))
    if not meta.get("ok", False):
        raise RuntimeError(f"GPU rejected stop_teach: {meta.get('error', 'unknown')}")

    shape = tuple(meta["shape"])
    if len(shape) != 2 or shape[1] != 3:
        raise RuntimeError(f"Unexpected path shape from GPU: {shape}")

    dtype = np.dtype(meta.get("dtype", "float64"))
    path = np.frombuffer(frames[1], dtype=dtype).reshape(shape).copy()
    path = path.astype(np.float64, copy=False)

    print(
        f"[autonomy] received path: {shape[0]} waypoints, "
        f"length={meta.get('total_length_m', float('nan')):.2f} m"
    )
    return path


# ---------------------------------------------------------------------------
# Background pose subscriber
# ---------------------------------------------------------------------------


@dataclass
class _LatestPose:
    pose: Optional[Pose]
    t_recv_ns: int


class PoseSubscriber:
    """Background ZMQ SUB that keeps the most recent pose under a lock."""

    def __init__(self, gpu_host: str, pose_port: int):
        self._endpoint = f"tcp://{gpu_host}:{pose_port}"
        self._lock = threading.Lock()
        self._latest = _LatestPose(pose=None, t_recv_ns=0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ctx = zmq.Context.instance()
        self._sock: Optional[zmq.Socket] = None

    def start(self) -> None:
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVHWM, 1)  # we only care about the freshest one
        self._sock.connect(self._endpoint)
        self._thread = threading.Thread(target=self._loop, name="PoseSub", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._sock is not None
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)

        while not self._stop.is_set():
            events = dict(poller.poll(timeout=200))
            if self._sock not in events:
                continue
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = json.loads(raw.decode("utf-8"))
                pose = Pose(
                    x=float(msg["x"]),
                    y=float(msg["y"]),
                    theta=float(msg["theta"]),
                )
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"[autonomy] bad pose message dropped: {e}")
                continue

            with self._lock:
                self._latest = _LatestPose(pose=pose, t_recv_ns=time.time_ns())

    def snapshot(self) -> _LatestPose:
        with self._lock:
            return _LatestPose(pose=self._latest.pose, t_recv_ns=self._latest.t_recv_ns)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


# ---------------------------------------------------------------------------
# ROS 2 entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Droideka autonomy node: SLAM pose -> Pure Pursuit -> /cmd_vel.",
    )
    parser.add_argument("--gpu-host", required=True, help="GPU server hostname / IP.")
    parser.add_argument("--pose-port", type=int, default=5556)
    parser.add_argument("--cmd-port", type=int, default=5557)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with VehicleConfig overrides.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help="Control loop frequency for /cmd_vel publishes.",
    )
    parser.add_argument(
        "--cmd-timeout-s",
        type=float,
        default=30.0,
        help="ZMQ timeout for the initial stop_teach request.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = (
        VehicleConfig.from_file(args.config)
        if args.config is not None
        else VehicleConfig()
    )
    print(f"[autonomy] vehicle config: {cfg.to_dict()}")

    print(
        f"[autonomy] requesting stop_teach from "
        f"tcp://{args.gpu_host}:{args.cmd_port} ..."
    )
    path = request_path_from_gpu(
        gpu_host=args.gpu_host,
        cmd_port=args.cmd_port,
        timeout_s=args.cmd_timeout_s,
    )

    sub = PoseSubscriber(gpu_host=args.gpu_host, pose_port=args.pose_port)
    sub.start()
    print(
        f"[autonomy] pose subscriber connected to "
        f"tcp://{args.gpu_host}:{args.pose_port}"
    )

    controller = PurePursuit(cfg)

    # rclpy / geometry_msgs are intentionally imported inside main() so this
    # module remains importable on dev machines without ROS 2 installed.
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    class AutonomyNode(Node):
        def __init__(self) -> None:
            super().__init__("droideka_autonomy")
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self._done_latched = False
            self._timer = self.create_timer(1.0 / float(args.rate_hz), self._tick)
            self.get_logger().info(
                f"Publishing /cmd_vel at {args.rate_hz:.1f} Hz; "
                f"path has {path.shape[0]} waypoints."
            )

        def _publish_zero(self) -> None:
            msg = Twist()
            self.pub.publish(msg)

        def _tick(self) -> None:
            if self._done_latched:
                self._publish_zero()
                return

            snap = sub.snapshot()
            now_ns = time.time_ns()
            if snap.pose is None:
                self._publish_zero()
                return

            age_s = (now_ns - snap.t_recv_ns) / 1e9
            if age_s > cfg.pose_timeout_s:
                self._publish_zero()
                self.get_logger().warn(
                    f"pose stale ({age_s:.2f}s > {cfg.pose_timeout_s:.2f}s) -> zero Twist",
                    throttle_duration_sec=1.0,
                )
                return

            cmd: ControlCommand = controller.compute(snap.pose, path)

            twist = Twist()
            twist.linear.x = float(cmd.v)
            twist.angular.z = float(cmd.delta)
            self.pub.publish(twist)

            if cmd.done:
                self._done_latched = True
                self._publish_zero()
                self.get_logger().info(
                    f"Goal reached (distance_to_goal={cmd.distance_to_goal:.3f} m); "
                    "publishing zero Twist."
                )

    rclpy.init()
    node = AutonomyNode()
    exit_code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; sending stop Twist.")
    except Exception as e:
        node.get_logger().error(f"unexpected error: {e}")
        exit_code = 1
    finally:
        try:
            # Final stop command before tearing things down.
            stop = Twist()
            node.pub.publish(stop)
        except Exception:
            pass
        try:
            node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        sub.stop()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
