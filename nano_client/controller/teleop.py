"""WASD keyboard teleoperation node for the droideka project.

Two public components:

* ``TeleopState``    — pure-Python keyboard → Twist state machine; no ROS or
                       termios dependency. Import this in tests.
* ``WasdTeleopNode`` — ROS 2 node that wraps TeleopState, defined inside
                       ``main()`` (same import-guard pattern as controller/node.py).

Entry point (Jetson Nano, interactive terminal):
    python -m nano_client.controller.teleop [--rate-hz 20] [--config cfg.json]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .config import VehicleConfig

# Milliseconds without a key press before both axes are zeroed automatically.
_DEADMAN_MS: int = 300


class TeleopState:
    """Keyboard → (v, delta) state machine.

    All public methods are safe to call from any thread. Python's GIL makes
    individual float/int assignments effectively atomic, so no explicit lock
    is needed for the three state variables used here.
    """

    def __init__(self, target_v: float, max_steer_rad: float) -> None:
        self.target_v = target_v
        self.max_steer_rad = max_steer_rad
        self._cmd_v: float = 0.0
        self._cmd_delta: float = 0.0
        # Initialised to now so the deadman does not fire before the first keypress.
        self._last_key_ns: int = time.time_ns()

    def handle_key(self, char: str) -> bool:
        """Update state for one character. Returns True if the operator quit."""
        if char.lower() == "q":
            return True
        self._last_key_ns = time.time_ns()
        if char.lower() == "w":
            self._cmd_v = self.target_v
        elif char.lower() == "s":
            self._cmd_v = -self.target_v
        elif char.lower() == "a":
            self._cmd_delta = -self.max_steer_rad
        elif char.lower() == "d":
            self._cmd_delta = self.max_steer_rad
        elif char == " ":
            self._cmd_v = 0.0
            self._cmd_delta = 0.0
        return False

    def compute_twist(self, now_ns: Optional[int] = None) -> tuple[float, float]:
        """Return ``(v_m_s, delta_rad)``, zeroing both if the deadman has fired."""
        if now_ns is None:
            now_ns = time.time_ns()
        age_ms = (now_ns - self._last_key_ns) / 1_000_000
        if age_ms > _DEADMAN_MS:
            self._cmd_v = 0.0
            self._cmd_delta = 0.0
        return self._cmd_v, self._cmd_delta


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WASD keyboard teleoperation node for droideka."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with VehicleConfig overrides.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="Control loop frequency for /cmd_vel publishes (default: 20 Hz).",
    )
    return parser.parse_args(argv)


def _keyboard_loop(
    state: TeleopState,
    stop_event,  # threading.Event
    on_state_change,  # Callable[[float, float], None]
) -> None:
    """Raw-terminal keyboard reader — runs on a daemon thread.

    Calls on_state_change(v, delta) after every key so the node can print
    the HUD without importing threading in the outer scope.
    """
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                char = sys.stdin.read(1)
                quit_requested = state.handle_key(char)
                v, delta = state._cmd_v, state._cmd_delta
                on_state_change(v, delta)
                if quit_requested:
                    stop_event.set()
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(argv: Optional[list[str]] = None) -> int:
    if not sys.stdin.isatty():
        print(
            "ERROR: stdin is not a TTY. "
            "Run this node in an interactive terminal (not piped/redirected).",
            file=sys.stderr,
        )
        return 1

    args = _parse_args(argv)
    cfg = VehicleConfig.from_file(args.config) if args.config else VehicleConfig()

    import math
    import threading

    # rclpy and geometry_msgs are imported inside main() so this module stays
    # importable on dev machines without ROS 2 installed.
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    class WasdTeleopNode(Node):
        """ROS 2 node: TeleopState → /cmd_vel at a fixed rate."""

        _HUD = (
            "\r[teleop] v={v:+.2f} m/s  δ={d:+.2f} rad ({dir:4s})"
            "  | W fwd  S rev  A/D steer  SPACE stop  Q quit"
        )

        def __init__(self) -> None:
            super().__init__("droideka_teleop")
            self._state = TeleopState(
                target_v=cfg.target_v,
                max_steer_rad=cfg.max_steer_rad,
            )
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_timer(1.0 / args.rate_hz, self._tick)
            self.get_logger().info(
                f"WASD teleop ready — rate={args.rate_hz:.0f} Hz, "
                f"v=±{cfg.target_v} m/s, "
                f"steer=±{math.degrees(cfg.max_steer_rad):.0f}°"
            )

        def _tick(self) -> None:
            v, delta = self._state.compute_twist()
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(delta)
            self.pub.publish(msg)

        def hud(self, v: float, delta: float) -> None:
            direction = (
                "LEFT" if delta < -1e-6 else "RGHT" if delta > 1e-6 else "STRT"
            )
            print(
                self._HUD.format(v=v, d=delta, dir=direction),
                end="",
                flush=True,
            )

        def publish_zero(self) -> None:
            self.pub.publish(Twist())

        @property
        def teleop_state(self) -> TeleopState:
            return self._state

    rclpy.init()
    node = WasdTeleopNode()
    stop_event = threading.Event()

    kb_thread = threading.Thread(
        target=_keyboard_loop,
        args=(node.teleop_state, stop_event, node.hud),
        daemon=True,
    )
    kb_thread.start()

    print(
        f"\n[teleop] Publishing /cmd_vel at {args.rate_hz:.0f} Hz. "
        "Drive carefully — kill this terminal before starting autonomy.\n"
    )

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        node.publish_zero()
        try:
            node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
