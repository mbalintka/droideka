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


def main(argv: Optional[list[str]] = None) -> int:
    # Placeholder — WasdTeleopNode and _keyboard_loop are added in Task 5.
    print("Teleop main() not yet implemented — see Task 5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
