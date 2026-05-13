"""Vehicle and controller configuration for Pure Pursuit.

All values are SI units (meters, seconds, radians). The defaults below are
sensible starting points for a small (~1/10 scale) RC car driving indoors;
override them via a YAML/JSON file or by constructing VehicleConfig directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class VehicleConfig:
    # --- Vehicle geometry ---
    # Distance between front and rear axles (kinematic bicycle wheelbase).
    # 0.32 m matches the HPI Trophy Buggy Flux (#107016) factory spec.
    wheelbase: float = 0.32

    # Hard limit on steering angle commanded to the servo.
    max_steer_rad: float = math.radians(30.0)

    # --- Speed schedule ---
    # Target forward velocity while following the path.
    target_v: float = 0.6

    # --- Lookahead schedule ---
    # Pure Pursuit lookahead distance: Ld = clip(gain*v + min, min, max).
    lookahead_min: float = 0.30
    lookahead_max: float = 1.50
    lookahead_gain: float = 0.40

    # --- Safety ---
    # If no fresh pose has arrived within this many seconds, the follow loop
    # publishes a zero-velocity stop until poses resume.
    pose_timeout_s: float = 0.5

    # Distance (m) to the final waypoint at which we declare the goal reached.
    goal_tolerance_m: float = 0.20

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "VehicleConfig":
        """Load a config from a JSON file. Unknown keys are ignored."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} must contain a JSON object.")
        # Only keep keys that are actually fields on this dataclass; this keeps
        # forward-compat config files from blowing up the constructor.
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        return asdict(self)
