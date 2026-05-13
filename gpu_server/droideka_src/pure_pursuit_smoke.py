import math
import os
import sys

# Allow `python /path/to/pure_pursuit_smoke.py` from any cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np

from pure_pursuit import PurePursuitConfig, PurePursuitController


def main() -> None:
    # Straight line path along +x
    path = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    cfg = PurePursuitConfig(wheelbase_m=0.32, lookahead_min_m=1.0, lookahead_max_m=1.0)
    ctrl = PurePursuitController(cfg)

    # Starting slightly above the line, heading along +x.
    pos = np.array([0.0, 0.5], dtype=np.float64)
    yaw = 0.0
    steer, target, reached = ctrl.compute_control(path_xy=path, position_xy=pos, yaw_rad=yaw, speed_mps=1.0)
    print("steer_rad:", steer)
    print("steer_deg:", steer * 180.0 / math.pi)
    print("target:", target)
    print("reached_goal:", reached)


if __name__ == "__main__":
    main()
