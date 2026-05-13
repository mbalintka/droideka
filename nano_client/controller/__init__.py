"""On-Nano control layer for the droideka project.

This package contains the path-following stack that consumes SLAM poses
(from the GPU server, over ZeroMQ) and emits Ackermann commands on a
ROS 2 topic. See README.md in this folder for the wire formats and CLI
entrypoints.
"""

from .config import VehicleConfig
from .pure_pursuit import PurePursuit, Pose

__all__ = ["VehicleConfig", "PurePursuit", "Pose"]
