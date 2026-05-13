"""
path_from_reconstruction.py
---------------------------
Extracts a 2D ground-plane path from a DROID-SLAM reconstruction (.pth) and
saves it as a NumPy array ready for PurePursuitController.

SLAM world-frame convention (OpenCV / camera axes):
    X  →  right
    Y  ↓  down   (vertical axis, dropped for 2D navigation)
    Z  →  forward

Ground plane used here: X–Z.
path_xy columns: [0] = SLAM X,  [1] = SLAM Z
yaw convention:  0 rad = heading along +X,  CCW positive
                 (atan2 of the forward vector projected onto the X–Z plane)

Output files (saved next to the .pth by default):
    path_xy.npy   – float64 (N, 2)  positions in SLAM X–Z plane
    path_yaw.npy  – float64 (N,)    heading per waypoint [rad]

Requires ``lietorch`` from your DROID-SLAM environment (pip/conda install).

Usage:
    python path_from_reconstruction.py nano_live_map.pth
    python path_from_reconstruction.py nano_live_map.pth --output_dir /tmp --min_step 0.05 --smooth 5
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from lietorch import SE3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_raw_path(poses_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-frame camera positions and headings from DROID pose tensor.

    Args:
        poses_tensor: (N, 7) float32 on any device — DROID world-to-camera
                      7-vectors [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        positions_xz : (N, 2) float64 — camera X and Z in SLAM world frame.
        yaws_rad      : (N,)   float64 — heading angle in X–Z plane [rad].
    """
    # SE3(poses).inv() = camera-to-world transform T_wc
    # .matrix() shape: (N, 4, 4)
    T_wc = SE3(poses_tensor).inv().matrix().cpu().numpy().astype(np.float64)  # (N,4,4)

    # Camera position in world frame: last column of T_wc
    positions_xyz = T_wc[:, :3, 3]          # (N, 3)  [X, Y, Z]
    positions_xz  = positions_xyz[:, [0, 2]] # (N, 2)  drop vertical Y

    # Camera forward direction in world frame:
    # The camera's local +Z axis = third column of the rotation block R_wc.
    # Project onto the X–Z ground plane and compute heading.
    forward_world = T_wc[:, :3, 2]          # (N, 3)  third column of R_wc
    yaws_rad = np.arctan2(
        forward_world[:, 2],  # Z component (forward)
        forward_world[:, 0],  # X component (right)
    ).astype(np.float64)      # (N,)

    return positions_xz, yaws_rad


def _remove_close_duplicates(
    positions_xz: np.ndarray, yaws_rad: np.ndarray, min_step_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop waypoints that are too close together (e.g. during SLAM keyframe bursts).

    Keeps the first point always; then only accepts a new point if its
    distance from the last kept point is >= min_step_m.

    Args:
        positions_xz : (N, 2)
        yaws_rad     : (N,)
        min_step_m   : minimum Euclidean distance between consecutive waypoints.

    Returns:
        filtered positions (M, 2) and yaws (M,), M <= N.
    """
    keep_idx = [0]
    for i in range(1, len(positions_xz)):
        d = float(np.linalg.norm(positions_xz[i] - positions_xz[keep_idx[-1]]))
        if d >= min_step_m:
            keep_idx.append(i)
    idx = np.array(keep_idx, dtype=int)
    return positions_xz[idx], yaws_rad[idx]


def _smooth_path(positions_xz: np.ndarray, window: int) -> np.ndarray:
    """Apply a centred moving-average (box filter) to smooth the path.

    Uses 'edge' padding so the endpoints do not shrink.
    The raw (unsmoothed) position is blended back at both ends over half
    the window width so the first and last waypoint stay close to their
    original position.

    Args:
        positions_xz : (N, 2)
        window       : number of samples in the moving average (odd preferred).

    Returns:
        smoothed positions (N, 2).
    """
    if window < 2 or len(positions_xz) <= window:
        return positions_xz.copy()

    w = int(window)
    kernel = np.ones(w, dtype=np.float64) / w
    # Pad edges so convolution output length == input length
    out = np.stack([
        np.convolve(positions_xz[:, c], kernel, mode="same") for c in range(2)
    ], axis=1)

    # Fix the boundary bias introduced by zero-padding in np.convolve "same".
    # Re-pad with edge values and recompute the first/last w//2 points.
    half = w // 2
    for c in range(2):
        col = positions_xz[:, c]
        padded = np.concatenate([np.full(half, col[0]), col, np.full(half, col[-1])])
        for i in range(half):
            out[i, c] = padded[i : i + w].mean()
        for i in range(len(positions_xz) - half, len(positions_xz)):
            out[i, c] = padded[i : i + w].mean()

    return out


def _recompute_yaw_from_path(positions_xz: np.ndarray) -> np.ndarray:
    """Recompute yaw from finite-difference tangents after smoothing.

    Uses forward differences except at the last point (backward difference).

    Args:
        positions_xz : (N, 2)

    Returns:
        yaws (N,) in radians.
    """
    yaws = np.empty(len(positions_xz), dtype=np.float64)
    for i in range(len(positions_xz) - 1):
        dx = positions_xz[i + 1, 0] - positions_xz[i, 0]
        dz = positions_xz[i + 1, 1] - positions_xz[i, 1]
        yaws[i] = math.atan2(dz, dx)
    yaws[-1] = yaws[-2]  # repeat last
    return yaws


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_path_from_poses(
    poses_tensor: torch.Tensor,
    min_step_m: float = 0.03,
    smooth_window: int = 7,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cleaned 2D path directly from a DROID-SLAM poses tensor.

    No disk I/O — intended for in-process use by ``live_slam.py`` when
    switching from teach to autonomous mode.

    Args:
        poses_tensor  : (N, 7) world-to-camera SE3 7-vectors
                        [tx, ty, tz, qx, qy, qz, qw]. CPU or CUDA.
        min_step_m    : minimum distance between consecutive waypoints after
                        deduplication. ~0.03 m (3 cm) works well.
        smooth_window : box-filter kernel size for smoothing (samples).
                        Set to 0 or 1 to skip smoothing.
        verbose       : print progress lines.

    Returns:
        path_xy  : (N, 2) float64 — waypoints in SLAM X–Z plane.
        path_yaw : (N,)   float64 — heading per waypoint [rad].
    """
    poses = torch.as_tensor(poses_tensor)
    n_raw = int(poses.shape[0])
    if verbose:
        print(f"  Raw keyframes: {n_raw}")

    # 1. Extract positions and headings
    pos_xz, yaws = _extract_raw_path(poses)

    # 2. Remove near-duplicate points
    if min_step_m > 0.0:
        pos_xz, yaws = _remove_close_duplicates(pos_xz, yaws, min_step_m)
        if verbose:
            print(
                f"  After deduplication (min_step={min_step_m} m): "
                f"{len(pos_xz)} waypoints"
            )

    # 3. Smooth the spatial coordinates
    if smooth_window >= 2:
        pos_xz = _smooth_path(pos_xz, smooth_window)
        # Recompute yaw from smoothed tangents for consistency
        yaws = _recompute_yaw_from_path(pos_xz)
        if verbose:
            print(f"  After smoothing (window={smooth_window}): path ready")

    if len(pos_xz) < 2:
        raise RuntimeError(
            f"Path has only {len(pos_xz)} waypoint(s) after cleaning — "
            "decrease min_step_m or use a longer recording."
        )

    return pos_xz.astype(np.float64), yaws.astype(np.float64)


def build_path(
    reconstruction_path: str,
    min_step_m: float = 0.03,
    smooth_window: int = 7,
    output_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DROID-SLAM .pth file and produce a cleaned 2D path.

    Thin disk-I/O wrapper around :func:`build_path_from_poses`.

    Args:
        reconstruction_path : path to nano_live_map.pth (or similar).
        min_step_m          : minimum distance between consecutive waypoints
                              after deduplication. ~0.03 m (3 cm) works well.
        smooth_window       : box-filter kernel size for smoothing (samples).
                              Set to 0 or 1 to skip smoothing.
        output_dir          : directory to write path_xy.npy and path_yaw.npy.
                              Defaults to the same directory as the .pth file.

    Returns:
        path_xy  : (N, 2) float64 — waypoints in SLAM X–Z plane.
        path_yaw : (N,)   float64 — heading per waypoint [rad].
    """
    print(f"Loading reconstruction: {reconstruction_path}")
    blob = torch.load(reconstruction_path, map_location="cpu")

    poses = torch.as_tensor(blob["poses"])  # (N, 7) world-to-camera

    pos_xz, yaws = build_path_from_poses(
        poses,
        min_step_m=min_step_m,
        smooth_window=smooth_window,
        verbose=True,
    )

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(reconstruction_path))

    path_xy_file  = os.path.join(output_dir, "path_xy.npy")
    path_yaw_file = os.path.join(output_dir, "path_yaw.npy")

    np.save(path_xy_file,  pos_xz)
    np.save(path_yaw_file, yaws)

    total_length = float(np.sum(np.linalg.norm(np.diff(pos_xz, axis=0), axis=1)))
    print(f"  Path length : {total_length:.2f} m  ({len(pos_xz)} waypoints)")
    print(f"  Saved path_xy  -> {path_xy_file}")
    print(f"  Saved path_yaw -> {path_yaw_file}")

    return pos_xz, yaws


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Pure Pursuit path from a DROID-SLAM reconstruction."
    )
    parser.add_argument(
        "reconstruction",
        type=str,
        help="Path to the DROID-SLAM .pth reconstruction file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write path_xy.npy / path_yaw.npy (default: same as input).",
    )
    parser.add_argument(
        "--min_step",
        type=float,
        default=0.03,
        help="Minimum distance (m) between consecutive waypoints (default: 0.03).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=7,
        help="Smoothing window size in samples (default: 7, set to 0 to disable).",
    )
    args = parser.parse_args()

    build_path(
        reconstruction_path=args.reconstruction,
        min_step_m=args.min_step,
        smooth_window=args.smooth,
        output_dir=args.output_dir,
    )
