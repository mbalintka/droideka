"""Export a DROID-SLAM .pth reconstruction to a colored PLY (Open3D).

Set DROID_SLAM_ROOT to your DROID-SLAM repository root so ``droid_backends``
and ``lietorch`` resolve when they are not on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import os
import sys


def _ensure_droid_slam_on_path() -> None:
    root = os.environ.get("DROID_SLAM_ROOT", "").strip()
    if not root:
        return
    root = os.path.abspath(os.path.expanduser(root))
    for sub in ("", "droid_slam"):
        p = os.path.join(root, sub) if sub else root
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_ensure_droid_slam_on_path()

import torch
import droid_backends
import open3d as o3d
import numpy as np

from lietorch import SE3


def export_reconstruction(
    filename: str,
    output_file: str,
    *,
    filter_thresh: float = 0.005,
    min_count: int = 1,
    disp_mean_ratio: float = 0.05,
    voxel_size: float = 0.0,
    outlier_std_ratio: float | None = None,
    outlier_nb_neighbors: int = 30,
) -> None:
    print(f"Loading reconstruction: {filename}")
    reconstruction_blob = torch.load(filename)

    images = torch.as_tensor(reconstruction_blob["images"]).cuda()
    disps = torch.as_tensor(reconstruction_blob["disps"]).cuda()
    poses = torch.as_tensor(reconstruction_blob["poses"]).cuda()

    n_keyframes = len(images)
    h, w = int(disps.shape[-2]), int(disps.shape[-1])
    print(f"Keyframes: {n_keyframes}  depth maps: {w}x{h}  ({w * h * n_keyframes:,} depth samples)")

    # disps_up from live_slam is full resolution; stored intrinsics are 1/8 scale.
    intrinsics = 8 * torch.as_tensor(reconstruction_blob["intrinsics"]).cuda()

    disps_contig = disps.contiguous()
    poses_inv_data = SE3(poses).inv().data.contiguous()
    intrinsics_contig = intrinsics[0].contiguous()

    index = torch.arange(len(images), device="cuda").contiguous()
    thresh = (filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))).contiguous()

    print("Projecting depth maps to 3D (GPU)...")
    points = droid_backends.iproj(poses_inv_data, disps_contig, intrinsics_contig)
    colors = (images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0).contiguous()

    print("Multi-view depth consistency filter (GPU)...")
    poses_contig = poses.contiguous()
    counts = droid_backends.depth_filter(
        poses_contig, disps_contig, intrinsics_contig, index, thresh
    )

    disp_floor = disp_mean_ratio * disps_contig.mean()
    mask = (counts >= min_count) & (disps_contig > disp_floor)

    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()
    print(
        f"After depth filter: {len(points_np):,} points "
        f"(min_count={min_count}, disp_mean_ratio={disp_mean_ratio}, "
        f"filter_threshold={filter_thresh})"
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    if voxel_size > 0.0:
        before = len(point_cloud.points)
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        print(f"Voxel downsample ({voxel_size} m): {before:,} -> {len(point_cloud.points):,} points")

    if outlier_std_ratio is not None:
        before = len(point_cloud.points)
        _, ind = point_cloud.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )
        point_cloud = point_cloud.select_by_index(ind)
        print(
            f"Statistical outlier removal (std_ratio={outlier_std_ratio}): "
            f"{before:,} -> {len(point_cloud.points):,} points"
        )

    print("Adding camera trajectory (red)...")
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()
    cam_positions = pose_mats[:, :3, 3]

    cam_cloud = o3d.geometry.PointCloud()
    cam_cloud.points = o3d.utility.Vector3dVector(cam_positions)
    cam_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    final_pcd = point_cloud + cam_cloud

    # Flip Y/Z so the export opens upright in CloudCompare / Open3D.
    flip_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    final_pcd.transform(flip_transform)

    o3d.io.write_point_cloud(output_file, final_pcd)
    print(f"Saved {len(point_cloud.points):,} map points + {n_keyframes} trajectory points -> {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a DROID-SLAM .pth reconstruction to a colored PLY.",
    )
    parser.add_argument("filename", type=str, help="Path to the .pth file")
    parser.add_argument("--output", type=str, default="map.ply", help="Output PLY path")
    parser.add_argument(
        "--filter-threshold",
        "--filter_threshold",
        dest="filter_threshold",
        type=float,
        default=0.005,
        help=(
            "Multi-view depth consistency threshold (default: 0.005). "
            "Increase slightly (e.g. 0.008) for a denser but noisier cloud."
        ),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help=(
            "Minimum number of agreeing views per point (default: 1 = densest). "
            "Use 2-3 for cleaner output."
        ),
    )
    parser.add_argument(
        "--disp-mean-ratio",
        type=float,
        default=0.05,
        help=(
            "Drop pixels with disparity below this fraction of the mean "
            "(default: 0.05). Lower = denser, includes farther points."
        ),
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help=(
            "Voxel downsample size in metres (default: 0 = disabled). "
            "Try 0.03-0.05 only if the cloud is too heavy for your viewer."
        ),
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=None,
        help=(
            "If set, run statistical outlier removal with this std_ratio "
            "(e.g. 2.5). Default: disabled for maximum density."
        ),
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=30,
        help="Neighbours for statistical outlier removal (default: 30).",
    )
    args = parser.parse_args()

    export_reconstruction(
        args.filename,
        args.output,
        filter_thresh=args.filter_threshold,
        min_count=args.min_count,
        disp_mean_ratio=args.disp_mean_ratio,
        voxel_size=args.voxel_size,
        outlier_std_ratio=args.outlier_std_ratio,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
