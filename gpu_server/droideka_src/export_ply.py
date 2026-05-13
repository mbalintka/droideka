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


def export_reconstruction(filename: str, output_file: str, filter_thresh=0.005):
    print("🧠 Térkép betöltése a DROID-SLAM gyári motorjával...")
    reconstruction_blob = torch.load(filename)

    # --- 1. Megtartjuk a 100%-os felbontást! ---
    images = torch.as_tensor(reconstruction_blob["images"]).cuda()
    disps = torch.as_tensor(reconstruction_blob["disps"]).cuda()
    poses = torch.as_tensor(reconstruction_blob["poses"]).cuda()

    intrinsics = 8 * torch.as_tensor(reconstruction_blob["intrinsics"]).cuda()

    disps_contig = disps.contiguous()
    poses_inv_data = SE3(poses).inv().data.contiguous()
    intrinsics_contig = intrinsics[0].contiguous()

    index = torch.arange(len(images), device="cuda").contiguous()
    thresh = (filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))).contiguous()

    print("⚙️ GPU gyorsított 3D rekonstrukció (Nagy felbontás)...")
    points = droid_backends.iproj(poses_inv_data, disps_contig, intrinsics_contig)
    colors = (images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0).contiguous()

    print("🧹 GPU mélység-szűrés...")
    poses_contig = poses.contiguous()
    counts = droid_backends.depth_filter(poses_contig, disps_contig, intrinsics_contig, index, thresh)

    # --- 2. Enyhébb szűrés a több pontért! ---
    mask = (counts >= 1) & (disps_contig > 0.10 * disps_contig.mean())

    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    # Pontfelhő építése
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    # --- 3. ÚJ: Open3D Utómunka (A Porszívó) ---
    print("🌪️ Open3D Utómunka: Lebegő zajok és por eltávolítása...")
    # Voxel downsample (Térbeli ritkítás az egyenletes felületekért)
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.15)

    # Statisztikai porszívó (Kiirtja a magányos/lebegő pontokat)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    point_cloud = point_cloud.select_by_index(ind)

    print("🚗 Kamera útvonal kinyerése...")
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()
    cam_positions = pose_mats[:, :3, 3]

    cam_cloud = o3d.geometry.PointCloud()
    cam_cloud.points = o3d.utility.Vector3dVector(cam_positions)
    cam_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Tiszta piros útvonal

    final_pcd = point_cloud + cam_cloud

    # CloudCompare-re igazítás
    flip_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    final_pcd.transform(flip_transform)

    print("💾 Fájl mentése...")
    o3d.io.write_point_cloud(output_file, final_pcd)
    print(f"✅ SIKER! A tisztított, tökéletes gyári 3D térkép elmentve ide: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to the .pth file")
    parser.add_argument("--output", type=str, default="kitti_tokeletes.ply", help="output file name")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    args = parser.parse_args()

    export_reconstruction(args.filename, args.output, args.filter_threshold)
