import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

print("Loading map...")
data = torch.load("nano_live_map.pth", map_location="cpu", weights_only=False)

poses = data["poses"] 
images = data["images"] 
disps = data["disps"] 
intrinsics = data["intrinsics"] 

pcd_combined = o3d.geometry.PointCloud()
trajectory = []

print("Generating 3D reconstruction using Open3D...")

# Helper: convert numpy/torch scalar-like values to plain Python floats.
# Open3D constructors do not accept torch scalar tensors directly.
def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        # torch scalar tensor -> python number
        return float(x.item())
    # numpy scalars / python numbers
    return float(x)


# Process every 2nd frame
for i in range(0, len(poses), 2):
    pose_i = poses[i]
    intr_i = intrinsics[i]

    # Ensure we never pass torch scalar tensors into numpy/Open3D APIs
    tx, ty, tz, qx, qy, qz, qw = (_to_float(v) for v in pose_i)
    fx, fy, cx, cy = (_to_float(v) for v in intr_i)
    
    # 1) Compute depth from DROID inverse depth (disparity-like) output
    disp = disps[i]
    valid = disp > 0.05
    depth = np.zeros_like(disp)
    depth[valid] = 1.0 / disp[valid]
    
    # Remove near-field noise and far range (0.5m - 35m)
    depth[(depth < 0.5) | (depth > 35.0)] = 0.0

    # 2) Normalize the RGB image to Open3D-friendly uint8
    img = np.transpose(images[i], (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = (img * 255).astype(np.uint8)
    
    # Open3D expects contiguous memory
    img = np.ascontiguousarray(img)
    depth = np.ascontiguousarray(depth.astype(np.float32))

    # 3) Create Open3D RGBD object
    color_o3d = o3d.geometry.Image(img)
    depth_o3d = o3d.geometry.Image(depth)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=35.0, convert_rgb_to_intensity=False
    )

    # 4) Camera intrinsics
    H, W = disp.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(W), int(H), fx, fy, cx, cy)

    # 5) Generate local point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # 6) Transform into world coordinates using the DROID-SLAM pose
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    pcd.transform(T)
    pcd_combined += pcd
    
    trajectory.append([tx, ty, tz])

# 7) Flip axes so the export is oriented nicely in common viewers (mirror Y and Z)
flip_transform = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])
pcd_combined.transform(flip_transform)

# 8) Add trajectory points (colored red)
path_points = o3d.geometry.PointCloud()
path_points.points = o3d.utility.Vector3dVector(trajectory)
path_points.paint_uniform_color([1, 0, 0])
path_points.transform(flip_transform)

print("Denoising point cloud...")
# Remove isolated outliers
pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.1)
cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_combined = pcd_combined.select_by_index(ind)

# Save
final_pcd = pcd_combined + path_points
output_file = "KITTI_UTCA_VEGLEGES.ply"
o3d.io.write_point_cloud(output_file, final_pcd)
print(f"SUCCESS: Wrote point cloud to '{output_file}'.")