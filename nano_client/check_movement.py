import torch
import numpy as np

print("Analyzing data...")
data = torch.load("nano_live_map.pth", map_location="cpu", weights_only=False)
poses = data["poses"]

start_pos = poses[0][:3]
end_pos = poses[-1][:3]

# Euclidean distance in 3D
distance = np.linalg.norm(end_pos - start_pos)

print("-" * 30)
print(f"Start position: {start_pos}")
print(f"End position  : {end_pos}")
print(f"Distance (SLAM): {distance:.4f} units")
print("-" * 30)