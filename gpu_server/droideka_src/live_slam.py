"""Live DROID-SLAM over ZeroMQ (GPU server).

Expects DROID-SLAM Python packages importable from your environment, or set
``DROID_SLAM_ROOT`` to the DROID-SLAM repo root (same layout as upstream:
``droid/`` and ``droid_slam/`` under that root).

Optional: ``DROID_SLAM_WEIGHTS`` overrides the default ``--weights`` path.
"""
from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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

import argparse

import cv2
import numpy as np
import torch
import zmq
from droid import Droid


def _default_weights_path() -> str:
    w = os.environ.get("DROID_SLAM_WEIGHTS", "").strip()
    if w:
        return os.path.abspath(os.path.expanduser(w))
    root = os.environ.get("DROID_SLAM_ROOT", "").strip()
    if root:
        p = os.path.join(os.path.abspath(os.path.expanduser(root)), "droid.pth")
        return p
    return "droid.pth"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=_default_weights_path())
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", nargs="+", type=int, default=[288, 960])
    parser.add_argument("--disable_vis", action="store_true", default=True)
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    parser.add_argument("--upsample", action="store_true", default=True)
    parser.add_argument(
        "--reconstruction_path",
        default=os.path.join(CURRENT_DIR, "nano_live_map.pth"),
    )
    return parser.parse_args()


def main():
    args = get_args()

    print("🧠 DROID-SLAM neurális hálózat betöltése a GPU-ra...")
    droid = Droid(args)
    print("✅ DROID-SLAM készen áll!")

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    print("📡 Várakozás a videófolyamára az 5555-ös porton...")

    # Új intrinsics a 960x288-as felbontáshoz (minden érték x1.5)
    intrinsics = torch.tensor([555.3, 552.0, 469.0, 142.2])
    frame_id = 0

    try:
        while True:
            message = socket.recv()
            npimg = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(npimg, 1)

            if frame is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
                image_tensor = image_tensor.unsqueeze(0)

                droid.track(frame_id, image_tensor, intrinsics=intrinsics)

                if frame_id % 10 == 0:
                    print(f"📍 SLAM frissítve! Feldolgozott képkockák: {frame_id}")
                frame_id += 1

    except KeyboardInterrupt:
        print("\n🛑 Vétel megszakítva. A 3D térkép véglegesítése folyamatban (Bundle Adjustment)...")
    finally:
        if frame_id > 0:
            print("⏳ Kérlek várj, ez eltarthat 1-2 percig...")
            try:
                droid.terminate()
            except Exception as e:
                print(f"⚠️ Befejezési optimalizáció hiba: {e}")

            print("💾 3D adatok kinyerése a GPU memóriából...")
            t = droid.video.counter.value
            map_data = {
                "images": droid.video.images[:t].cpu().numpy(),
                "disps": droid.video.disps_up[:t].cpu().numpy(),
                "poses": droid.video.poses[:t].cpu().numpy(),
                "intrinsics": droid.video.intrinsics[:t].cpu().numpy(),
            }
            torch.save(map_data, args.reconstruction_path)
            print(f"✅ A kész térkép elmentve ide: {args.reconstruction_path}")


if __name__ == "__main__":
    main()
