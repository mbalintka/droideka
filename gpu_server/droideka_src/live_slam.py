"""Live DROID-SLAM over ZeroMQ (GPU server).

Single continuous SLAM session with two states:

    TEACH       — accumulate keyframes from the Nano's JPEG stream
                  (default startup state).
    AUTONOMOUS  — keep tracking, but also publish the latest rear-axle
                  pose on every successful ``droid.track()``.

ZeroMQ sockets:

* ``tcp://*:5555``  PULL — JPEG frames from the Nano (existing).
* ``tcp://*:5556``  PUB  — pose stream in AUTONOMOUS state:
                           ``{"x", "y", "theta", "frame_id", "t_ns"}`` (JSON).
* ``tcp://*:5557``  REP  — command channel. Currently understood commands:
                              ``{"cmd": "stop_teach", ...path-build options}``
                           Reply on ``stop_teach`` is a packed ``(N, 3)`` float64
                           ``[x, y, theta]`` path (multipart: ``[meta_json, bytes]``).

The world frame is the SLAM frame established during teach; the path returned
on ``stop_teach`` and the live pose stream are both expressed in the same
frame, so the Nano's Pure Pursuit can consume them without any extra
transformation. Pose values are at the **rear axle** (camera offset already
applied here — see ``pure_pursuit.camera_pose_to_vehicle_pose``).

Environment:

* ``DROID_SLAM_ROOT``    — root of a DROID-SLAM checkout (adds ``droid_slam/``
                           to ``sys.path`` for the ``droid`` import).
* ``DROID_SLAM_WEIGHTS`` — optional override for ``--weights``.
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
import json
import math
import time

import cv2
import numpy as np
import torch
import zmq
from droid import Droid
from lietorch import SE3

from .path_from_reconstruction import build_path_from_poses
from .pure_pursuit import CAMERA_OFFSET_FORWARD_M, camera_pose_to_vehicle_pose


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

    # Networking
    parser.add_argument("--frames_port", type=int, default=5555)
    parser.add_argument("--pose_port", type=int, default=5556)
    parser.add_argument("--cmd_port", type=int, default=5557)

    # Path-build defaults (overridable per request inside the stop_teach payload)
    parser.add_argument("--min_step_m", type=float, default=0.03)
    parser.add_argument("--smooth_window", type=int, default=7)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------


def _latest_rear_axle_pose(droid) -> tuple[float, float, float, int] | None:
    """Return the most recent rear-axle pose ``(x, y, theta, frame_id)``.

    ``x`` and ``y`` are the SLAM X / Z coordinates (Y / vertical dropped),
    ``theta`` is the heading in the X–Z plane (0 along +X, CCW positive).
    Returns ``None`` if no pose has been added to the SLAM video yet.
    """
    t = int(droid.video.counter.value)
    if t <= 0:
        return None

    # World-to-camera SE3 at the last keyframe -> camera-to-world matrix.
    pose7 = droid.video.poses[t - 1].detach().clone().cpu()  # (7,)
    T_wc = SE3(pose7[None]).inv().matrix()[0].numpy()  # (4, 4)

    cam_x = float(T_wc[0, 3])
    cam_z = float(T_wc[2, 3])

    # Camera forward = third column of R_wc.
    fwd_x = float(T_wc[0, 2])
    fwd_z = float(T_wc[2, 2])
    yaw = math.atan2(fwd_z, fwd_x)

    # Camera sits CAMERA_OFFSET_FORWARD_M ahead of the rear axle.
    rear_x, rear_y = camera_pose_to_vehicle_pose(
        cam_x, cam_z, yaw, cam_offset_fwd_m=CAMERA_OFFSET_FORWARD_M
    )
    return rear_x, rear_y, yaw, t - 1


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------


def _handle_stop_teach(
    droid,
    payload: dict,
    default_min_step_m: float,
    default_smooth_window: int,
) -> tuple[np.ndarray, dict]:
    """Build the (N,3) [x,y,theta] path from the current SLAM buffer.

    Returns ``(path_xyz_theta, meta)``. ``meta`` is the JSON-serialisable
    dict that will be sent as the first frame of the multipart reply.
    """
    t = int(droid.video.counter.value)
    if t < 2:
        raise RuntimeError(
            f"Cannot build path: only {t} keyframe(s) accumulated so far."
        )

    poses = droid.video.poses[:t].detach().clone().cpu()  # (t, 7)
    min_step_m = float(payload.get("min_step_m", default_min_step_m))
    smooth_window = int(payload.get("smooth_window", default_smooth_window))

    pos_xz, yaws = build_path_from_poses(
        poses,
        min_step_m=min_step_m,
        smooth_window=smooth_window,
        verbose=True,
    )

    path_xy_theta = np.concatenate(
        [pos_xz.astype(np.float64), yaws.astype(np.float64)[:, None]], axis=1
    )
    path_xy_theta = np.ascontiguousarray(path_xy_theta, dtype=np.float64)

    total_length = float(
        np.sum(np.linalg.norm(np.diff(path_xy_theta[:, :2], axis=0), axis=1))
    )
    meta = {
        "ok": True,
        "shape": list(path_xy_theta.shape),
        "dtype": "float64",
        "frame": "slam_xz",
        "yaw_convention": "atan2(forward_z, forward_x)",
        "total_length_m": total_length,
        "min_step_m": min_step_m,
        "smooth_window": smooth_window,
    }
    return path_xy_theta, meta


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    args = get_args()

    print("Loading DROID-SLAM network onto GPU...")
    droid = Droid(args)
    print("DROID-SLAM ready.")

    context = zmq.Context.instance()

    frames_sock = context.socket(zmq.PULL)
    frames_sock.bind(f"tcp://*:{args.frames_port}")

    pose_sock = context.socket(zmq.PUB)
    pose_sock.bind(f"tcp://*:{args.pose_port}")

    cmd_sock = context.socket(zmq.REP)
    cmd_sock.bind(f"tcp://*:{args.cmd_port}")

    poller = zmq.Poller()
    poller.register(frames_sock, zmq.POLLIN)
    poller.register(cmd_sock, zmq.POLLIN)

    print(
        f"Sockets bound: frames=PULL:{args.frames_port}  "
        f"pose=PUB:{args.pose_port}  cmd=REP:{args.cmd_port}"
    )

    # Image size used to build the intrinsics: matches --image_size default
    # 288x960. If you change --image_size, also change these.
    intrinsics = torch.tensor([555.3, 552.0, 469.0, 142.2])
    frame_id = 0

    state = "TEACH"
    print(f"State: {state} — drive manually and stream frames on :{args.frames_port}.")

    try:
        while True:
            # Poll both sockets so commands are handled promptly even if frames
            # are still flowing in.
            events = dict(poller.poll(timeout=10))

            if cmd_sock in events:
                try:
                    raw = cmd_sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    raw = None

                if raw is not None:
                    try:
                        payload = json.loads(raw.decode("utf-8")) if raw else {}
                    except json.JSONDecodeError:
                        payload = {}
                    cmd = str(payload.get("cmd", "")).strip().lower()
                    print(f"[cmd] received: {payload}")

                    if cmd == "stop_teach":
                        try:
                            path_xyz_theta, meta = _handle_stop_teach(
                                droid,
                                payload,
                                args.min_step_m,
                                args.smooth_window,
                            )
                            cmd_sock.send_multipart(
                                [
                                    json.dumps(meta).encode("utf-8"),
                                    path_xyz_theta.tobytes(order="C"),
                                ]
                            )
                            state = "AUTONOMOUS"
                            print(
                                f"State: {state} — publishing live pose on "
                                f":{args.pose_port}."
                            )
                        except Exception as e:
                            err = {"ok": False, "error": str(e)}
                            cmd_sock.send_multipart(
                                [json.dumps(err).encode("utf-8"), b""]
                            )
                            print(f"[cmd] stop_teach failed: {e}")
                    elif cmd == "ping":
                        cmd_sock.send_multipart(
                            [
                                json.dumps(
                                    {"ok": True, "state": state, "frames": frame_id}
                                ).encode("utf-8"),
                                b"",
                            ]
                        )
                    else:
                        cmd_sock.send_multipart(
                            [
                                json.dumps(
                                    {"ok": False, "error": f"unknown cmd: {cmd!r}"}
                                ).encode("utf-8"),
                                b"",
                            ]
                        )

            if frames_sock in events:
                try:
                    message = frames_sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    message = None

                if message is not None:
                    npimg = np.frombuffer(message, dtype=np.uint8)
                    frame = cv2.imdecode(npimg, 1)

                    if frame is not None:
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
                        image_tensor = image_tensor.unsqueeze(0)

                        try:
                            droid.track(frame_id, image_tensor, intrinsics=intrinsics)
                        except Exception as e:
                            print(f"[track] frame {frame_id} failed: {e}")
                            frame_id += 1
                            continue

                        if frame_id % 10 == 0:
                            print(
                                f"[{state}] tracked frame {frame_id} "
                                f"(keyframes: {int(droid.video.counter.value)})"
                            )
                        frame_id += 1

                        if state == "AUTONOMOUS":
                            try:
                                pose = _latest_rear_axle_pose(droid)
                            except Exception as e:
                                pose = None
                                print(f"[pose] extract failed: {e}")

                            if pose is not None:
                                rear_x, rear_y, yaw, kf_idx = pose
                                msg = {
                                    "x": rear_x,
                                    "y": rear_y,
                                    "theta": yaw,
                                    "frame_id": int(kf_idx),
                                    "t_ns": int(time.time_ns()),
                                }
                                try:
                                    pose_sock.send_string(
                                        json.dumps(msg), flags=zmq.NOBLOCK
                                    )
                                except zmq.Again:
                                    pass

    except KeyboardInterrupt:
        print("\nInterrupted. Finalising 3D map (Bundle Adjustment)...")
    finally:
        if frame_id > 0:
            print("Please wait, this may take 1-2 minutes...")
            try:
                droid.terminate()
            except Exception as e:
                print(f"Termination optimisation error: {e}")

            print("Extracting 3D data from GPU memory...")
            t = droid.video.counter.value
            map_data = {
                "images": droid.video.images[:t].cpu().numpy(),
                "disps": droid.video.disps_up[:t].cpu().numpy(),
                "poses": droid.video.poses[:t].cpu().numpy(),
                "intrinsics": droid.video.intrinsics[:t].cpu().numpy(),
            }
            torch.save(map_data, args.reconstruction_path)
            print(f"Map saved to: {args.reconstruction_path}")

        try:
            pose_sock.close(linger=0)
            cmd_sock.close(linger=0)
            frames_sock.close(linger=0)
            context.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()
