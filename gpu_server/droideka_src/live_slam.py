"""Live DROID-SLAM over ZeroMQ (GPU server).

Single continuous SLAM session with two states:

    TEACH       — accumulate keyframes from the Nano's JPEG stream
                  (default startup state).
    AUTONOMOUS  — keep tracking, but also publish the latest rear-axle
                  pose on every successful ``droid.track()``.

ZeroMQ sockets:

* ``tcp://*:5555``  PULL — JPEG frames: one raw part, or multipart
  ``[utf8_json, jpeg]`` with ``{"fx","fy","cx","cy"}`` (see README).
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
* ``DROIDEKA_INTRINSICS_JSON`` — optional path to ``{"fx","fy","cx","cy"}`` used
  when frames arrive as **single-part** JPEG (see ``--intrinsics-json``).
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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import zmq
from droid import Droid
from lietorch import SE3

from .path_from_reconstruction import build_path_from_poses
from .pure_pursuit import CAMERA_OFFSET_FORWARD_M, camera_pose_to_vehicle_pose
from .recording import record_frame


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
    parser.add_argument(
        "--buffer",
        type=int,
        default=512,
        help=(
            "Max keyframes in GPU ring buffer (DROID-SLAM DepthVideo size). "
            "On a 12 GB GPU use 512–768 for teach; 2048 pre-allocates ~10 GB "
            "at 288×960 and leaves no headroom for tracking or global BA. "
            "Each 512 slots reserve ~2.5 GB VRAM at 288×960."
        ),
    )
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

    # Default pinhole intrinsics when the client sends single-part JPEG only.
    # Precedence: ``--intrinsics`` > ``--intrinsics-json`` / ``DROIDEKA_INTRINSICS_JSON`` >
    # built-in default (960×288 pipeline).
    parser.add_argument(
        "--intrinsics-json",
        default=None,
        help=(
            "Path to JSON {\"fx\",\"fy\",\"cx\",\"cy\"} for single-part JPEG streams. "
            "Default: env DROIDEKA_INTRINSICS_JSON if set, else built-in default."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="Comma-separated fx,fy,cx,cy (overrides --intrinsics-json when set).",
    )

    # Path-build defaults (overridable per request inside the stop_teach payload)
    parser.add_argument("--min_step_m", type=float, default=0.03)
    parser.add_argument("--smooth_window", type=int, default=7)

    # Recording
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "If set, save every incoming JPEG frame to this directory as "
            "<frame_id:06d>.jpg. Use --record-teach-only to stop at stop_teach."
        ),
    )
    parser.add_argument(
        "--record-teach-only",
        action="store_true",
        default=False,
        help="Stop recording when stop_teach is received (default: record always).",
    )

    parser.add_argument(
        "--skip-terminate",
        action="store_true",
        default=False,
        help=(
            "Skip global bundle adjustment on shutdown. Frontend poses/disps "
            "are exported directly (recommended for ≤12 GB GPUs or when "
            "--buffer is large)."
        ),
    )
    parser.add_argument(
        "--no-auto-skip-terminate",
        action="store_true",
        default=False,
        help=(
            "Always attempt global BA on shutdown even when free VRAM is low. "
            "Default: auto-skip when headroom is insufficient."
        ),
    )
    parser.add_argument(
        "--terminate-max-keyframes",
        type=int,
        default=768,
        help=(
            "Auto-skip global BA when keyframe count exceeds this limit "
            "(default: 768). DROID backend allocates ~16× keyframes worth "
            "of GPU memory and can OOM on consumer GPUs."
        ),
    )

    return parser.parse_args()


_BUILTIN_INTRINSICS_288x960 = (555.3, 552.0, 469.0, 142.2)


def _load_fallback_intrinsics_tensor(args: argparse.Namespace) -> torch.Tensor:
    """Intrinsics used for single-part JPEG or when the wire JSON is invalid."""
    im_h, im_w = int(args.image_size[0]), int(args.image_size[1])

    if getattr(args, "intrinsics", None):
        raw = [p.strip() for p in str(args.intrinsics).split(",")]
        if len(raw) != 4:
            raise ValueError("--intrinsics expects exactly four comma-separated floats")
        fx, fy, cx, cy = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        if not _intrinsics_valid(fx, fy, cx, cy, im_h, im_w):
            raise ValueError(
                f"--intrinsics failed bounds check for image_size=({im_h}, {im_w})"
            )
        print(f"[intrinsics] using --intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}")
        return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    json_path = (getattr(args, "intrinsics_json", None) or "").strip() or None
    if not json_path:
        json_path = os.environ.get("DROIDEKA_INTRINSICS_JSON", "").strip() or None

    if json_path:
        with open(os.path.abspath(os.path.expanduser(json_path)), encoding="utf-8") as f:
            d = json.load(f)
        fx = float(d["fx"])
        fy = float(d["fy"])
        cx = float(d["cx"])
        cy = float(d["cy"])
        if not _intrinsics_valid(fx, fy, cx, cy, im_h, im_w):
            raise ValueError(
                f"intrinsics in {json_path!r} failed bounds check "
                f"for image_size=({im_h}, {im_w})"
            )
        print(
            f"[intrinsics] loaded {json_path}: fx={fx} fy={fy} cx={cx} cy={cy}"
        )
        return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    fx, fy, cx, cy = _BUILTIN_INTRINSICS_288x960
    print(
        f"[intrinsics] using built-in default (match {im_w}x{im_h} stream / --image_size)"
    )
    return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)


def _intrinsics_valid(fx: float, fy: float, cx: float, cy: float, im_h: int, im_w: int) -> bool:
    if fx <= 0.0 or fy <= 0.0 or im_h <= 0 or im_w <= 0:
        return False
    margin_x = 0.5 * float(im_w)
    margin_y = 0.5 * float(im_h)
    if not (-margin_x <= cx <= float(im_w) + margin_x):
        return False
    if not (-margin_y <= cy <= float(im_h) + margin_y):
        return False
    return True


def _parse_wire_intrinsics_json(
    raw: bytes, im_h: int, im_w: int
) -> tuple[float, float, float, float] | None:
    try:
        d = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        fx = float(d["fx"])
        fy = float(d["fy"])
        cx = float(d["cx"])
        cy = float(d["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if not _intrinsics_valid(fx, fy, cx, cy, im_h, im_w):
        return None
    return fx, fy, cx, cy


def _decode_frames_message(parts: list[bytes]) -> tuple[bytes | None, bytes | None]:
    """Return ``(jpeg_bytes, intrinsics_json_bytes)``.

    * One part: legacy raw JPEG, no intrinsics sidecar.
    * Two parts: ``[utf8_json, jpeg_bytes]`` with keys ``fx,fy,cx,cy``.
    """
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[1], parts[0]
    return None, None


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------


def _keyframe_count(droid, buffer_size: int) -> int:
    """Clamp keyframe count to the allocated DepthVideo buffer."""
    return min(int(droid.video.counter.value), buffer_size)


def _latest_rear_axle_pose(droid, buffer_size: int) -> tuple[float, float, float, int] | None:
    """Return the most recent rear-axle pose ``(x, y, theta, frame_id)``.

    ``x`` and ``y`` are the SLAM X / Z coordinates (Y / vertical dropped),
    ``theta`` is the heading in the X–Z plane (0 along +X, CCW positive).
    Returns ``None`` if no pose has been added to the SLAM video yet.
    """
    t = _keyframe_count(droid, buffer_size)
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
    buffer_size: int,
) -> tuple[np.ndarray, dict]:
    """Build the (N,3) [x,y,theta] path from the current SLAM buffer.

    Returns ``(path_xyz_theta, meta)``. ``meta`` is the JSON-serialisable
    dict that will be sent as the first frame of the multipart reply.
    """
    t = _keyframe_count(droid, buffer_size)
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


# Minimum free VRAM (GiB) needed for frontend correlation volumes / backend BA.
_MIN_TRACK_HEADROOM_GIB = 1.5
_MIN_TERMINATE_HEADROOM_GIB = 0.75


def _estimate_depth_video_gib(im_h: int, im_w: int, buffer: int, stereo: bool) -> float:
    """Upper-bound VRAM for the pre-allocated DepthVideo ring buffer."""
    ht8, wd8 = im_h // 8, im_w // 8
    c = 2 if stereo else 1
    bytes_per_slot = (
        3 * im_h * im_w  # images (uint8)
        + 2 * ht8 * wd8 * 4  # disps + disps_sens (float32)
        + im_h * im_w * 4  # disps_up (float32)
        + c * 128 * ht8 * wd8 * 2  # fmaps (float16)
        + 2 * 128 * ht8 * wd8 * 2  # nets + inps (float16)
        + 64  # poses, intrinsics, flags (negligible)
    )
    return buffer * bytes_per_slot / (1024**3)


def _gpu_mem_gib() -> tuple[float, float]:
    """Return (free_gib, total_gib) for the current CUDA device."""
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b / (1024**3), total_b / (1024**3)


def _warn_buffer_vram(args: argparse.Namespace) -> None:
    """Print VRAM estimate after Droid init; warn when headroom is tight."""
    im_h, im_w = int(args.image_size[0]), int(args.image_size[1])
    est_gib = _estimate_depth_video_gib(im_h, im_w, args.buffer, args.stereo)
    free_gib, total_gib = _gpu_mem_gib()
    print(
        f"[vram] DepthVideo buffer ({args.buffer} slots @ {im_w}x{im_h}): "
        f"~{est_gib:.1f} GiB reserved  |  GPU free {free_gib:.1f}/{total_gib:.1f} GiB"
    )
    if free_gib < _MIN_TRACK_HEADROOM_GIB:
        # Recommend a buffer that leaves ~2.5 GiB for the network + factor graph.
        target_gib = max(total_gib - 2.5, 1.0)
        per_slot_gib = _estimate_depth_video_gib(im_h, im_w, 1, args.stereo)
        suggested = max(128, int(target_gib / per_slot_gib))
        suggested = min(suggested, args.buffer)
        print(
            f"[vram] WARNING: only {free_gib:.1f} GiB free — tracking may OOM "
            f"(needs ~{_MIN_TRACK_HEADROOM_GIB:.1f} GiB headroom).\n"
            f"[vram] On this GPU try --buffer {suggested} (or add --skip-terminate). "
            f"Current --buffer {args.buffer} pre-allocates almost all VRAM."
        )
    elif not args.skip_terminate and free_gib < _MIN_TERMINATE_HEADROOM_GIB + 1.0:
        print(
            "[vram] Low headroom for global BA on shutdown — "
            "will auto-skip terminate unless --no-auto-skip-terminate is set."
        )


def _track_frame(droid, frame_id: int, image_tensor: torch.Tensor, intrinsics: torch.Tensor) -> None:
    """Run droid.track with one OOM recovery attempt (empty_cache + retry)."""
    try:
        droid.track(frame_id, image_tensor, intrinsics=intrinsics)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        try:
            droid.track(frame_id, image_tensor, intrinsics=intrinsics)
        except RuntimeError:
            raise e from None


def _release_frontend(droid) -> None:
    """Free frontend / factor-graph state before map export."""
    if getattr(droid, "frontend", None) is not None:
        del droid.frontend
        droid.frontend = None
        torch.cuda.empty_cache()


def _run_termination_ba(droid, args: argparse.Namespace) -> None:
    """Optional global bundle adjustment before map export."""
    t = _keyframe_count(droid, args.buffer)

    if args.skip_terminate:
        print(
            "[terminate] skipped global BA (--skip-terminate). "
            "Exporting frontend poses/disps."
        )
        _release_frontend(droid)
        return

    if not args.no_auto_skip_terminate:
        free_gib, _ = _gpu_mem_gib()
        if free_gib < _MIN_TERMINATE_HEADROOM_GIB:
            print(
                f"[terminate] skipped global BA ({free_gib:.2f} GiB free < "
                f"{_MIN_TERMINATE_HEADROOM_GIB:.2f} GiB needed). "
                "Exporting frontend poses/disps; map export is unaffected."
            )
            _release_frontend(droid)
            return

    if t > args.terminate_max_keyframes:
        print(
            f"[terminate] skipped global BA ({t} keyframes > "
            f"limit {args.terminate_max_keyframes}). "
            "Exporting frontend poses/disps; map quality is usually fine."
        )
        _release_frontend(droid)
        return

    print("[terminate] running global bundle adjustment...")
    try:
        droid.terminate()
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            print(
                f"[terminate] global BA OOM: {e}\n"
                "[terminate] continuing with frontend poses/disps "
                "(map export is unaffected)."
            )
            _release_frontend(droid)
            return
        raise


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    args = get_args()
    im_h, im_w = int(args.image_size[0]), int(args.image_size[1])

    record_dir: Optional[Path] = None
    recording_active: bool = False
    if args.record_dir is not None:
        record_dir = Path(args.record_dir)
        if not record_dir.parent.exists():
            print(
                f"ERROR: parent of --record-dir does not exist: {record_dir.parent}",
                file=sys.stderr,
            )
            sys.exit(1)
        record_dir.mkdir(parents=True, exist_ok=True)
        recording_active = True
        print(f"[record] saving frames to {record_dir}")

    print("Loading DROID-SLAM network onto GPU...")
    droid = Droid(args)
    print("DROID-SLAM ready.")
    _warn_buffer_vram(args)

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

    fallback_intrinsics = _load_fallback_intrinsics_tensor(args)
    last_wire_tuple: tuple[float, float, float, float] | None = None
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
                    cmd_parts = cmd_sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    cmd_parts = None

                if cmd_parts:
                    # REQ sends a leading delimiter frame; take the last part as JSON body.
                    raw = cmd_parts[-1] if cmd_parts else b""
                    try:
                        payload = json.loads(raw.decode("utf-8")) if raw else {}
                    except json.JSONDecodeError:
                        payload = {}
                    cmd = str(payload.get("cmd", "")).strip().lower()
                    print(f"[cmd] received: {payload}")

                    # Two application frames [meta_json, payload]; pyzmq/libzmq adds the
                    # REQ/REP delimiter on the wire — do not prepend b"" here or the Nano
                    # REQ recv_multipart becomes [b"", meta, body] and json.loads(frames[0])
                    # fails (and some stacks never complete the reply).
                    def _cmd_reply(meta_bytes: bytes, body: bytes) -> None:
                        cmd_sock.send_multipart([meta_bytes, body])

                    if cmd == "stop_teach":
                        try:
                            path_xyz_theta, meta = _handle_stop_teach(
                                droid,
                                payload,
                                args.min_step_m,
                                args.smooth_window,
                                args.buffer,
                            )
                            _cmd_reply(
                                json.dumps(meta).encode("utf-8"),
                                path_xyz_theta.tobytes(order="C"),
                            )
                            state = "AUTONOMOUS"
                            print(
                                f"State: {state} — publishing live pose on "
                                f":{args.pose_port}."
                            )
                            if args.record_teach_only and recording_active:
                                recording_active = False
                                print(
                                    f"[record] stopped at frame {frame_id} "
                                    "(teach-only mode)"
                                )
                        except Exception as e:
                            err = {"ok": False, "error": str(e)}
                            _cmd_reply(json.dumps(err).encode("utf-8"), b"")
                            print(f"[cmd] stop_teach failed: {e}")
                    elif cmd == "ping":
                        _cmd_reply(
                            json.dumps(
                                {"ok": True, "state": state, "frames": frame_id}
                            ).encode("utf-8"),
                            b"",
                        )
                    else:
                        _cmd_reply(
                            json.dumps(
                                {"ok": False, "error": f"unknown cmd: {cmd!r}"}
                            ).encode("utf-8"),
                            b"",
                        )

            if frames_sock in events:
                try:
                    parts = frames_sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    parts = None

                if parts:
                    jpeg_bytes, meta_bytes = _decode_frames_message(parts)
                    if jpeg_bytes is None:
                        print(
                            f"[frames] expected 1 or 2 ZMQ parts, got {len(parts)}; skipping"
                        )
                        continue

                    if meta_bytes is not None:
                        parsed = _parse_wire_intrinsics_json(meta_bytes, im_h, im_w)
                        if parsed is None:
                            print(
                                "[frames] invalid intrinsics JSON on wire; "
                                "using fallback intrinsics for this frame"
                            )
                            intrinsics = fallback_intrinsics
                        else:
                            intrinsics = torch.tensor(parsed, dtype=torch.float32)
                            if parsed != last_wire_tuple:
                                fx, fy, cx, cy = parsed
                                print(
                                    f"[intrinsics] from wire: fx={fx} fy={fy} cx={cx} cy={cy}"
                                )
                                last_wire_tuple = parsed
                    else:
                        intrinsics = fallback_intrinsics

                    npimg = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(npimg, 1)

                    if frame is not None:
                        # --- recording tee (before SLAM track) ---
                        record_frame(record_dir, frame_id, jpeg_bytes, recording_active)
                        # --- end recording tee ---

                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
                        image_tensor = image_tensor.unsqueeze(0)

                        kf_count = int(droid.video.counter.value)
                        if kf_count >= args.buffer:
                            if frame_id % 50 == 0:
                                print(
                                    f"[track] keyframe buffer full "
                                    f"({kf_count}/{args.buffer}). "
                                    "Send stop_teach or restart with a larger "
                                    "--buffer. SLAM updates skipped; CUDA may "
                                    "already be unusable if asserts appeared."
                                )
                            frame_id += 1
                            continue

                        if (
                            kf_count >= int(0.9 * args.buffer)
                            and frame_id % 100 == 0
                        ):
                            print(
                                f"[track] keyframe buffer "
                                f"{kf_count}/{args.buffer} "
                                f"({100 * kf_count // args.buffer}%) — "
                                "approaching limit; send stop_teach soon or "
                                "restart with --buffer <N>."
                            )

                        try:
                            _track_frame(droid, frame_id, image_tensor, intrinsics)
                        except Exception as e:
                            print(f"[track] frame {frame_id} failed: {e}")
                            frame_id += 1
                            continue

                        if frame_id % 10 == 0:
                            print(
                                f"[{state}] tracked frame {frame_id} "
                                f"(keyframes: {int(droid.video.counter.value)}"
                                f"/{args.buffer})"
                            )
                        frame_id += 1

                        if state == "AUTONOMOUS":
                            try:
                                pose = _latest_rear_axle_pose(droid, args.buffer)
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
        print("\nInterrupted. Finalising 3D map...")
    finally:
        if frame_id > 0:
            print("Please wait, this may take 1-2 minutes...")
            try:
                _run_termination_ba(droid, args)
            except Exception as e:
                print(f"Termination optimisation error: {e}")
                _release_frontend(droid)

            print("Extracting 3D data from GPU memory...")
            t = _keyframe_count(droid, args.buffer)
            try:
                map_data = {
                    "images": droid.video.images[:t].cpu().numpy(),
                    # Native 1/8 disparities (always updated during tracking).
                    "disps": droid.video.disps[:t].cpu().numpy(),
                    # Full-res upsampled maps (optional, may be sparse before BA).
                    "disps_up": droid.video.disps_up[:t].cpu().numpy(),
                    "poses": droid.video.poses[:t].cpu().numpy(),
                    "intrinsics": droid.video.intrinsics[:t].cpu().numpy(),
                }
                torch.save(map_data, args.reconstruction_path)
                print(f"Map saved to: {args.reconstruction_path} ({t} keyframes)")
            except RuntimeError as e:
                print(
                    f"Could not extract map ({t} keyframes): {e}\n"
                    "If you saw CUDA device-side asserts, restart the process "
                    "with a larger --buffer before the buffer fills."
                )

        try:
            pose_sock.close(linger=0)
            cmd_sock.close(linger=0)
            frames_sock.close(linger=0)
            context.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()
