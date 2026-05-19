# Testing Setup: WASD Teleop Node & Camera Recording

**Date:** 2026-05-19  
**Status:** Approved — ready for implementation

---

## Context

droideka is an autonomous RC-car stack: a Jetson Nano runs Pure Pursuit over poses
from DROID-SLAM (GPU server), communicating via ZeroMQ. The teach phase requires
manual driving while the camera stream feeds SLAM. Two pieces are missing from the
in-repo tooling:

1. A WASD teleop node — currently referenced in the CHEATSHEET as an external
   `wasd_teleop.py` that does not exist in this repo.
2. Camera recording — the existing streamers forward frames to the GPU for SLAM
   only; there is no way to save footage for later replay or review.

---

## 1. WASD Teleop Node

### 1.1 Location

`nano_client/controller/teleop.py`

Sits in the existing `controller` package alongside `node.py` (autonomy) and
`config.py`. Same import guard pattern: `rclpy` is imported inside `main()` so the
module remains importable on Windows/CI without ROS 2.

### 1.2 Entry point

```
python -m nano_client.controller.teleop [--rate-hz 20] [--config path/to/config.json]
```

`--config` accepts the same `VehicleConfig` JSON override used by the autonomy node.

### 1.3 Architecture

`WasdTeleopNode(rclpy.node.Node)`:

- **Publisher:** `/cmd_vel` (`geometry_msgs/Twist`, depth 10), same topic as the
  autonomy node and `car_control_node`.
- **Timer:** fires at `--rate-hz` (default 20 Hz) and publishes the current
  commanded Twist.
- **Keyboard thread:** daemon thread using Linux `termios` raw-terminal mode (no
  extra deps on the Jetson). Reads one byte at a time and updates two shared state
  variables: `_cmd_v` (float) and `_cmd_delta` (float), plus `_last_key_ns`
  (monotonic timestamp in nanoseconds). Python's GIL makes individual float
  assignments effectively atomic; no explicit lock is needed for these three
  variables. `_last_key_ns` is initialised to `time.time_ns()` at node
  construction so the deadman does not fire before the first key press.

### 1.4 Key bindings

| Key | Effect |
|-----|--------|
| `W` | `_cmd_v = +VehicleConfig.target_v` (forward) |
| `S` | `_cmd_v = -VehicleConfig.target_v` (reverse) |
| `A` | `_cmd_delta = -VehicleConfig.max_steer_rad` (left) |
| `D` | `_cmd_delta = +VehicleConfig.max_steer_rad` (right) |
| `Space` | `_cmd_v = 0`, `_cmd_delta = 0` (immediate stop) |
| `Q` / Ctrl+C | graceful shutdown |

`W`/`S` and `A`/`D` are independent state variables — holding `W`+`A` gives
forward-left. Releasing a key does not automatically zero its axis; the deadman
switch (§1.5) handles that.

### 1.5 Passive deadman switch

The timer callback checks `time.time_ns() - _last_key_ns`. If the gap exceeds
300 ms (configurable constant `_DEADMAN_MS`), both `_cmd_v` and `_cmd_delta` are
zeroed before publishing. This means releasing all keys stops the car within one
control tick — no explicit key-up event needed, which `termios` raw mode does not
provide reliably.

### 1.6 Terminal HUD

On every state change (key press or deadman trigger), the keyboard thread prints
one line to stdout:

```
[teleop] v=+0.30 m/s  δ=-0.52 rad (LEFT)  — W·S·A·D steer | SPACE stop | Q quit
```

### 1.7 Shutdown

`finally` block in `main()`:
1. Restore terminal to cooked mode (`termios.tcsetattr`).
2. Publish one zero `Twist` on `/cmd_vel`.
3. `rclpy.shutdown()`.

If `termios` fails to set raw mode (non-TTY, Windows), the node prints a clear
error and exits before ROS init.

### 1.8 Mutual exclusion with autonomy node

Both teleop and autonomy nodes publish on `/cmd_vel`. The CHEATSHEET step is
updated to say: **kill the teleop terminal before launching the autonomy node**.
No software interlock is added — the `car_control_node` watchdog catches a stale
stream, and the autonomy node always publishes a zero Twist on startup, resolving
any momentary overlap.

---

## 2. Camera Recording

### 2.1 Where the change lands

`gpu_server/droideka_src/live_slam.py` — two new CLI arguments:

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--record-dir PATH` | `Path\|None` | `None` (off) | Directory to save incoming JPEGs |
| `--record-teach-only` | flag | `False` | Stop saving frames once `stop_teach` is received |

### 2.2 Implementation

In the frame-receive branch (after `_decode_frames_message`, before `droid.track()`):

```python
if record_dir is not None and jpeg_bytes is not None:
    (record_dir / f"{frame_id:06d}.jpg").write_bytes(jpeg_bytes)
    if frame_id % 50 == 0:
        print(f"[record] frame {frame_id:06d} -> {record_dir}")
```

`record_dir` is created with `mkdir(parents=True, exist_ok=True)` on startup.
If the parent path does not exist, `live_slam.py` exits before loading DROID-SLAM
to avoid a slow fail.

Disk write errors (full disk, permissions) are caught per-frame, logged as
warnings, and do not interrupt the SLAM loop — recording is best-effort.

### 2.3 Frame naming

`000001.jpg`, `000002.jpg`, … using the existing sequential `frame_id` counter.
Zero-padded to six digits so shell globs and `zmq_video_sender.py` sort correctly.

### 2.4 Console feedback

Every 50 frames: `[record] frame 000100 -> /path/to/frames/`  
On `--record-teach-only` + `stop_teach`: `[record] stopped at frame 000237 (teach-only mode)`

### 2.4a `--record-teach-only` implementation detail

`main()` maintains a boolean `_recording_active` (default `True` when
`--record-dir` is set). When the `stop_teach` command is processed in the cmd
branch, if `--record-teach-only` is set, `_recording_active` is flipped to
`False` and the console message above is printed. The tee guard becomes:

```python
if record_dir is not None and _recording_active and jpeg_bytes is not None:
```

### 2.5 Companion offline script: `make_video.py`

`gpu_server/droideka_src/make_video.py`

```
python -m droideka_src.make_video \
    --frames-dir runs/20260519_112000/frames \
    --output     runs/20260519_112000/teach_run.mp4 \
    --fps        10
```

- Globs `<frames-dir>/*.jpg` in sorted order.
- Uses `cv2.VideoWriter` with codec `mp4v` (already a transitive dep via
  `live_slam.py`).
- `--fps` default: 10 (matches typical DROID-SLAM throughput on real hardware).
- Exits with a clear error if the directory is empty or no JPEGs are found.

### 2.6 Replay into SLAM

Saved JPEGs are in the exact format `zmq_video_sender.py` already accepts. To
replay a recorded teach session through SLAM:

```
python -m nano_client.zmq_video_sender \
    --server-ip <gpu> \
    --source "runs/20260519_112000/frames/*.jpg"
```

Intrinsics must be provided separately to `live_slam.py` via `--intrinsics-json`
when replaying a USB/webcam recording (same requirement as a live USB stream).
RealSense recordings already embed intrinsics in the multipart wire format and
replay correctly without the extra flag.

### 2.7 Run directory convention

`--record-dir` is free-form. Recommended convention: point it at a subdirectory
of the autonomy node's `--run-dir` so all artefacts land together:

```
runs/20260519_112000/
  frames/          ← --record-dir
  path_taught.npy  ← autonomy node
  meta.json        ← autonomy node
  autonomy.jsonl   ← autonomy node
  teach_run.mp4    ← make_video.py output
```

---

## 3. Operator Workflow (updated CHEATSHEET steps)

```
Terminal 1 (GPU)   python -m droideka_src.live_slam --record-dir runs/<stamp>/frames
Terminal 2 (Nano)  ros2 run car_control car_control_node
Terminal 3 (Nano)  python -m nano_client.controller.teleop          ← replaces external wasd_teleop
Terminal 4 (Nano)  python -m nano_client.realsense_streamer --server-ip <gpu>
                   [ drive the route; Ctrl+C Terminal 3 when done teaching ]
Terminal 5 (Nano)  python -m nano_client.controller.node --gpu-host <gpu> --run-dir runs/<stamp>
```

After the run:

```
python -m droideka_src.make_video \
    --frames-dir runs/<stamp>/frames \
    --output     runs/<stamp>/teach_run.mp4
```

---

## 4. Files Changed / Created

| File | Change |
|------|--------|
| `nano_client/controller/teleop.py` | **New** — `WasdTeleopNode` + `main()` |
| `nano_client/controller/__init__.py` | Export `WasdTeleopNode` |
| `gpu_server/droideka_src/live_slam.py` | Add `--record-dir` / `--record-teach-only` args + tee logic |
| `gpu_server/droideka_src/make_video.py` | **New** — JPEG → MP4 stitcher |
| `CHEATSHEET.md` | Update step 3 (teleop command) + add recording flags + post-run `make_video` step |

---

## 5. Out of Scope

- Software interlock between teleop and autonomy nodes (YAGNI — operator procedure is sufficient).
- Depth stream or IMU recording (future emergency-brake feature, tracked in AGENT.md).
- Windows `msvcrt` keyboard support for the teleop node (Jetson is Linux-only).
- Automatic intrinsics embedding in JPEG-only recordings (USB webcam limitation; use RealSense for embedded intrinsics).
