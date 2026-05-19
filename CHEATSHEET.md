# droideka live-test cheat sheet

One page, top-to-bottom. Open six terminals (GPU server / `car_control_node`
/ WASD teleop / camera streamer / `ping` probe / autonomy) and walk down
in order. The **camera streamer** in step 4 is usually the Nano for USB
(`streamer.py`); for `realsense_streamer.py` use any machine with
`pyrealsense2` and the D435i on USB (often a laptop on the same LAN as the
GPU). Every command assumes you've `cd`'d into the relevant repo root
and that `<gpu>` is the GPU server's IP/hostname.

---

## 0. Pre-flight (do this once per test session)

- Camera mounted, lens unobstructed, lens cap off.
- Both batteries seated; ESC disarmed (or transmitter binding ready).
- `auto_control_ws` built and sourced on the Nano.
- `DROID_SLAM_ROOT` / `DROID_SLAM_WEIGHTS` exported on the GPU box.
- Same physical Wi-Fi / Ethernet net between Nano and GPU.
- Open space cleared along the planned route.
- A pen + the run-dir path (you'll get it in step 8) to jot field notes.

---

## 1. GPU server — start SLAM (Terminal 1, GPU box)

```bash
cd gpu_server
python -m droideka_src.live_slam
```

If the Nano runs **USB/V4L2** only (`streamer.py`), JPEGs carry no intrinsics;
pass a one-time calibration file so SLAM matches your D435i at 960×288:

```bash
python -m droideka_src.live_slam --intrinsics-json path/to/intrinsics.json
# or: export DROIDEKA_INTRINSICS_JSON=path/to/intrinsics.json
# or: --intrinsics fx,fy,cx,cy
```

When you use **`realsense_streamer.py`** or **`bag_streamer.py`**, they send
multipart frames with `{"fx","fy","cx","cy"}` on the wire — **no** extra
`live_slam` flags needed.

To record the camera feed for replay and review, add `--record-dir`:

```bash
python -m droideka_src.live_slam --record-dir runs/$(date +%Y%m%d_%H%M%S)/frames
# or teach-only (stops saving after stop_teach):
python -m droideka_src.live_slam \
    --record-dir runs/$(date +%Y%m%d_%H%M%S)/frames \
    --record-teach-only
```

Look for: `Sockets bound: frames=PULL:5555  pose=PUB:5556  cmd=REP:5557`
followed by `State: TEACH`.

---

## 2. ROS 2 driver — arm `/cmd_vel` (Terminal 2, Nano)

```bash
# Source ROS 2 + the auto_control_ws overlay first
ros2 run car_control car_control_node
```

Look for: the node prints that it's subscribed to `/cmd_vel`. Its ~1 s
watchdog will keep the car still until something starts publishing.

---

## 3. WASD teleop — manual driver (Terminal 3, Nano)

```bash
cd /path/to/droideka
python -m nano_client.controller.teleop
# Optional overrides:
#   --rate-hz 20          control loop frequency (default 20 Hz)
#   --config cfg.json     VehicleConfig JSON override
```

Key bindings:

| Key | Action |
|-----|--------|
| W | Forward (target_v m/s) |
| S | Reverse |
| A | Steer left |
| D | Steer right |
| Space | Immediate stop |
| Q / Ctrl+C | Quit (publishes zero Twist before exit) |

**IMPORTANT:** The deadman switch zeros the car if no key is pressed for 300 ms.
Kill this terminal **before** launching the autonomy node in step 8.

---

## 4. Stream the camera — feed SLAM (Terminal 4)

Pick **one** video source. It must match what `live_slam` expects: default
**960×288** JPEGs on port **5555** (same as `--image_size 288 960` on the GPU).

### 4a. Jetson USB camera (V4L2) — no RealSense SDK on the Nano

```bash
python nano_client/streamer.py --server-ip <gpu>
```

Single-part JPEG only. On the **GPU**, supply intrinsics (see step 1) unless
the baked default is good enough for your lens.

### 4b. Intel RealSense D435i via SDK — factory intrinsics on the wire

Run on any host that has **`pip install pyrealsense2`** (typical: your dev
laptop or the GPU box with the camera plugged in USB — **not** most Jetson
images, where the wheel is missing).

```bash
python nano_client/realsense_streamer.py --server-ip <gpu>
```

What it does:

- Opens the first available RealSense device, starts **BGR8** color (tries
  your `--resize` size first, e.g. **960×288**; if the firmware does not offer
  that mode, it falls back to **1280×720** or another common size, then
  **downscales** frames and **scales** `fx,fy,cx,cy` to match the output size).
- Reads factory intrinsics from the active stream profile, then every frame
  sends **two ZMQ parts**: `[utf8_json, jpeg_bytes]` with
  `{"fx","fy","cx","cy"}` so `live_slam` tracks with your camera’s K.

Useful flags (same env vars as `streamer.py`):

| Flag / env | Meaning |
|------------|---------|
| `--server-ip` / `DROIDEKA_SERVER_IP` | GPU hostname (default `127.0.0.1`) |
| `--port` / `DROIDEKA_SERVER_PORT` | Frames port (default **5555**) |
| `--resize` | Output **WIDTH×HEIGHT** (default **960x288**) — must stay aligned with `live_slam --image_size` |
| `--jpeg-quality` | JPEG 0–100 (default **95**) |

Example with explicit port and size:

```bash
python nano_client/realsense_streamer.py --server-ip 192.168.1.50 --port 5555 --resize 960x288
```

On the GPU you should see a one-time line like
`[intrinsics] from wire: fx=... fy=... cx=... cy=...` after the first frame,
then `[TEACH] tracked frame N (keyframes: K)` as in 4a.

### What to look for (4a or 4b)

`Sent: 30 frames | FPS: ~30 | Size: ~XX KB` every second (realsense path
logs the same style). JPEG quality default is **95** — confirm the link
keeps up.

In Terminal 1 (GPU), `[TEACH] tracked frame N (keyframes: K)` should climb.

---

## 5. Sanity-check the GPU state (Terminal 5, any host)

```bash
python -c "import zmq, json; \
s = zmq.Context().socket(zmq.REQ); \
s.connect('tcp://<gpu>:5557'); \
s.send_string(json.dumps({'cmd':'ping'})); \
print(s.recv_multipart())"
```

Look for: a reply containing `"state": "TEACH"` and a non-zero `"frames"`
counter. If you see a timeout or `state` is wrong, fix that first — the
autonomy node won't recover from a missing GPU.

---

## 6. Drive the route manually (Terminals 3 + 4 active)

CRITICAL: **do not pick up the car between teach and autonomy.** SLAM is
one continuous session — physically teleporting the car would corrupt the
world frame and either confuse the tracker or get rejected as a tracking
failure. The car's location at the moment you launch the autonomy node
in step 8 **is** the start of the autonomous follow.

Pick a route shape that matches what you want autonomy to do:

| Goal                            | Teach with WASD          | Where to stop teach | Autonomy will...               |
|---------------------------------|--------------------------|---------------------|--------------------------------|
| Closed loop                     | A -> ... -> A            | A                   | Drive the loop back to A.      |
| Out-and-back                    | A -> B -> A              | A                   | Re-drive A -> B -> A.          |
| One-way A -> B then stop        | A -> B                   | A (not B!)          | Drive A -> B and stop.         |
| (Anti-pattern: stop at B)       | A -> B                   | B                   | Immediately latch `goal_reached`, publish zero Twist, log a 0-second run. Useful only as a wiring sanity check. |

The path that `stop_teach` builds is the **entire** SLAM trajectory
recorded so far, so the "out-and-back" pattern works simply by driving
back manually before letting go of the teleop.

While driving:

- Keep the speed gentle on the first runs; the DROID-SLAM front-end likes
  smooth motion and steady illumination.
- Watch the GPU terminal: keyframe count should keep growing. If it
  plateaus, you're tracking poorly — slow down, add texture, or restart.

When you're happy with the route:

1. Bring the car to your intended **autonomy start** location (see table
   above) and Space-stop it there.
2. Ctrl+C the **WASD teleop** (Terminal 3). Leaving it running alongside
   autonomy will cause `/cmd_vel` fights.
3. Leave the **streamer** running — autonomy needs continuous frames to
   keep localising.

---

## 7. Post-run — stitch recorded frames into video (GPU box, optional)

```bash
cd gpu_server
python -m droideka_src.make_video \
    --frames-dir runs/<stamp>/frames \
    --output     runs/<stamp>/teach_run.mp4 \
    --fps        10
```

The resulting `teach_run.mp4` can be opened in any video player. The raw
frames in `runs/<stamp>/frames/` can also be replayed into DROID-SLAM:

```bash
# From droideka repo root:
python -m nano_client.zmq_video_sender \
    --server-ip <gpu> \
    --source "runs/<stamp>/frames/*.jpg"
# Provide intrinsics if recorded from a USB/V4L2 camera (not RealSense):
#   live_slam.py --intrinsics-json path/to/intrinsics.json
```

---

## 8. Hand off to autonomy (Terminal 6, Nano)

```bash
python -m nano_client.controller.node --gpu-host <gpu>
```

Optional flags:

```bash
# Pin the run-dir name (e.g. for a labelled experiment):
python -m nano_client.controller.node --gpu-host <gpu> \
    --run-dir nano_client/runs/2026-05-13_lap_A

# Override the controller defaults (target_v capped at 0.3 m/s out of the box):
python -m nano_client.controller.node --gpu-host <gpu> --config my_car.json
```

Look for, in order:

1. `[autonomy] vehicle config: {...}` — confirm `target_v` is what you
   expect (0.3 m/s default).
2. `[autonomy] run artefacts -> nano_client/runs/<YYYYMMDD_HHMMSS>/` —
   **jot this path down**.
3. `[autonomy] received path: N waypoints, length=L m` — N ≥ 2, L > 0.
4. `[autonomy] taught path saved -> .../path_taught.npy`.
5. `Publishing /cmd_vel at 30.0 Hz; path has N waypoints.`

The car will start moving toward the first waypoint. Be ready to kill the
process or yank the battery if anything looks off.

---

## 9. During autonomy — what to watch

- **GPU terminal**: `[AUTONOMOUS] tracked frame N` lines keep climbing.
- **Autonomy terminal**: occasional `pose stale` warnings are OK in
  isolation; sustained means the SLAM has lost tracking.
- **Behaviour**: smooth heading corrections; if it lunges, target_v is
  probably too high or the path was too noisy.

End of run: the node prints `Goal reached (distance_to_goal=...)` and
latches a zero Twist. The car will not move again until you Ctrl+C and
restart.

---

## 10. Shut down (in reverse order)

1. Ctrl+C the **autonomy node** (Terminal 6). It will publish a final
   stop Twist and write a `shutdown` record to `autonomy.jsonl`.
2. Ctrl+C the **streamer** (Terminal 4).
3. Ctrl+C the **car_control_node** (Terminal 2).
4. Ctrl+C the **GPU server** (Terminal 1). It will run a final bundle
   adjustment and save `nano_live_map.pth` — give it 1–2 minutes.

---

## 11. Collect artefacts

Per-run bundle from step 8.2:

```
nano_client/runs/<YYYYMMDD_HHMMSS>/
├── path_taught.npy   # the reference path (N, 3) [x, y, theta]
├── meta.json         # GPU build params + VehicleConfig + CLI args
└── autonomy.jsonl    # one record per pose / tick / event / shutdown
```

Quick "did it go well?" check:

```bash
# Just look at the shutdown summary — last line of the log:
tail -n 1 nano_client/runs/<ts>/autonomy.jsonl
```

Healthy run looks like:

```json
{"kind":"shutdown","reason":"completed","n_ticks":900,"n_poses":850,
 "min_pose_age_s":0.01,"max_pose_age_s":0.05,
 "final_dist_to_goal_m":0.18,"wall_duration_s":30.1, ...}
```

Red flags:

- `reason` other than `completed` (`interrupted`/`error`).
- `max_pose_age_s` > `pose_timeout_s` (0.5 s by default) — SLAM stalled.
- `final_dist_to_goal_m` > a couple of `goal_tolerance_m` — never
  converged.
- `n_ticks` ≪ expected (rate_hz * wall_duration_s) — control loop was
  starved.

---

## Quick reference — ports & topics

| Channel        | Address                | Pattern  | Notes                                        |
|----------------|------------------------|----------|----------------------------------------------|
| JPEG frames    | `tcp://<gpu>:5555`     | PUSH/PULL| Nano -> GPU: **one** part = raw JPEG (`streamer.py`, `zmq_video_sender.py`), or **two** parts = `[{"fx","fy","cx","cy"} JSON, JPEG]` (`realsense_streamer.py`, `bag_streamer.py`). |
| Pose stream    | `tcp://<gpu>:5556`     | PUB/SUB  | GPU -> Nano (autonomous state only)          |
| Command (REQ)  | `tcp://<gpu>:5557`     | REQ/REP  | `stop_teach` / `ping`                        |
| `/cmd_vel`     | ROS 2 topic            | pub/sub  | Twist; only one publisher at a time          |

## Quick reference — knob locations

| Knob                       | Where                                                     | Default |
|----------------------------|-----------------------------------------------------------|---------|
| Target velocity            | `VehicleConfig.target_v` ([config.py](nano_client/controller/config.py)) | 0.3 m/s |
| Max steering               | `VehicleConfig.max_steer_rad`                             | 30°    |
| Wheelbase                  | `VehicleConfig.wheelbase`                                 | 0.32 m  |
| Pose-stale threshold       | `VehicleConfig.pose_timeout_s`                            | 0.5 s   |
| Goal tolerance             | `VehicleConfig.goal_tolerance_m`                          | 0.20 m  |
| JPEG quality (all streamers)| `--jpeg-quality` CLI                                     | 95      |
| Resize (streamers)         | `--resize` CLI                                            | 960x288 |
| SLAM intrinsics            | Wire JSON+JPEG from [realsense_streamer.py](nano_client/realsense_streamer.py) / [bag_streamer.py](nano_client/bag_streamer.py); else `--intrinsics-json` / `DROIDEKA_INTRINSICS_JSON` / `--intrinsics` on [live_slam.py](gpu_server/droideka_src/live_slam.py); else built-in default for 960×288 |

## Quick reference — emergency stops

- **Software**: Ctrl+C the autonomy node. It publishes a zero Twist on its
  way out, and the `car_control_node` watchdog also catches the silence
  within ~1 s.
- **ROS-side**: from any sourced terminal, spam a zero Twist:
  `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'`
- **Hardware**: kill the ESC (battery disconnect or transmitter cutoff).
  Always do this if the software stop isn't working within a second.
