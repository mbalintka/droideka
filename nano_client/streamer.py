"""Live USB-camera streamer for the Nano (teach-mode video source).

Reads frames from a V4L2 USB camera, downscales and JPEG-encodes them, and
pushes the bytes to the GPU server over ZeroMQ ``PUSH`` — which pairs with
the ``PULL`` socket bound by ``gpu_server/droideka_src/live_slam.py`` on
port 5555.
"""

import argparse
import os
import time
from typing import Optional

import cv2
import zmq


def _try_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def start_streaming(
    server_ip: str,
    port: int,
    camera_index: int,
    resize: tuple[int, int],
    jpeg_quality: int,
    rate_hz: float,
) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{server_ip}:{port}")
    print(f"ZeroMQ PUSH connected to tcp://{server_ip}:{port}.")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera index {camera_index}.")
        socket.close(linger=0)
        return

    print("Camera opened. Streaming started... (Stop: Ctrl+C)")

    delay_s = 1.0 / max(rate_hz, 1e-6) if rate_hz > 0.0 else 0.0
    if delay_s > 0.0:
        print(f"Outgoing frame rate capped at {rate_hz:.1f} Hz ({delay_s * 1000:.0f} ms between sends).")
    else:
        print("Outgoing frame rate uncapped (sending as fast as the camera delivers).")

    frame_count = 0
    start_time = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: No frame received from camera.")
                time.sleep(0.1)
                continue

            # Downscale to reduce bandwidth (DROID-SLAM does not need full resolution).
            frame_resized = cv2.resize(frame, resize)

            success, encoded_image = cv2.imencode(".jpg", frame_resized, encode_param)
            if not success:
                continue

            payload = encoded_image.tobytes()
            socket.send(payload)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-6)
                size_kb = len(payload) / 1024
                print(
                    f"Sent: {frame_count} frames | FPS: {fps:.1f} | "
                    f"Size: {size_kb:.1f} KB"
                )

            if delay_s > 0.0:
                time.sleep(delay_s)

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        cap.release()
        socket.close(linger=0)
        print("Camera and network closed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a USB camera as JPEG frames to the GPU SLAM server (ZeroMQ PUSH).",
    )
    parser.add_argument(
        "--server-ip",
        default=os.environ.get("DROIDEKA_SERVER_IP", "127.0.0.1"),
        help="Server IP / hostname (default: 127.0.0.1 or env DROIDEKA_SERVER_IP).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_try_int(os.environ.get("DROIDEKA_SERVER_PORT")) or 5555,
        help="Server port (default: 5555 or env DROIDEKA_SERVER_PORT).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=_try_int(os.environ.get("DROIDEKA_CAMERA_INDEX")) or 2,
        help="V4L2 camera index (default: 2 or env DROIDEKA_CAMERA_INDEX).",
    )
    parser.add_argument(
        "--resize",
        default="960x288",
        help="Resize frames to WIDTHxHEIGHT before encoding (default: 960x288).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (0-100, default: 95).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help=(
            "Outgoing frame rate cap in Hz; 0 means send as fast as the camera "
            "delivers (default: 10, matching typical DROID-SLAM throughput)."
        ),
    )
    args = parser.parse_args()

    if "x" not in args.resize.lower():
        raise ValueError("--resize must be WIDTHxHEIGHT (e.g. 960x288)")
    w_str, h_str = args.resize.lower().split("x", 1)
    resize = (int(w_str), int(h_str))

    start_streaming(
        server_ip=args.server_ip,
        port=args.port,
        camera_index=args.camera_index,
        resize=resize,
        jpeg_quality=args.jpeg_quality,
        rate_hz=args.rate_hz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
