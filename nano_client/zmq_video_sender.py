import cv2
import zmq
import time
import os
import glob
import argparse
from typing import Optional

def _try_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _encode_jpeg(frame_bgr, quality: int) -> bytes:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not success:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buffer.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream KITTI image sequences (or a video) as JPEG frames over ZeroMQ.",
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
        "--source",
        default=os.environ.get(
            "DROIDEKA_DATA_SOURCE",
            os.path.join(
                "nano_client",
                "2011_09_26",
                "2011_09_26_drive_0029_sync",
                "image_02",
                "data",
                "*.png",
            ),
        ),
        help="Video path or glob pattern for image sequence.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Streaming FPS for image sequences.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="For glob / image-sequence sources only: repeat the sequence forever "
        "(needed to keep SLAM fed after stop_teach during autonomy tests).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (0-100, default: 95).",
    )
    parser.add_argument(
        "--resize",
        default="960x288",
        help="Resize frames to WIDTHxHEIGHT before encoding (default: 960x288). Use 'none' to disable.",
    )
    args = parser.parse_args()

    resize = None
    if args.resize.lower() != "none":
        if "x" not in args.resize:
            raise ValueError("--resize must be WIDTHxHEIGHT or 'none'")
        w_str, h_str = args.resize.lower().split("x", 1)
        resize = (int(w_str), int(h_str))

    delay_s = 1.0 / max(args.fps, 1e-6)

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{args.server_ip}:{args.port}")
    print(f"Connected to server: tcp://{args.server_ip}:{args.port}")

    frame_count = 0
    source = args.source

    try:
        if "*" in source or "?" in source:
            # Image sequence
            files = sorted(glob.glob(source))
            if not files:
                print(f"ERROR: No images found for glob: {source}")
                return 2

            mode = "looping" if args.loop else "one pass"
            print(
                f"Streaming image sequence: {len(files)} frames ({mode}). "
                "Stop with Ctrl+C."
            )
            while True:
                for file_path in files:
                    frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if resize is not None:
                        frame = cv2.resize(frame, resize)

                    payload = _encode_jpeg(frame, quality=args.jpeg_quality)
                    socket.send(payload)

                    frame_count += 1
                    time.sleep(delay_s)
                if not args.loop:
                    break
        else:
            # Video file
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"ERROR: Could not open video: {source}")
                return 2

            fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
            delay_s = 1.0 / max(float(fps), 1e-6)
            print(f"Streaming video at {fps:.2f} FPS. Stop with Ctrl+C.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if resize is not None:
                    frame = cv2.resize(frame, resize)

                payload = _encode_jpeg(frame, quality=args.jpeg_quality)
                socket.send(payload)

                frame_count += 1
                time.sleep(delay_s)
            cap.release()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        socket.close(linger=0)

    print(f"Done. Sent {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())