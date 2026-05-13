import cv2
import zmq
import time

def start_streaming(camera_index=2, port=5555):
    # 1. ZeroMQ network setup (publisher mode)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # '*' publishes on all interfaces (Wi-Fi, LAN)
    socket.bind(f"tcp://*:{port}")
    print(f"ZeroMQ publisher started on port {port}.")

    # 2. Camera initialization
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera index {camera_index}.")
        return

    print("Camera opened. Streaming started... (Stop: Ctrl+C)")
    
    # Frame counter + timer for FPS measurement
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: No frame received from camera.")
                time.sleep(0.1)
                continue

            # --- PREPROCESSING ON THE NANO ---
            
            # A) Downscale to reduce bandwidth (DROID-SLAM does not need full resolution).
            frame_resized = cv2.resize(frame, (640, 480))
            
            # B) JPEG compression to dramatically reduce payload size.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, encoded_image = cv2.imencode('.jpg', frame_resized, encode_param)
            
            if success:
                # 3. Send encoded image over the network (raw bytes payload).
                socket.send(encoded_image.tobytes())
                
                frame_count += 1
                
                # Status line roughly once per second
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    size_kb = len(encoded_image.tobytes()) / 1024
                    print(f"Sent: {frame_count} frames | FPS: {fps:.1f} | Size: {size_kb:.1f} KB")

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        # Cleanup resources on exit
        cap.release()
        socket.close()
        context.term()
        print("Camera and network closed.")

if __name__ == "__main__":
    # If `find_usb_camera.py` finds another index, update it here.
    start_streaming(camera_index=2)