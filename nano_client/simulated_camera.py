import cv2
import time
import os

def test_camera_headless():
    print("Testing camera in headless mode...\n" + "-" * 40)
    
    # Try indices 0, 1, 2 (typical for USB cameras on Linux)
    for index in range(3):
        print(f"[index={index}] Checking /dev/video{index} ...")
        
        # Use V4L2 backend for Linux
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            print(f"Camera opened at index {index}.")
            print("  Warming up (auto-exposure settling)...")
            
            # Read a few frames to let exposure/focus settle
            for _ in range(10):
                ret, frame = cap.read()
                time.sleep(0.05)
            
            if ret:
                height, width, channels = frame.shape
                print(f"  Resolution: {width}x{height} (Channels: {channels})")
                
                # Save the last captured frame as an image file
                filename = f"test_camera_frame_index_{index}.jpg"
                cv2.imwrite(filename, frame)
                
                print(f"SUCCESS: Test frame saved to: {filename}")
                
                cap.release()
                return True
            else:
                print("WARNING: Camera opened, but no frame was captured.")
            
            cap.release()
        else:
            print(f"No device available at index {index}.")

    print("-" * 40 + "\nERROR: No working camera found. Is the USB device connected?")
    return False

if __name__ == "__main__":
    test_camera_headless()