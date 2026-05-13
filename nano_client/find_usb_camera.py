import cv2

def find_working_camera():
    """
    Iterate over the first 3 video indices (0, 1, 2) to find a working USB camera.
    On Linux, using the cv2.CAP_V4L2 backend is recommended for stability.
    """
    print(f"OpenCV version: {cv2.__version__}")
    print("Searching for camera...\n" + "-" * 30)

    # Try indices 0, 1, 2
    for index in range(3):
        print(f"[index={index}] Trying /dev/video{index} ...")
        
        # cv2.CAP_V4L2 tells OpenCV to use the Linux V4L2 backend
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # If it opened, try grabbing a frame as well
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS: Working camera found at index {index}.")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return index
            else:
                print(
                    f"WARNING: Index {index} opened, but no frame was read (could be a virtual device)."
                )
        else:
            print(f"No device available at index {index}.")
            
        cap.release()

    print("-" * 30 + "\nERROR: No working camera found. Check the USB connection.")
    return None

if __name__ == "__main__":
    found_index = find_working_camera()