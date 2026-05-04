# Independent Laboratory: Autonomous RC Car Navigation using DROID-SLAM

## Project Overview
The goal of this project is to implement autonomous navigation for an Ackermann-steered RC car in an indoor/laboratory environment. The car follows a pre-recorded (or taught) reference path using the Pure Pursuit algorithm while continuously localizing itself in space using the DROID-SLAM network. Additionally, it uses the camera's depth data (Depth map) for reactive obstacle avoidance (emergency brake).

## Hardware
* **Camera:** Intel RealSense D435i (RGB-D camera with built-in IMU and hardware calibration).
* **Vehicle:** Ackermann-steered RC car (controlled via target velocity [v] and steering angle [delta]).
* **Compute Units:**
  1. Client (On-board Jetson Nano): Camera reading and network transmission.
  2. Server (Nvidia GPU server on Linux): Performing heavy visual computations.

## Software Architecture (Distributed System)

The system consists of two main components communicating over a TCP network (ZeroMQ):

### 1. Client (Video Sender & Control)
* Runtime environment: On the car, Jetson Nano.
* Responsibilities:
  * Initializes the RealSense camera using `pyrealsense2`.
  * Retrieves the camera's factory intrinsic parameters.
  * Sends the frames (RGB + Depth) to the server via ZeroMQ.
  * Receives position data (x, y, theta) from the server and runs the Pure Pursuit control algorithm to command the vehicle hardware.

### 2. Server (DROID-SLAM Receiver)
* Runtime environment: High-performance GPU machine.
* Responsibilities:
  * Receives images and camera calibration data from the client.
  * Runs the DROID-SLAM neural network (PyTorch + CUDA lietorch backend).
  * Continuously updates the map and generates the Trajectory (Camera path).
  * Can generate a `point_cloud.ply` 3D point cloud using Open3D to visually verify the taught path and environment reconstruction.