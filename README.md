# TurtleBot Human Follower 

This ROS package enables a TurtleBot to follow a human using an RGB-D camera (like Intel RealSense). It leverages **YOLOv8** for robust person detection and **MediaPipe** for pose estimation to allow gesture-based control (starting and stopping the following behavior).

## Features
- **Human Detection:** Uses YOLOv8 to ensure a human is in the frame.
- **Gesture Control (MediaPipe):**
  - Raise **RIGHT hand** to start following.
  - Raise **LEFT hand** to stop following.
- **Depth-Based Tracking:** Uses the depth camera to maintain a safe distance from the human (target distance: 1.2m, emergency stop: 0.8m).
- **PID Control:** Smoothly drives the TurtleBot toward the human.

## Requirements

### System Requirements
- Ubuntu 20.04 (ROS Noetic) or Ubuntu 18.04 (ROS Melodic)
- Python 3.x

### Dependencies
Install the required ROS packages and Python libraries:

```bash
# ROS Dependencies
sudo apt-get install ros-noetic-cv-bridge ros-noetic-geometry-msgs ros-noetic-sensor-msgs ros-noetic-std-msgs

# Python Dependencies
pip3 install ultralytics mediapipe opencv-python numpy
```

*(Note: The package requires `yolov8n.pt` which will be downloaded automatically by the `ultralytics` library on first run).*

## Installation

1. Clone this repository into the `src` folder of your Catkin workspace:
```bash
cd ~/catkin_ws/src
git clone https://github.com/UshaniSanjana/TurtleBot.git
```

2. Build the workspace:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

3. Make the Python scripts executable:
```bash
chmod +x ~/catkin_ws/src/human_following/scripts/*.py
```

## Usage

### 1. Launch the Camera and Robot Base
Ensure your TurtleBot and camera (e.g., RealSense) are running.
```bash
roslaunch turtlebot_bringup minimal.launch
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

### 2. Run the Human Follower Node
You can run the follower using the provided launch file:
```bash
roslaunch human_following human_follower.launch
```

Alternatively, you can run the python script directly:
```bash
rosrun human_following follower.py
```

## ROS API

### Subscribed Topics
- `/camera/color/image_raw` (`sensor_msgs/Image`) - RGB image for YOLO and MediaPipe processing.
- `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`) - Depth image for calculating the distance to the human.

### Published Topics
- `/cmd_vel_mux/input/teleop` (`geometry_msgs/Twist`) - Velocity commands sent to the TurtleBot.

### Parameters
Parameters can be modified in `launch/human_follower.launch` or `config/params.yaml`:
- `color_topic`: RGB camera topic.
- `depth_topic`: Depth camera topic.
- `cmd_vel_topic`: Velocity command topic.
