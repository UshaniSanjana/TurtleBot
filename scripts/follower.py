#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import mediapipe as mp
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist  # Required for robot movement
from cv_bridge import CvBridge
from ultralytics import YOLO

class HumanFollower:
    def __init__(self):
        rospy.init_node('human_follower', anonymous=True)
        
        # --- Configs ---
        self.target_dist = 1.2  # Robot tries to stay 1.2m away
        self.stop_dist = 0.8    # Emergency stop distance
        self.active = False     # Robot state (Gesture Controlled)
        
        # --- PID Control Gains (Tune these if robot is too twitchy) ---
        self.linear_k = 0.4     # Speed multiplier for forward/back
        self.angular_k = 0.003  # Speed multiplier for turning
        
        # --- Vision Tools ---
        rospy.loginfo("Loading YOLOv8... (The Gatekeeper)")
        self.yolo_model = YOLO("yolov8n.pt") 
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.bridge = CvBridge()
        
        # --- Data Holders ---
        self.depth_image = None
        
        # --- Publishers & Subscribers ---
        # The 'cmd_vel_mux' topic is the safest way to drive a TurtleBot
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        
        rospy.Subscriber("/camera/color/image_raw", Image, self.color_cb)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb)
        
        rospy.loginfo("System Ready. Raise RIGHT hand to Start, LEFT to Stop.")

    def depth_cb(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr("Depth error: %s", e)

    def color_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(frame)
        except Exception as e:
            rospy.logerr("Color error: %s", e)

    def process_frame(self, frame):
        if self.depth_image is None:
            return

        # --- STEP 1: YOLO Detection ---
        yolo_results = self.yolo_model(frame, classes=0, conf=0.5, verbose=False)
        person_detected = False
        
        if len(yolo_results[0].boxes) > 0:
            person_detected = True
        
        # Safety: If no person, stop immediately
        if not person_detected:
            self.stop_robot()
            cv2.imshow("Robot Vision", frame)
            cv2.waitKey(1)
            return

        # --- STEP 2: MediaPipe Pose ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.pose.process(rgb_frame)
        
        if mp_results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, mp_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = mp_results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # --- TASK D: Gestures ---
            right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
            
            if right_wrist_y < nose_y: 
                self.active = True
                rospy.loginfo("GESTURE: Start Following!")
            elif left_wrist_y < nose_y:
                self.active = False
                rospy.loginfo("GESTURE: Stop Following!")

            # --- TASK B & C: Distance & Movement ---
            hip_x = int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w)
            hip_y = int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h)
            
            if 0 <= hip_x < w and 0 <= hip_y < h:
                # Get distance
                dist_m = self.depth_image[hip_y, hip_x] * 0.001 
                
                # Visuals
                color = (0, 255, 0) if self.active else (0, 0, 255)
                cv2.circle(frame, (hip_x, hip_y), 8, color, -1)
                cv2.putText(frame, f"{dist_m:.2f}m", (hip_x+10, hip_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # DRIVE LOGIC
                if self.active:
                    self.move_robot(dist_m, hip_x, w)
                else:
                    self.stop_robot()

        cv2.imshow("Robot Vision", frame)
        cv2.waitKey(1)

    def move_robot(self, dist, center_x, width):
        twist = Twist()
        
        # 1. Linear Control (Move Forward/Backward)
        # Error = Current Distance - Target Distance
        # If Error is positive (too far), move forward. Negative (too close), stop/back up.
        error_dist = dist - self.target_dist
        
        if dist < self.stop_dist:
            twist.linear.x = 0.0  # Too close! Safety stop.
        else:
            twist.linear.x = self.linear_k * error_dist
            # Limit max speed for safety
            twist.linear.x = max(min(twist.linear.x, 0.3), -0.1)

        # 2. Angular Control (Turn Left/Right)
        # We want the person (center_x) to be in the center of the image (width/2)
        error_turn = (width / 2) - center_x
        twist.angular.z = self.angular_k * error_turn
        
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist() # Zero velocity
        self.cmd_vel_pub.publish(twist)

if __name__ == '__main__':
    try:
        node = HumanFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
