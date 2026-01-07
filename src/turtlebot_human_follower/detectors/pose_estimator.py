## File: src/turtlebot_human_follower/detectors/pose_estimator.py

#!/usr/bin/env python3
"""
TASK A: Pose Estimation using MediaPipe
Tracks human skeleton keypoints for gesture recognition and tracking.
"""

import cv2
import mediapipe as mp
import rospy

class PoseEstimator:
    def __init__(self, min_detection_conf=0.7, min_tracking_conf=0.7):
        """
        Initialize MediaPipe Pose estimator.
        
        Args:
            min_detection_conf: Minimum confidence for detection
            min_tracking_conf: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        # Optimized settings for real-time CPU performance
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for tracking
            model_complexity=0,  # 0=Lite, 1=Full, 2=Heavy (use 0 for speed)
            smooth_landmarks=True,  # Smooth jitter
            enable_segmentation=False,  # Disable segmentation for speed
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def estimate(self, frame):
        """
        Estimate pose from RGB frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            pose_landmarks: MediaPipe pose landmarks or None
        """
        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        return results.pose_landmarks
    
    def get_landmark_position(self, landmarks, landmark_id, frame_shape):
        """
        Get pixel coordinates of a specific landmark.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_id: ID of the landmark (e.g., LEFT_HIP)
            frame_shape: Shape of the frame (h, w, c)
            
        Returns:
            tuple: (x, y) pixel coordinates or None
        """
        if landmarks is None:
            return None
            
        h, w = frame_shape[:2]
        landmark = landmarks.landmark[landmark_id]
        
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        # Validate coordinates
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        
        return None
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: BGR image
            landmarks: MediaPipe pose landmarks
        """
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS
            )
