## File: src/turtlebot_human_follower/recognition/gesture_recognizer.py

#!/usr/bin/env python3
"""
TASK D: Hand Gesture Recognition
Recognizes hand signals for robot start/stop control.
"""

import rospy
import mediapipe as mp

class GestureRecognizer:
    def __init__(self, activation_threshold=0.15):
        """
        Initialize gesture recognizer.
        
        Args:
            activation_threshold: How much hand must be above nose (normalized)
        """
        self.mp_pose = mp.solutions.pose
        self.threshold = activation_threshold
        self.last_gesture = None
        
    def recognize(self, landmarks):
        """
        Recognize hand gestures from pose landmarks.
        
        Gestures:
        - Right hand raised above head → START following
        - Left hand raised above head → STOP following
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            str: 'start', 'stop', or None
        """
        if landmarks is None:
            return None
            
        # Get landmark positions (normalized 0-1)
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        right_wrist_y = right_wrist.y
        left_wrist_y = left_wrist.y
        nose_y = nose.y
        
        # Check if right hand is raised (START signal)
        if (nose_y - right_wrist_y) > self.threshold:
            if self.last_gesture != 'start':
                rospy.loginfo("🟢 GESTURE DETECTED: START Following!")
                self.last_gesture = 'start'
            return 'start'
        
        # Check if left hand is raised (STOP signal)
        elif (nose_y - left_wrist_y) > self.threshold:
            if self.last_gesture != 'stop':
                rospy.loginfo("🔴 GESTURE DETECTED: STOP Following!")
                self.last_gesture = 'stop'
            return 'stop'
        
        return None
    
    def reset(self):
        """Reset gesture state."""
        self.last_gesture = None
