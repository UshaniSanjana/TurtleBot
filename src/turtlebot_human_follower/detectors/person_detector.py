## File: src/turtlebot_human_follower/detectors/person_detector.py

#!/usr/bin/env python3
"""
TASK A: Person Detection using YOLO
Detects humans in RGB images and returns bounding boxes.
"""

import rospy
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.7):  # Increased from 0.5
        """
        Initialize YOLO person detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold (0.7 reduces false positives)
        """
        rospy.loginfo(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect(self, frame):
        """
        Detect persons in the frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            list: List of detection results, empty if no person found
        """
        # Detect only person class (class=0 in COCO dataset)
        # Optimizations for real-time performance:
        # - imgsz=416: Smaller input size for faster inference
        # - device='cpu': Use CPU (change to '0' if you have CUDA GPU)
        # - augment=False: Disable test-time augmentation
        results = self.model(
            frame, 
            classes=0, 
            conf=self.confidence, 
            verbose=False,
            imgsz=416,
            device='cpu',
            augment=False
        )
        
        if len(results[0].boxes) > 0:
            rospy.logdebug(f"Detected {len(results[0].boxes)} person(s)")
            return results[0].boxes
        
        return []
    
    def get_closest_person(self, boxes, frame_width):
        """
        Select the closest person to the center of the frame.
        
        Args:
            boxes: List of detection boxes
            frame_width: Width of the frame
            
        Returns:
            box: The box closest to frame center, or None
        """
        if len(boxes) == 0:
            return None
        
        frame_center = frame_width / 2.0
        min_dist = float('inf')
        closest_box = None
        
        for box in boxes:
            # Get bounding box center
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_center_x = (x1 + x2) / 2.0
            
            # Calculate distance from frame center
            dist = abs(box_center_x - frame_center)
            
            if dist < min_dist:
                min_dist = dist
                closest_box = box
        
        return closest_box
    
    def has_person(self, frame):
        """
        Quick check if any person is detected.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            bool: True if at least one person detected
        """
        detections = self.detect(frame)
        return len(detections) > 0
