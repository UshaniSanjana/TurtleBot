#!/usr/bin/env python3
"""
Obstacle Detection Module
Detects obstacles in robot's path using depth camera.
"""

import numpy as np
import rospy

class ObstacleDetector:
    def __init__(self, min_distance=0.5, scan_width_ratio=0.6, scan_height_ratio=0.4):
        """
        Initialize obstacle detector.
        
        Args:
            min_distance: Minimum safe distance to obstacles (meters)
            scan_width_ratio: Width of scan region (0-1, centered)
            scan_height_ratio: Height of scan region (0-1, from bottom)
        """
        self.min_distance = min_distance
        self.scan_width_ratio = scan_width_ratio
        self.scan_height_ratio = scan_height_ratio
        self.depth_scale = 0.001  # Convert mm to meters
        
        rospy.loginfo(f"Obstacle detector initialized. Min distance: {min_distance}m")
    
    def detect_obstacle(self, depth_image, person_bbox=None):
        """
        Detect obstacles in robot's path.
        
        Args:
            depth_image: Depth image (16UC1 format)
            person_bbox: Bounding box of tracked person to exclude [x1, y1, x2, y2]
            
        Returns:
            tuple: (has_obstacle, min_distance, obstacle_region)
        """
        if depth_image is None:
            return False, None, None
        
        h, w = depth_image.shape
        
        # Define scan region (center-bottom of frame)
        scan_width = int(w * self.scan_width_ratio)
        scan_height = int(h * self.scan_height_ratio)
        
        x_start = (w - scan_width) // 2
        x_end = x_start + scan_width
        y_start = h - scan_height
        y_end = h
        
        # Extract scan region
        scan_region = depth_image[y_start:y_end, x_start:x_end].copy()
        
        # Exclude tracked person from scan region
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            
            # Convert person bbox to scan region coordinates
            px1 = max(0, int(x1) - x_start)
            py1 = max(0, int(y1) - y_start)
            px2 = min(scan_width, int(x2) - x_start)
            py2 = min(scan_height, int(y2) - y_start)
            
            # Mask out person region (set to max value to ignore)
            if px1 < scan_width and py1 < scan_height and px2 > 0 and py2 > 0:
                scan_region[py1:py2, px1:px2] = 10000  # High value = ignore
        
        # Filter valid depth values
        valid_depths = scan_region[(scan_region > 0) & (scan_region < 10000)]
        
        if len(valid_depths) == 0:
            return False, None, None
        
        # Find minimum distance in scan region
        min_depth_mm = np.min(valid_depths)
        min_distance = min_depth_mm * self.depth_scale
        
        # Check if obstacle is too close
        has_obstacle = min_distance < self.min_distance
        
        if has_obstacle:
            rospy.logdebug(f"Obstacle detected at {min_distance:.2f}m")
        
        return has_obstacle, min_distance, (x_start, y_start, x_end, y_end)
    
    def get_scan_region_bounds(self, depth_image):
        """
        Get the bounds of the scan region for visualization.
        
        Returns:
            tuple: (x_start, y_start, x_end, y_end)
        """
        if depth_image is None:
            return None
        
        h, w = depth_image.shape
        
        scan_width = int(w * self.scan_width_ratio)
        scan_height = int(h * self.scan_height_ratio)
        
        x_start = (w - scan_width) // 2
        x_end = x_start + scan_width
        y_start = h - scan_height
        y_end = h
        
        return (x_start, y_start, x_end, y_end)
