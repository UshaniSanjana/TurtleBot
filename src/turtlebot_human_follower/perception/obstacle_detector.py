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
    
    def detect_obstacle_direction(self, depth_image, person_bbox=None):
        """
        Detect obstacles and determine best avoidance direction.
        Scans LEFT, CENTER, RIGHT regions of the path.
        
        Args:
            depth_image: Depth image (16UC1 format)
            person_bbox: Bounding box of tracked person to exclude [x1, y1, x2, y2]
            
        Returns:
            tuple: (direction, min_distance)
                direction: 'clear', 'turn_left', 'turn_right', or 'blocked'
                min_distance: Distance to nearest obstacle in meters
        """
        if depth_image is None:
            return 'clear', None
        
        h, w = depth_image.shape
        
        # Define scan regions (bottom 50% of frame)
        scan_height = int(h * self.scan_height_ratio)
        y_start = h - scan_height
        y_end = h
        
        # Divide width into LEFT (0-30%), CENTER (30-70%), RIGHT (70-100%)
        left_start = 0
        left_end = int(w * 0.3)
        center_start = int(w * 0.3)
        center_end = int(w * 0.7)
        right_start = int(w * 0.7)
        right_end = w
        
        # Extract regions
        left_region = depth_image[y_start:y_end, left_start:left_end].copy()
        center_region = depth_image[y_start:y_end, center_start:center_end].copy()
        right_region = depth_image[y_start:y_end, right_start:right_end].copy()
        
        # Exclude person from all regions if provided
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            
            # Mask out person in each region
            for region, region_start in [(left_region, left_start), 
                                         (center_region, center_start), 
                                         (right_region, right_start)]:
                px1 = max(0, int(x1) - region_start)
                py1 = max(0, int(y1) - y_start)
                px2 = min(region.shape[1], int(x2) - region_start)
                py2 = min(region.shape[0], int(y2) - y_start)
                
                if px1 < region.shape[1] and py1 < region.shape[0] and px2 > 0 and py2 > 0:
                    region[py1:py2, px1:px2] = 10000  # Ignore person
        
        # Get minimum distances in each region
        def get_min_distance(region):
            valid = region[(region > 0) & (region < 10000)]
            if len(valid) == 0:
                return float('inf')
            return np.min(valid) * self.depth_scale
        
        left_dist = get_min_distance(left_region)
        center_dist = get_min_distance(center_region)
        right_dist = get_min_distance(right_region)
        
        min_dist = min(left_dist, center_dist, right_dist)
        
        # Improved decision logic - turn away from obstacles even on sides
        if left_dist <= self.min_distance and right_dist > self.min_distance:
            # Obstacle on LEFT - turn RIGHT
            return 'turn_right', min_dist
        elif right_dist <= self.min_distance and left_dist > self.min_distance:
            # Obstacle on RIGHT - turn LEFT
            return 'turn_left', min_dist
        elif center_dist <= self.min_distance:
            # Center blocked - turn toward clearer side
            if left_dist > self.min_distance and right_dist > self.min_distance:
                # Both sides clear - turn toward clearer side
                if left_dist > right_dist:
                    return 'turn_left', min_dist
                else:
                    return 'turn_right', min_dist
            elif left_dist > self.min_distance:
                # Only left clear
                return 'turn_left', min_dist
            elif right_dist > self.min_distance:
                # Only right clear
                return 'turn_right', min_dist
            else:
                # All blocked
                return 'blocked', min_dist
        else:
            # All clear
            return 'clear', min_dist
