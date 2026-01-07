## File: src/turtlebot_human_follower/perception/depth_processor.py

#!/usr/bin/env python3
"""
TASK B: Improved Depth-Based Distance Estimation with Filtering
Calculates stable, filtered distance to person using depth camera.
"""

import numpy as np
import rospy
from collections import deque

class DepthProcessor:
    def __init__(self):
        """Initialize depth processor with temporal filtering."""
        self.depth_scale = 0.001  # Convert mm to meters for RealSense
        
        # Temporal filtering (smooths over time)
        self.distance_history = deque(maxlen=7)  # Reduced from 10 for faster response
        
        # Outlier rejection parameters
        self.max_depth_change = 0.4  # Reduced from 0.5 for stricter rejection
        self.prev_valid_distance = None
        
    def get_distance(self, depth_image, x, y, window_size=25):
        """
        Get stable, filtered distance at specific pixel coordinates.
        
        Args:
            depth_image: Depth image (16UC1 format)
            x, y: Pixel coordinates
            window_size: Size of averaging window for spatial stability
            
        Returns:
            float: Filtered distance in meters, or None if invalid
        """
        if depth_image is None:
            return None
            
        h, w = depth_image.shape
        
        # Validate coordinates
        if not (0 <= x < w and 0 <= y < h):
            return None
        
        # === SPATIAL FILTERING: Use window average ===
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        
        depth_window = depth_image[y_min:y_max, x_min:x_max]
        
        # Filter out invalid depth values with stricter range
        # Min: 300mm (30cm), Max: 5000mm (5m) - filters sensor noise
        valid_depths = depth_window[(depth_window > 300) & (depth_window < 5000)]
        
        if len(valid_depths) == 0:
            rospy.logdebug("No valid depth readings in window")
            return self.prev_valid_distance  # Return last known good value
        
        # Use median for robustness against outliers
        raw_distance = np.median(valid_depths) * self.depth_scale
        
        # === OUTLIER REJECTION ===
        if self.prev_valid_distance is not None:
            distance_change = abs(raw_distance - self.prev_valid_distance)
            
            # Reject sudden jumps (likely noise or occlusion)
            if distance_change > self.max_depth_change:
                rospy.logdebug(f"Outlier rejected: change={distance_change:.2f}m")
                return self.prev_valid_distance
        
        # === TEMPORAL FILTERING: Moving average over time ===
        self.distance_history.append(raw_distance)
        
        if len(self.distance_history) < 4:  # Need at least 4 samples
            # Not enough history yet, use raw value
            filtered_distance = raw_distance
        else:
            # Use median of history (very robust to outliers)
            filtered_distance = float(np.median(list(self.distance_history)))
        
        # Update last valid distance
        self.prev_valid_distance = filtered_distance
        
        return filtered_distance
    
    def is_in_safety_zone(self, distance, stop_distance):
        """
        Check if distance is within safety zone.
        
        Args:
            distance: Current distance in meters
            stop_distance: Minimum safe distance
            
        Returns:
            bool: True if too close (in danger zone)
        """
        if distance is None:
            return True  # Assume danger if no valid reading
            
        return distance < stop_distance
    
    def reset(self):
        """Reset filtering state (call when person is lost)."""
        self.distance_history.clear()
        self.prev_valid_distance = None
