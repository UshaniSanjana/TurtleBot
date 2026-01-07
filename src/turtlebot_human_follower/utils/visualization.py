## File: src/turtlebot_human_follower/utils/visualization.py

#!/usr/bin/env python3
"""
Visualization utilities for debugging and display.
"""

import cv2

class Visualizer:
    def __init__(self):
        """Initialize visualizer."""
        self.window_name = "TurtleBot Human Follower"
        self.window_created = False
        
    def draw_tracking_info(self, frame, position, distance, is_active, is_safe, fps=0.0, has_obstacle=False, obstacle_distance=None):
        """
        Draw tracking information on frame.
        
        Args:
            frame: BGR image
            position: (x, y) tuple of person position
            distance: Distance in meters
            is_active: Whether robot is actively following
            is_safe: Whether current state is safe
            fps: Current frames per second
            has_obstacle: Whether obstacle is detected
            obstacle_distance: Distance to obstacle in meters
        """
        if position is None:
            return
            
        x, y = position
        
        # Choose color based on state
        if not is_safe:
            color = (0, 0, 255)  # Red - danger
            status = "DANGER"
        elif is_active:
            color = (0, 255, 0)  # Green - active
            status = "FOLLOWING"
        else:
            color = (0, 165, 255)  # Orange - standby
            status = "STANDBY"
        
        # Draw tracking circle (smaller)
        cv2.circle(frame, (x, y), 4, color, -1)  # Reduced from 10
        cv2.circle(frame, (x, y), 8, color, 1)  # Reduced from 15, thickness 2→1
        
        # Draw distance text (smaller font)
        if distance:
            cv2.putText(frame, f"{distance:.2f}m", (x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # Size 0.5→0.3, thickness 2→1
        
        # Draw status (smaller font)
        cv2.putText(frame, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Font size 0.8, thickness 2
        
        # Draw FPS in top-right corner (smaller)
        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]  # Size 0.7→0.5, thickness 2→1
            fps_x = frame.shape[1] - text_size[0] - 10
            cv2.putText(frame, fps_text, (fps_x, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Size 0.7→0.5, thickness 2→1
        
        # Draw obstacle warning if detected
        if has_obstacle and obstacle_distance is not None:
            # warning_text = f"⚠ OBSTACLE: {obstacle_distance:.2f}m"
            warning_text = f"OBSTACLE DETECTED: {obstacle_distance:.2f}m"
            cv2.putText(frame, warning_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red warning
        
    def draw_gesture_hints(self, frame):
        """Draw gesture control hints (smaller font)."""
        hints = [
            "Right Hand UP = START",
            "Left Hand UP = STOP"
        ]
        
        y_offset = frame.shape[0] - 40  # Reduced from 60
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (10, y_offset + i * 18),  # Reduced spacing from 25
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)  # Size 0.6→0.4, thickness 2→1
    
    def show(self, frame):
        """Display frame."""
        # Create and resize window on first use
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            self.window_created = True
        
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)
    
    def close(self):
        """Close visualization windows."""
        cv2.destroyAllWindows()
