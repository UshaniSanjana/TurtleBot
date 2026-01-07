## File: src/turtlebot_human_follower/control/safety_manager.py

#!/usr/bin/env python3
"""
TASK B & C: Safety Zone Management
Monitors safety conditions and triggers emergency stops.
"""

import rospy

class SafetyManager:
    def __init__(self, stop_distance, target_distance):
        """
        Initialize safety manager.
        
        Args:
            stop_distance: Minimum safe distance (meters)
            target_distance: Target following distance (meters)
        """
        self.stop_distance = stop_distance
        self.target_distance = target_distance
        self.person_lost_count = 0
        self.max_lost_frames = 10  # Stop after losing person for this many frames
        
    def check_distance_safety(self, distance):
        """
        Check if distance is safe.
        
        Args:
            distance: Current distance in meters
            
        Returns:
            bool: True if safe to move, False if should stop
        """
        if distance is None:
            rospy.logwarn("Invalid distance reading - stopping for safety")
            return False
            
        if distance < self.stop_distance:
            rospy.logwarn(f"TOO CLOSE! Distance: {distance:.2f}m < {self.stop_distance}m")
            return False
            
        return True
    
    def check_person_tracking(self, person_detected):
        """
        Check if person is still being tracked.
        
        Args:
            person_detected: Boolean indicating if person is detected
            
        Returns:
            bool: True if safe to continue, False if person lost
        """
        if not person_detected:
            self.person_lost_count += 1
            if self.person_lost_count > self.max_lost_frames:
                rospy.logwarn("Person lost - stopping robot")
                return False
        else:
            self.person_lost_count = 0
            
        return True
    
    def reset(self):
        """Reset safety counters."""
        self.person_lost_count = 0
