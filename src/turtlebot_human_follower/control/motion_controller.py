## File: src/turtlebot_human_follower/control/motion_controller.py

#!/usr/bin/env python3
"""
TASK C: Improved Robot Motion Control with Smoothing
Controls TurtleBot velocity commands for smooth, stable following.
"""

import rospy
from geometry_msgs.msg import Twist
import numpy as np

class MotionController:
    def __init__(self, cmd_vel_topic, target_dist, linear_gain, angular_gain,
                 max_linear, max_angular):
        """
        Initialize motion controller with smoothing.
        
        Args:
            cmd_vel_topic: ROS topic for velocity commands
            target_dist: Desired following distance
            linear_gain: PID gain for forward/backward
            angular_gain: PID gain for turning
            max_linear: Maximum linear speed
            max_angular: Maximum angular speed
        """
        self.target_distance = target_dist
        self.linear_k = linear_gain
        self.angular_k = angular_gain
        self.max_linear = max_linear
        self.max_angular = max_angular
        
        # Smoothing parameters (balanced for smooth yet responsive motion)
        self.smoothing_alpha = 0.45  # Increased from 0.4 for backward smoothness
        self.deadzone_distance = 0.12  # Slightly tighter deadzone
        self.deadzone_angular = 20  # Increased from 15px for more stable centering
        
        # Velocity buffers for smoothing
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        
        # Moving average buffer for distance (balanced)
        self.distance_buffer = []
        self.distance_buffer_size = 3  # Increased from 2 for smoother distance tracking
        
        # PID components
        self.integral_linear = 0.0
        self.prev_error_linear = 0.0
        self.integral_angular = 0.0
        self.prev_error_angular = 0.0
        
        # PID gains (tune these for better performance)
        self.kp_linear = linear_gain  # Proportional
        self.ki_linear = 0.01         # Integral (prevents steady-state error)
        self.kd_linear = 0.05         # Derivative (reduces oscillation)
        
        self.kp_angular = angular_gain
        self.ki_angular = 0.0005
        self.kd_angular = 0.002
        
        # Acceleration limits (prevents sudden jerks)
        self.max_linear_accel = 0.25  # Reduced from 0.3 for smoother acceleration
        self.max_angular_accel = 0.6  # Reduced from 0.8 for smoother turns
        self.dt = 0.1  # Reduced from 0.15 for better control granularity
        
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        rospy.loginfo(f"Motion controller initialized. Target distance: {target_dist}m")
        
    def follow(self, distance, person_center_x, frame_width):
        """
        Generate smooth velocity commands to follow person.
        
        Args:
            distance: Current distance to person (meters)
            person_center_x: X coordinate of person center
            frame_width: Width of camera frame
        """
        # Smooth distance using moving average
        self.distance_buffer.append(distance)
        if len(self.distance_buffer) > self.distance_buffer_size:
            self.distance_buffer.pop(0)
        smoothed_distance = np.median(self.distance_buffer)
        
        # === LINEAR CONTROL (Forward/Backward) ===
        distance_error = smoothed_distance - self.target_distance
        
        # Deadzone: Don't move if close enough to target
        if abs(distance_error) < self.deadzone_distance:
            linear_vel = 0.0
            self.integral_linear = 0.0  # Reset integral when in deadzone
        else:
            # PID control for linear velocity
            self.integral_linear += distance_error * self.dt
            self.integral_linear = np.clip(self.integral_linear, -1.0, 1.0)  # Anti-windup
            
            derivative_linear = (distance_error - self.prev_error_linear) / self.dt
            
            linear_vel = (self.kp_linear * distance_error + 
                         self.ki_linear * self.integral_linear +
                         self.kd_linear * derivative_linear)
            
            self.prev_error_linear = distance_error
        
        # === ANGULAR CONTROL (Turning) ===
        frame_center = frame_width / 2.0
        angular_error = frame_center - person_center_x
        
        # Deadzone: Don't turn if person is already centered
        if abs(angular_error) < self.deadzone_angular:
            angular_vel = 0.0
            self.integral_angular = 0.0
        else:
            # PID control for angular velocity
            self.integral_angular += angular_error * self.dt
            self.integral_angular = np.clip(self.integral_angular, -100.0, 100.0)
            
            derivative_angular = (angular_error - self.prev_error_angular) / self.dt
            
            angular_vel = (self.kp_angular * angular_error +
                          self.ki_angular * self.integral_angular +
                          self.kd_angular * derivative_angular)
            
            self.prev_error_angular = angular_error
        
        # === VELOCITY SMOOTHING (Exponential Moving Average) ===
        linear_vel = self.smooth_velocity(linear_vel, self.prev_linear, self.smoothing_alpha)
        angular_vel = self.smooth_velocity(angular_vel, self.prev_angular, self.smoothing_alpha)
        
        # === ACCELERATION LIMITING (Prevents jerks) ===
        linear_vel = self.limit_acceleration(
            linear_vel, self.prev_linear, self.max_linear_accel, self.dt
        )
        angular_vel = self.limit_acceleration(
            angular_vel, self.prev_angular, self.max_angular_accel, self.dt
        )
        
        # === VELOCITY CLAMPING ===
        # Allow full backward speed for smooth backing up when too close
        linear_vel = np.clip(linear_vel, -self.max_linear, self.max_linear)
        angular_vel = np.clip(angular_vel, -self.max_angular, self.max_angular)
        
        # === TURN-AWARE SPEED SCALING ===
        # Reduce linear speed during sharp turns for tighter turning radius
        # Only apply to forward movement, not backward
        if linear_vel > 0:  # Only for forward movement
            angular_ratio = abs(angular_vel) / self.max_angular if self.max_angular > 0 else 0
            if angular_ratio > 0.3:  # If turning significantly
                # Scale down linear speed proportionally to turn sharpness
                turn_scale = 1.0 - (angular_ratio * 0.5)  # Reduce up to 50% during max turn
                linear_vel *= max(turn_scale, 0.5)  # Keep at least 50% speed
        
        # Store for next iteration
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        # Publish command
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
        
        rospy.logdebug(f"Cmd: lin={linear_vel:.2f}, ang={angular_vel:.2f}, dist_err={distance_error:.2f}")
        
    def smooth_velocity(self, new_vel, prev_vel, alpha):
        """
        Exponential moving average for smooth velocity transitions.
        
        Args:
            new_vel: Newly calculated velocity
            prev_vel: Previous velocity
            alpha: Smoothing factor (0-1, lower = smoother)
            
        Returns:
            float: Smoothed velocity
        """
        return alpha * new_vel + (1 - alpha) * prev_vel
    
    def limit_acceleration(self, target_vel, current_vel, max_accel, dt):
        """
        Limit acceleration to prevent sudden jerky movements.
        
        Args:
            target_vel: Desired velocity
            current_vel: Current velocity
            max_accel: Maximum allowed acceleration
            dt: Time step
            
        Returns:
            float: Velocity limited by max acceleration
        """
        max_delta = max_accel * dt
        delta = target_vel - current_vel
        
        if abs(delta) > max_delta:
            return current_vel + np.sign(delta) * max_delta
        return target_vel
    
    def stop(self):
        """Smoothly stop the robot (gradual deceleration)."""
        # Gradually reduce to zero over more iterations for smoother stop
        for i in range(5):  # Increased from 3 for smoother deceleration
            self.prev_linear *= 0.6  # Gentler reduction (was 0.5)
            self.prev_angular *= 0.6
            
            twist = Twist()
            twist.linear.x = self.prev_linear
            twist.angular.z = self.prev_angular
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.04)  # Slightly longer intervals
        
        # Final stop
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        # Reset states
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.prev_error_linear = 0.0
        self.prev_error_angular = 0.0
        self.distance_buffer = []
        
        rospy.logdebug("Robot stopped smoothly")
