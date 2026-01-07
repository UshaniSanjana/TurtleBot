#!/usr/bin/env python3
"""
Main ROS node that integrates all modules for human following.
Coordinates detection, recognition, perception, and control.
"""

import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import our modular components (FIXED: using relative imports with dot)
from .detectors.person_detector import PersonDetector
from .detectors.pose_estimator import PoseEstimator
from .recognition.gesture_recognizer import GestureRecognizer
from .perception.depth_processor import DepthProcessor
from .perception.obstacle_detector import ObstacleDetector
from .control.motion_controller import MotionController
from .control.safety_manager import SafetyManager
from .utils.visualization import Visualizer

class HumanFollowerNode:
    def __init__(self):
        """Initialize the human follower ROS node."""
        rospy.init_node('human_follower', anonymous=True)
        rospy.loginfo("=== TurtleBot Human Follower Starting ===")
        
        # Load parameters from config
        self.load_parameters()
        
        # Initialize all modules
        self.init_modules()
        
        # ROS communication
        self.bridge = CvBridge()
        self.depth_image = None
        
        # Performance optimization: scale down frames for faster processing
        self.processing_scale = 0.4  # Process at 40% resolution for speed
        
        # Frame skipping for MediaPipe (run every N frames)
        self.mediapipe_skip_frames = 3  # Run MediaPipe every 3rd frame
        self.frame_count = 0
        self.cached_landmarks = None  # Cache landmarks for skipped frames
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Subscribe to camera topics with queue_size=1 to prevent buffering
        # Large buff_size ensures we don't drop messages due to network issues
        rospy.Subscriber(self.color_topic, Image, self.color_callback, 
                        queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback,
                        queue_size=1, buff_size=2**24)
        
        # Robot state
        self.is_following = False  # Controlled by gestures
        
        # Person tracking lock (follow one person at a time)
        self.locked_person_box = None  # Bounding box of locked person
        self.person_lost_frames = 0  # Counter for frames without locked person
        self.max_lost_frames = 15  # Unlock after this many frames
        self.iou_threshold = 0.3  # IoU threshold for tracking same person
        
        # Temporal filtering for smooth motion (multi-frame averaging)
        from collections import deque
        self.tracking_point_buffer = deque(maxlen=5)  # Last 5 tracking points
        self.distance_buffer_temporal = deque(maxlen=5)  # Last 5 distance readings
        
        rospy.loginfo("✓ System ready! Raise RIGHT hand to start, LEFT to stop.")
        
    def load_parameters(self):
        """Load parameters from ROS parameter server."""
        # Control parameters
        self.target_dist = rospy.get_param('~target_distance', 1.5)
        self.stop_dist = rospy.get_param('~stop_distance', 0.8)
        self.max_linear = rospy.get_param('~max_linear_speed', 0.3)
        self.max_angular = rospy.get_param('~max_angular_speed', 0.5)
        self.linear_gain = rospy.get_param('~linear_gain', 0.4)
        self.angular_gain = rospy.get_param('~angular_gain', 0.003)
        
        # Detection parameters
        self.yolo_model = rospy.get_param('~yolo_model', 'yolov8n.pt')
        self.yolo_conf = rospy.get_param('~yolo_confidence', 0.5)
        self.pose_det_conf = rospy.get_param('~pose_detection_confidence', 0.5)
        self.pose_track_conf = rospy.get_param('~pose_tracking_confidence', 0.5)
        
        # Topics
        self.color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel_mux/input/teleop')
        
        # Gesture parameters
        self.gesture_threshold = rospy.get_param('~activation_threshold', 0.15)
        
    def init_modules(self):
        """Initialize all component modules."""
        # TASK A: Detection & Tracking
        self.person_detector = PersonDetector(self.yolo_model, self.yolo_conf)
        self.pose_estimator = PoseEstimator(self.pose_det_conf, self.pose_track_conf)
        
        # TASK D: Gesture Recognition
        self.gesture_recognizer = GestureRecognizer(self.gesture_threshold)
        
        # TASK B: Depth Processing
        self.depth_processor = DepthProcessor()
        
        # Obstacle Detection
        obstacle_enabled = rospy.get_param('~obstacle_detection/enabled', True)
        obstacle_min_dist = rospy.get_param('~obstacle_detection/min_distance', 0.5)
        obstacle_width = rospy.get_param('~obstacle_detection/scan_width_ratio', 0.6)
        obstacle_height = rospy.get_param('~obstacle_detection/scan_height_ratio', 0.4)
        
        self.obstacle_detector = ObstacleDetector(
            min_distance=obstacle_min_dist,
            scan_width_ratio=obstacle_width,
            scan_height_ratio=obstacle_height
        ) if obstacle_enabled else None
        
        # TASK C: Motion Control
        self.motion_controller = MotionController(
            self.cmd_vel_topic,
            self.target_dist,
            self.linear_gain,
            self.angular_gain,
            self.max_linear,
            self.max_angular
        )
        
        # TASK B & C: Safety Management
        self.safety_manager = SafetyManager(self.stop_dist, self.target_dist)
        
        # Visualization
        self.visualizer = Visualizer()
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1.xyxy[0].cpu().numpy()
        x1_2, y1_2, x2_2, y2_2 = box2.xyxy[0].cpu().numpy()
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_locked_person(self, all_persons):
        """Find the locked person in current detections using IoU overlap."""
        if self.locked_person_box is None:
            return None
        
        best_iou = 0.0
        best_match = None
        
        for person_box in all_persons:
            iou = self.calculate_iou(self.locked_person_box, person_box)
            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_match = person_box
        
        return best_match
        
    def depth_callback(self, msg):
        """Handle incoming depth images."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Depth conversion error: {e}")
            
    def color_callback(self, msg):
        """Handle incoming color images and process the pipeline."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Downscale frame for faster processing
            if self.processing_scale < 1.0:
                frame = cv2.resize(frame, None, fx=self.processing_scale, fy=self.processing_scale)
            
            self.process_frame(frame)
        except Exception as e:
            rospy.logerr(f"Color conversion error: {e}")
            
    def process_frame(self, frame):
        """
        Main processing pipeline for each frame.
        Integrates all tasks: detection, tracking, gesture, control.
        """
        if self.depth_image is None:
            rospy.logwarn_throttle(5.0, "Waiting for depth image...")
            return
        
        # Increment frame counter
        self.frame_count += 1
        
        # === TASK A: Person Detection with Locking (every frame) ===
        all_persons = self.person_detector.detect(frame)
        
        # Person locking logic: Track one person at a time
        selected_person_box = None
        
        if len(all_persons) > 0:
            if self.locked_person_box is None:
                # No locked person - select closest person to center
                selected_person_box = self.person_detector.get_closest_person(all_persons, frame.shape[1])
                self.locked_person_box = selected_person_box
                self.person_lost_frames = 0
                rospy.loginfo("🔒 Locked onto person")
            else:
                # Have locked person - try to find them in current detections
                selected_person_box = self.find_locked_person(all_persons)
                
                if selected_person_box is not None:
                    # Found locked person - update box and reset counter
                    self.locked_person_box = selected_person_box
                    self.person_lost_frames = 0
                else:
                    # Lost locked person - increment counter
                    self.person_lost_frames += 1
                    rospy.logwarn_throttle(1.0, f"Lost locked person ({self.person_lost_frames}/{self.max_lost_frames} frames)")
                    
                    if self.person_lost_frames >= self.max_lost_frames:
                        # Unlock and select new person
                        rospy.loginfo("🔓 Unlocked - selecting new person")
                        self.locked_person_box = None
                        selected_person_box = self.person_detector.get_closest_person(all_persons, frame.shape[1])
                        self.locked_person_box = selected_person_box
                        self.person_lost_frames = 0
                        rospy.loginfo("🔒 Locked onto new person")
        else:
            # No persons detected at all
            self.person_lost_frames += 1
            if self.person_lost_frames >= self.max_lost_frames:
                self.locked_person_box = None
        
        # Safety check: No selected person = stop immediately
        if selected_person_box is None:
            self.motion_controller.stop()
            self.visualizer.draw_gesture_hints(frame)
            self.visualizer.show(frame)
            self.cached_landmarks = None  # Clear cache
            return
        
        # === TASK A: Pose Estimation (skip frames for performance) ===
        # Run MediaPipe only every Nth frame on full frame (not crop)
        if self.frame_count % self.mediapipe_skip_frames == 0:
            landmarks = self.pose_estimator.estimate(frame)
            if landmarks is not None:
                self.cached_landmarks = landmarks  # Cache for next frames
        else:
            landmarks = self.cached_landmarks  # Use cached landmarks
        
        if landmarks is None:
            self.motion_controller.stop()
            self.visualizer.show(frame)
            return
        
        # Draw skeleton only if FPS is decent (conditional drawing)
        if self.current_fps < 5 or self.frame_count % 2 == 0:
            # Skip drawing on alternate frames at low FPS
            self.pose_estimator.draw_landmarks(frame, landmarks)
        
        # === TASK D: Gesture Recognition ===
        gesture = self.gesture_recognizer.recognize(landmarks)
        if gesture == 'start':
            self.is_following = True
            rospy.loginfo("🟢 GESTURE DETECTED: START Following!")
        elif gesture == 'stop':
            self.is_following = False
            rospy.loginfo("🔴 GESTURE DETECTED: STOP Following!")
        
        # === TASK B: Distance Estimation ===
        # Get tracking point with fallback for partial body visibility
        # Priority: LEFT_HIP > RIGHT_HIP > LEFT_SHOULDER > RIGHT_SHOULDER > NOSE
        tracking_point = None
        landmark_name = None
        
        # Try hips first (best for full body)
        hip_pos = self.pose_estimator.get_landmark_position(
            landmarks,
            self.pose_estimator.mp_pose.PoseLandmark.LEFT_HIP,
            frame.shape
        )
        
        if hip_pos is not None:
            tracking_point = hip_pos
            landmark_name = "LEFT_HIP"
        else:
            # Try right hip
            hip_pos = self.pose_estimator.get_landmark_position(
                landmarks,
                self.pose_estimator.mp_pose.PoseLandmark.RIGHT_HIP,
                frame.shape
            )
            if hip_pos is not None:
                tracking_point = hip_pos
                landmark_name = "RIGHT_HIP"
            else:
                # Try shoulders (for upper body only)
                shoulder_pos = self.pose_estimator.get_landmark_position(
                    landmarks,
                    self.pose_estimator.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    frame.shape
                )
                if shoulder_pos is not None:
                    tracking_point = shoulder_pos
                    landmark_name = "LEFT_SHOULDER"
                else:
                    # Try right shoulder
                    shoulder_pos = self.pose_estimator.get_landmark_position(
                        landmarks,
                        self.pose_estimator.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        frame.shape
                    )
                    if shoulder_pos is not None:
                        tracking_point = shoulder_pos
                        landmark_name = "RIGHT_SHOULDER"
                    else:
                        # Last resort: use nose (for very close/partial view)
                        nose_pos = self.pose_estimator.get_landmark_position(
                            landmarks,
                            self.pose_estimator.mp_pose.PoseLandmark.NOSE,
                            frame.shape
                        )
                        if nose_pos is not None:
                            tracking_point = nose_pos
                            landmark_name = "NOSE"
        
        if tracking_point is None:
            rospy.logwarn_throttle(1.0, "No valid tracking landmarks found")
            self.motion_controller.stop()
            self.visualizer.show(frame)
            return
        
        rospy.logdebug(f"Tracking using: {landmark_name}")
        
        # === TEMPORAL FILTERING: Smooth tracking point across frames ===
        # Add current tracking point to buffer
        self.tracking_point_buffer.append(tracking_point)
        
        # Calculate smoothed tracking point (average of last N frames)
        if len(self.tracking_point_buffer) >= 3:  # Need at least 3 frames
            smoothed_x = int(np.mean([pt[0] for pt in self.tracking_point_buffer]))
            smoothed_y = int(np.mean([pt[1] for pt in self.tracking_point_buffer]))
            hip_x, hip_y = smoothed_x, smoothed_y
        else:
            hip_x, hip_y = tracking_point  # Use raw value if not enough history
        
        # CRITICAL: Scale coordinates back to original resolution for depth lookup
        # The frame is downscaled to 0.4, but depth image is full resolution
        depth_x = int(hip_x / self.processing_scale)
        depth_y = int(hip_y / self.processing_scale)
        
        # Debug: Log depth image and coordinate info
        if self.depth_image is not None:
            depth_h, depth_w = self.depth_image.shape
            rospy.loginfo_throttle(5.0, f"Depth image size: {depth_w}x{depth_h}, Scaled coords: ({depth_x}, {depth_y}), Frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Check if coordinates are within bounds
            if not (0 <= depth_x < depth_w and 0 <= depth_y < depth_h):
                rospy.logwarn(f"Coordinates OUT OF BOUNDS! depth_x={depth_x}, depth_y={depth_y}, depth size={depth_w}x{depth_h}")
            else:
                # Coordinates are valid, check actual depth value
                depth_value = self.depth_image[depth_y, depth_x]
                rospy.loginfo_throttle(2.0, f"Depth pixel value at ({depth_x}, {depth_y}): {depth_value}")
        else:
            rospy.logwarn_throttle(2.0, "Depth image is None!")
        
        # Get distance using scaled coordinates
        raw_distance = self.depth_processor.get_distance(self.depth_image, depth_x, depth_y)
        
        # === TEMPORAL FILTERING: Smooth distance across frames ===
        if raw_distance is not None:
            self.distance_buffer_temporal.append(raw_distance)
            
            # Calculate smoothed distance (median of last N frames for robustness)
            if len(self.distance_buffer_temporal) >= 3:
                distance = float(np.median(list(self.distance_buffer_temporal)))
            else:
                distance = raw_distance
        else:
            distance = None
        
        # Debug logging with None-safe formatting
        dist_str = f"{distance:.2f}m" if distance is not None else "None"
        rospy.logdebug(f"Hip pos (scaled frame): ({hip_x}, {hip_y}), Depth lookup: ({depth_x}, {depth_y}), Distance: {dist_str}")
        
        # === TASK B & C: Safety Checks ===
        is_safe = self.safety_manager.check_distance_safety(distance)
        
        # === Obstacle Detection ===
        has_obstacle = False
        obstacle_distance = None
        
        if self.obstacle_detector is not None and self.locked_person_box is not None:
            # Get person bounding box to exclude from obstacle detection
            person_bbox = self.locked_person_box.xyxy[0].cpu().numpy()
            
            # Check for obstacles in path
            has_obstacle, obstacle_distance, _ = self.obstacle_detector.detect_obstacle(
                self.depth_image, person_bbox
            )
        
        # === TASK C: Motion Control ===
        if self.is_following and distance:
            # Determine if we can move forward
            can_move_forward = True
            
            if has_obstacle and distance > self.target_dist:
                # Obstacle ahead and trying to move forward - STOP
                can_move_forward = False
                rospy.logwarn_throttle(1.0, f"⚠️ OBSTACLE AHEAD at {obstacle_distance:.2f}m - Stopping")
            
            if is_safe and can_move_forward:
                # Normal following mode - safe distance, no obstacles
                rospy.loginfo_throttle(1.0, f"FOLLOWING: distance={distance:.2f}m, hip_x={hip_x}")
                self.motion_controller.follow(distance, hip_x, frame.shape[1])
            elif not is_safe:
                # Too close - will back up smoothly (ignore obstacles when backing up)
                rospy.logwarn_throttle(1.0, f"TOO CLOSE: Backing up (distance={distance:.2f}m)")
                self.motion_controller.follow(distance, hip_x, frame.shape[1])
            else:
                # Obstacle blocking forward movement
                self.motion_controller.stop()
        else:
            # Stopped mode (gesture off or invalid distance)
            if not self.is_following:
                rospy.loginfo_throttle(2.0, "NOT MOVING: Waiting for START gesture (raise RIGHT hand)")
            elif not distance:
                rospy.logwarn_throttle(1.0, "NOT MOVING: No valid distance measurement")
            self.motion_controller.stop()
        
        # === Visualization ===
        self.visualizer.draw_tracking_info(
            frame, hip_pos, distance, self.is_following, is_safe, self.current_fps
        )
        self.visualizer.draw_gesture_hints(frame)
        self.visualizer.show(frame)
    
    def shutdown(self):
        """Clean shutdown of the node."""
        rospy.loginfo("Shutting down human follower...")
        self.motion_controller.stop()
        self.visualizer.close()

def main():
    """Main entry point."""
    try:
        node = HumanFollowerNode()
        
        # Register shutdown hook
        rospy.on_shutdown(node.shutdown)
        
        # Keep node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")

if __name__ == '__main__':
    main()
