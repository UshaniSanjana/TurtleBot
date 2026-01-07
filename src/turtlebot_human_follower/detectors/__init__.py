"""
Detection modules for person and pose detection.

Modules:
- person_detector: YOLO-based person detection
- pose_estimator: MediaPipe-based pose estimation and skeleton tracking
"""

from .person_detector import PersonDetector
from .pose_estimator import PoseEstimator

__all__ = ['PersonDetector', 'PoseEstimator']
