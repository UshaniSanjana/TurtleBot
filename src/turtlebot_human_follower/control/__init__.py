"""
Control modules for robot motion and safety management.

Modules:
- motion_controller: PID-based velocity control for smooth following
- safety_manager: Safety zone monitoring and emergency stops
"""

from .motion_controller import MotionController
from .safety_manager import SafetyManager

__all__ = ['MotionController', 'SafetyManager']
