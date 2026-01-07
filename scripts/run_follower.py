## File: scripts/run_follower.py

#!/usr/bin/env python3
"""
Entry point script to launch the human follower node.
"""

import sys
import os

# Add package to Python path
pkg_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, pkg_path)

from turtlebot_human_follower.main_node import main

if __name__ == '__main__':
    main()


