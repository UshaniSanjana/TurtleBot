## File: setup.py

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['turtlebot_human_follower'],
    package_dir={'': 'src'}
)

setup(**d)
```
