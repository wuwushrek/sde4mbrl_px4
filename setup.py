from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['mpc4px4'],
    package_dir={'mpc4px4': 'mpc4px4'},
)

other_dict = dict(author='Franck Djeumou',
    author_email = 'frmbouwe@gmail.com',
    description = 'A ROS package for PX4 control using Neural SDE-Based Model Predictive Control',
    license='GNU Public License'
)

setup(**{**d, **other_dict})
# setproctitle