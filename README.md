# MPC4PX4

MPC4PX4 is a ROS package for PX4 control using Neural SDE-Based Model Predictive Control. This package enables learning-based control of PX4 using neural SDE to obtain uncertain-aware models of the quadrotor/hexacopter dynamics, and gradient-based model predictive control to perform low-level control.

## Requirements

MPC4PX4 requires ROS and the following packages at building times:

- [catkin_simple](https://github.com/catkin/catkin_simple)
- [mavros](https://github.com/mavlink/mavros)
- [eigen_catkin](https://github.com/ethz-asl/eigen_catkin)

You can install these packages by cloning their respective repositories into your catkin workspace and building the packages.

```bash
cd catkin_ws/src
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/ethz-asl/eigen_catkin.git
git clone https://github.com/wuwushrek/mpc4px4.git
cd ..
catkin build mpc4px4
```

## Usage

The package provides a basic_control.py script that allows you to control the quadrotor using a command-line interface.

# `basic_control.py`

The basic_control.py script provides a simple command-line interface to control the quadrotor. You can enter the following commands:
```
Available commands: [Spacing between arguments is important]
Available functions:
arm
disarm
takeoff alt | takeoff alt=1.0
land
pos x y z yaw | pos x=0.0 y=0.0 z=0.0 yaw=0.0
relpos dx dy dz dyaw | relpos dx=0.0 dy=0.0 dz=0.0 dyaw=0.0
offboard
controller_init config_name | controller_init config_name=config_name.yaml
controller_on
controller_off
controller_idle
controller_test
set_box  x=0.2 y=0.2 z=0.2
rm_box
ctrl_pos x=0.0 y=0.0 z=0.0 yaw=0.0
```