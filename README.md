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

### `basic_control.py`

The `basic_control.py` script provides the following commands:

```
Available commands: [Spacing between arguments is important]
Available functions:

- `arm`: Arm the quadrotor.
- `disarm`: Disarm the quadrotor.
- `takeoff alt | takeoff alt=1.0`: Take off the quadrotor to a specified altitude (`alt` in meters) or to a default altitude of 1.0 meter if not specified.
- `land`: Land the quadrotor.
- `pos x y z yaw | pos x=0.0 y=0.0 z=0.0 yaw=0.0`: Set the quadrotor to a specified position (`x`, `y`, `z` in meters, and `yaw` in radians) or to a default position of (0.0, 0.0, 0.0, 0.0) if not specified. Use PID of PX4.
- `relpos dx dy dz dyaw | relpos dx=0.0 dy=0.0 dz=0.0 dyaw=0.0`: Set the quadrotor to a position relative to its current position (`dx`, `dy`, `dz` in meters, and `dyaw` in radians) or to a default relative position of (0.0, 0.0, 0.0, 0.0) if not specified. Use PID of PX4
- `offboard`: Switch to offboard mode.
- `controller_init config_name | controller_init config_name=config_name.yaml`: Initialize the controller with a specified configuration file (`config_name`) or with a default configuration file if not specified.
- `controller_on`: Turn the trajectory controller on. For instance the MPC controller
- `controller_off`: Turn the controller off. For instance, the MPC controller
- `controller_idle`: Put the controller in idle mode. This pushes the MPC controller to move to the initial point in the trajectory and wait there until controller_on is sent.
- `controller_test`: Test the controller. COmpute output of MPC and send it to PX4, but PX4 won't use it.
- `set_box  x=0.2 y=0.2 z=0.2`: Set the safety box size for returning to PID if MPC fails  (`x`, `y`, `z` in meters).
- `rm_box`: Remove the safety box.
- `ctrl_pos x=0.0 y=0.0 z=0.0 yaw=0.0`: Control the vehicle to a specified position (`x`, `y`, `z` in meters, and `yaw` in radians). Note that this command requires the controller to be initialized.
```
