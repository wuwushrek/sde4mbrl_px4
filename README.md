# sde4mbrl_px4

sde4mbrl_px4 is a ROS package for PX4 control using Neural SDE-Based Model Predictive Control. This package enables learning-based control of PX4 using neural SDE to obtain uncertain-aware models of the quadrotor/hexacopter dynamics, and gradient-based model predictive control to perform low-level control.

## Requirements

sde4mbrl_px4 requires ROS and the following packages at building times:

- [catkin_simple](https://github.com/catkin/catkin_simple)
- [Mavros with custom MPC mesages](https://github.com/wuwushrek/mavros), please follow the instructions in the mpc_franck branch.
- [eigen_catkin](https://github.com/ethz-asl/eigen_catkin)

You can install these packages by cloning their respective repositories into your catkin workspace and building the packages.

```bash
cd catkin_ws/src
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/ethz-asl/eigen_catkin.git
git clone https://github.com/wuwushrek/sde4mbrl_px4.git
cd ..
catkin build sde4mbrl_px4
```

## Setting up the PX4 SITL
We need separate terminals for the PX4 SITL, mavros setup, the high-level basic controller, and the low-level MPC controller.

1. Start the PX4 SITL in one terminal:
```bash
cd ~/Documents/PX4-Autopilot
make px4_sitl gazebo # For iris
# make px4_sitl gazebo_myhexa # For the hexacopter
```

2. Start the mavros setup, including mavlink-router to retarget the mavlink messages to the MPC controller, in another terminal:
- Mavros helps us to communicate with the PX4 SITL via ROS. 
- Mavlink-router is used to retarget some of the mavlink messages to the MPC controller. 
- The mavlink-router is a tool that allows to route mavlink messages from one endpoint to another. 
- In our case, we want to route the mavlink messages MPC_FULL_STATE from the PX4 SITL to the MPC controller, without going through ROS for avoiding latency. 
- The PX4 SITL is running on the port 14540, and the MPC controller is running on the port 14998.
- So, mavlink-router is going to listen to the port 14540 and forward some messages to the port 14998, and the full messages on 14999 where now mavros is listening to.
- The router setting is given in the files `scripts/sitl_route_mavlink.sh` and `scripts/router_sitl.conf`.
- Mavros launch file is given in the file `launch/px4_sitl.launch`, where the mavlink-router is started and the mavros node is started.

```bash
roslaunch sde4mbrl_px4 px4_sitl.launch
```

3. Start the high-level basic controller in another terminal:
The high-level basic controller is used to send commands to the PX4 SITL, or initialize the MPC controller.
The basic controller is given in the file `sde4mbrl_px4/basic_control.py`.
```bash
cd ~/catkin_ws/src/sde4mbrl_px4/sde4mbrl_px4
python basic_control.py
```

4. Start the low-level MPC controller in another terminal:
```bash
roslaunch sde4mbrl_px4 iris_sdectrl.launch # For iris
# roslaunch sde4mbrl_px4 hexa_sitl_sdectrl.launch # For the hexacopter
```
The controller settings are given in the file `sde4mbrl_px4/config/iris_sdectrl.yaml`.


## Setting up in the AHG Lab



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