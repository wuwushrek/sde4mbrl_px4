from importlib.util import LazyLoader
import rclpy
from rclpy.node import Node

import numpy as np

# IMport px4_msgs
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleControlMode
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import Timesync
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import TrajectorySetpoint

# Contain useful functions for changing coordinate frames
import helper

# Contain the command line interface
from input_command import handle_user_input

import threading
    

class DummyLogger():
    """ A dummy ros2 logger that does nothing """
    def __init__(self):
        pass
    def info(self, msg, *l_args, **args):
        pass
    def warn(self, msg, *l_args, **args):
        pass
    def error(self, msg, *l_args, **args):
        pass
    def debug(self, msg, *l_args, **args):
        pass
    def fatal(self, msg, *l_args, **args):
        pass

        
class BasicControl(Node):
    """ A basic control class that can be used to control the drone """
    def __init__(self):
        # TODO : Add ros2 parameters for wait frequency and others
        super().__init__('basic_control')

        # Store a dummy logger
        self.dummy_logger = DummyLogger()

        # Save the promp state
        self.userin = False

        # Log out that the class has been initialized
        self.get_logger().info("BasicControl node has been started")

        ############ Log messages ############
        # Offboard control mode
        self.offboard_control_mode = OffboardControlMode()
        # Vehicle control mode
        self.vehicle_control_mode = VehicleControlMode()
        # time sync
        self.timesync_ = Timesync()
        # Vehcile command
        self.vehicle_command = VehicleCommand()
        # Vehicle odometry
        self.vehicle_odometry = VehicleOdometry()
        # Vehicle local position setpoint
        self.setpoint = TrajectorySetpoint()

        ############ Internal states ############
        # Stop offboard mode when needed
        self.stop_offboard_mode = False

        # Create a rate of 50Hz for functions requiring waiting in a loop
        self.rate = self.create_rate(50)

        # Is there a command to send?
        self.command_to_send = False
        # Has the command been sent?
        self.command_succeed = lambda : False
        # Has an action been completed?
        self.action_completed = lambda : False
        self.default_no_action = lambda : False

        ############ Subscribers ############
        # Create a subscriber to the vehicle control mode topic
        self.vehicle_control_sub = self.create_subscription(
            VehicleControlMode,
            'fmu/vehicle_control_mode/out',
            self.control_mode_callback,
            10)
        self.vehicle_control_sub   # prevent unused variable warning

        # Create a subscriber to the time sync topic
        self.timesync_sub = self.create_subscription(
            Timesync,
            'fmu/timesync/out',
            self.timesync_callback,
            10)
        self.timesync_sub   # prevent unused variable warning

        # Create a subscriber to the vehicle odometry topic
        self.vehicle_odometry_sub = self.create_subscription(
            VehicleOdometry,
            'fmu/vehicle_odometry/out',
            self.odometry_callback,
            10)
        self.vehicle_odometry_sub   # prevent unused variable warning

        ############ Publishers ############
        # Create a publisher to the offboard control mode topic
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            'fmu/offboard_control_mode/in',
            10)
        
        # Create a publisher to the vehicle command topic
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            'fmu/vehicle_command/in',
            10)
        
        # Create a publisher to the trajectory setpoint topic
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            'fmu/trajectory_setpoint/in',
            10)

        # Create a timer to send the offboard control mode, 20Hz timer callback
        self.timer_offboard = self.create_timer(0.1, self.global_publisher)
    
    def get_logger(self):
        """ Return the logger """
        if self.userin:
            return self.dummy_logger
        return super().get_logger()
    
    def wait_for_command(self):
        """ Wait for the command to be sent """
        while self.command_to_send:
            self.rate.sleep()
    
    def build_vehicle_command(self, **args):
        """ Construct a vehicle command
        (https://docs.px4.io/main/en/msg_docs/vehicle_command.html#vehicle-command-uorb-message)
        """
        self.vehicle_command = VehicleCommand()
        self.vehicle_command.timestamp = self.timesync_.timestamp
        for m_attr, value in args.items():
            setattr(self.vehicle_command, m_attr, value)
        # TODO: Does this even matter? Is it the right way to do it?
        self.vehicle_command.target_system = 1
        self.vehicle_command.target_component = 1
        self.vehicle_command.source_system = 1
        self.vehicle_command.source_component = 1
        self.vehicle_command.from_external = True
    
    def arm(self):
        """ Arm the motors and send a command via vehicle_command
        """
        self.get_logger().info("Arming the motors...")
        self.vehicle_command = VehicleCommand()
        self.build_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0)
        def arm_callback_and_msg():
            done = self.vehicle_control_mode.flag_armed
            if done:
                self.get_logger().info("Motors armed")
            return done
        self.command_succeed = arm_callback_and_msg
        self.command_to_send = True


    def disarm(self):
        """
        Disarm the motors and send a command via vehicle_command
        """
        self.get_logger().info("Disarming the motors...")
        self.stop_offboard_mode = True
        self.vehicle_command = VehicleCommand()
        self.build_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0)
        def disarm_callback_and_msg():
            done = not self.vehicle_control_mode.flag_armed
            if done:
                self.get_logger().info("Motors disarmed")
            return done
        self.command_succeed = disarm_callback_and_msg
        self.command_to_send = True
    
    def offboard(self):
        """
        Switch to offboard mode and send a command via vehicle_command
        """
        self.stop_offboard_mode = False
        self.offboard_msg(position=True)
        self.setpoint_msg(  x=self.vehicle_odometry.position[0], 
                            y=self.vehicle_odometry.position[1], 
                            z=self.vehicle_odometry.position[2], 
                            yaw=helper.quaternion_get_yaw(self.vehicle_odometry.q))

        self.get_logger().info("Switching to offboard mode...")
        self.build_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0)
        def offboard_callback_and_msg():
            done = self.vehicle_control_mode.flag_control_offboard_enabled
            if done:
                self.get_logger().info("Offboard mode enabled")
            return done
        self.command_succeed = offboard_callback_and_msg
        self.command_to_send = True


    def takeoff(self, z=1.0, yaw=None):
        """ Takeoff to a certain altitude
        Args:
            z: Altitude to takeoff to
            yaw: Yaw angle
        """
        # Arm the motors first if needed and same for offboard mode
        self.arm_and_offboard_if_needed()

        # Convert odemetry quaternion to yaw
        curr_yaw = helper.quaternion_get_yaw(self.vehicle_odometry.q)
        yaw_sp = helper.enu_euler_to_ned_euler(0,0,yaw)[2] if yaw is not None else curr_yaw
        yaw_sp_enu= helper.ned_euler_to_enu_euler(0,0,yaw_sp)[2]

        # Log warn takeoff and z position and yaw value
        self.get_logger().warn("Takeoff to {}m, yaw={}".format(z, yaw_sp_enu))

        # ztarget is the target altitude in meters
        ztarget = helper.enu_to_ned_z(z)
        self.setpoint_msg(
            x=self.vehicle_odometry.position[0],
            y=self.vehicle_odometry.position[1],
            z=ztarget,
            yaw= yaw_sp)
        
        def takeoff_callback_and_msg():
            done = abs(self.vehicle_odometry.position[2] - ztarget) <= 0.1
            if done:
                self.get_logger().info("Takeoff finished")
            return done

        self.action_completed = takeoff_callback_and_msg
    
    def land(self):
        """
        Land the vehicle via build_vehicle_command
        """
        self.get_logger().warn("Landing...")
        self.stop_offboard_mode = True
        self.offboard_msg()
        self.build_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_NAV_LAND)

        def land_callback_and_msg():
            done = -self.vehicle_odometry.position[2] < 0.1
            if done:
                self.get_logger().info("Landed")
            return done
        self.command_succeed = land_callback_and_msg
        self.command_to_send = True
    
    def pos(self, x=None, y=None, z=None, yaw=None):
        """ Move to a certain position (In ENU frame)
            Args:
                x: X position
                y: Y position
                z: Z position
                yaw: Yaw angle
        """
        # Arm the motors first if needed and same for offboard mode
        self.arm_and_offboard_if_needed()
        # Convert odemetry quaternion to yaw
        curr_yaw = helper.quaternion_get_yaw(self.vehicle_odometry.q)
        curr_yaw = helper.ned_euler_to_enu_euler(0,0,curr_yaw)[2]
        yaw_sp =  yaw if yaw is not None else curr_yaw
        # Convert the current position from NED to ENU
        curr_x, curr_y, curr_z = helper.ned_to_enu_position(
                                    [self.vehicle_odometry.position[0],
                                        self.vehicle_odometry.position[1],
                                        self.vehicle_odometry.position[2]
                                    ])

        # Assign x,y,z values if not None else assign current position
        x_sp = x if x is not None else curr_x
        y_sp = y if y is not None else curr_y
        z_sp = z if z is not None else curr_z

        self.get_logger().warn("Position to x={}, y={}, z={}, yaw={}".format(x_sp, y_sp, z_sp, yaw_sp))

        # COnvert the position from ENU to NED and assign to *target
        xtarget, ytarget, ztarget = helper.enu_to_ned_position([x_sp, y_sp, z_sp])
        yaw_sp = helper.enu_euler_to_ned_euler(0,0,yaw_sp)[2]

        # Set the setpoint message
        self.setpoint_msg(
            x=xtarget,
            y=ytarget,
            z=ztarget,
            yaw=yaw_sp)
        # Set the offboard message
        self.offboard_msg(position=True)

        def pos_callback_and_msg():
            done = np.linalg.norm(np.array([xtarget, ytarget, ztarget]) - np.array(self.vehicle_odometry.position)) <= 0.1
            if done:
                self.get_logger().info("Position reached")
            return done
        self.action_completed = pos_callback_and_msg
    
    def relpos(self, dx=0, dy=0, dz=0, dyaw=0):
        """
        Move to a certain position relative to the current position (In ENU frame)
        """
        # Convert odemetry quaternion to yaw
        curr_yaw = helper.quaternion_get_yaw(self.vehicle_odometry.q)
        yaw_sp = dyaw +  helper.ned_euler_to_enu_euler(0,0,curr_yaw)[2]
        # Convert the current position from NED to ENU
        curr_x, curr_y, curr_z = helper.ned_to_enu_position(self.vehicle_odometry.position)
        # Assign x,y,z values if not None else assign current position
        x_sp = curr_x + dx
        y_sp = curr_y + dy
        z_sp = curr_z + dz
        self.pos(x_sp, y_sp, z_sp, yaw_sp)


    def global_publisher(self):
        
        # Check if a command has to be sent  and send it
        if self.command_to_send:
            self.vehicle_command_pub.publish(self.vehicle_command)
        
        # Check if the command has been sent or need to be sent again
        if self.command_to_send and self.command_succeed():
            self.command_to_send = False
            self.command_succeed = lambda : False

        if self.action_completed():
            self.action_completed = lambda : False
        
        if self.stop_offboard_mode:
            return

        # TODO Later on I need to not publish it when in MPC mode
        # Publish offboard_control_mode
        self.offboard_control_mode.timestamp = self.timesync_.timestamp
        self.offboard_control_mode_pub.publish(self.offboard_control_mode)

        # Publish the setpoint position
        self.setpoint.timestamp = self.timesync_.timestamp
        self.setpoint_pub.publish(self.setpoint)
    

    def offboard_msg(self, **args):
        """ 
            Set up an offboard message based on args and store it
        """
        self.offboard_control_mode = OffboardControlMode()
        for m_attr, value in args.items():
            setattr(self.offboard_control_mode, m_attr, value)
        self.offboard_control_mode.timestamp = self.timesync_.timestamp
    
    def setpoint_msg(self, **args):
        """ 
            Set up a position setpoint message based on args and store it
        """
        self.setpoint = TrajectorySetpoint()
        # Fill the setpoint position with x,y,z from args
        self.setpoint.position = [float(args.get(v, np.nan)) for v in ['x', 'y', 'z']]
        # Fill the setpoint velocity with vx,vy,vz from args
        self.setpoint.velocity = [float(args.get(v, np.nan)) for v in ['vx', 'vy', 'vz']]
        # Fill the setpoint acceleration with ax,ay,az from args
        self.setpoint.acceleration = [float(args.get(v, np.nan)) for v in ['ax', 'ay', 'az']]
        # Fill the setpoint yaw with yaw from args
        self.setpoint.yaw = float(args.get('yaw', np.nan))
        # Fill the setpoint yaw speed with yaw_speed from args
        self.setpoint.yawspeed = float(args.get('yawspeed', np.nan))
        self.setpoint.timestamp = self.timesync_.timestamp
    
    def arm_and_offboard_if_needed(self):
        """ 
            Arm the motors and switch to offboard mode if needed
        """
        if not self.vehicle_control_mode.flag_armed:
            self.arm()
            self.wait_for_command()
        
        if not self.vehicle_control_mode.flag_control_offboard_enabled:
            self.offboard()
            self.wait_for_command()

    
    def control_mode_callback(self, msg):
        """ Callback for the vehicle control mode topic """
        # Print the control mode variables that changed
        if msg.flag_armed != self.vehicle_control_mode.flag_armed:
            self.get_logger().warn("Armed: {}".format(msg.flag_armed))
        if msg.flag_control_offboard_enabled != self.vehicle_control_mode.flag_control_offboard_enabled:
            self.get_logger().warn("Offboard control enabled: {}".format(msg.flag_control_offboard_enabled))
        if msg.flag_control_altitude_enabled != self.vehicle_control_mode.flag_control_altitude_enabled:
            self.get_logger().warn("Altitude control enabled: {}".format(msg.flag_control_altitude_enabled))
        if msg.flag_control_velocity_enabled != self.vehicle_control_mode.flag_control_velocity_enabled:
            self.get_logger().warn("Velocity control enabled: {}".format(msg.flag_control_velocity_enabled))
        if msg.flag_control_position_enabled != self.vehicle_control_mode.flag_control_position_enabled:
            self.get_logger().warn("Position control enabled: {}".format(msg.flag_control_position_enabled))
        if msg.flag_control_climb_rate_enabled != self.vehicle_control_mode.flag_control_climb_rate_enabled:
            self.get_logger().warn("Climb rate control enabled: {}".format(msg.flag_control_climb_rate_enabled))
        if msg.flag_control_termination_enabled != self.vehicle_control_mode.flag_control_termination_enabled:
            self.get_logger().warn("Termination control enabled: {}".format(msg.flag_control_termination_enabled))
        if msg.flag_control_manual_enabled != self.vehicle_control_mode.flag_control_manual_enabled:
            self.get_logger().warn("Manual control enabled: {}".format(msg.flag_control_manual_enabled))
        if msg.flag_control_auto_enabled != self.vehicle_control_mode.flag_control_auto_enabled:
            self.get_logger().warn("Auto control enabled: {}".format(msg.flag_control_auto_enabled))
        if msg.flag_control_rates_enabled != self.vehicle_control_mode.flag_control_rates_enabled:
            self.get_logger().warn("Rates control enabled: {}".format(msg.flag_control_rates_enabled))
        if msg.flag_control_attitude_enabled != self.vehicle_control_mode.flag_control_attitude_enabled:
            self.get_logger().warn("Attitude control enabled: {}".format(msg.flag_control_attitude_enabled))
        if msg.flag_control_acceleration_enabled != self.vehicle_control_mode.flag_control_acceleration_enabled:
            self.get_logger().warn("Acceleration control enabled: {}".format(msg.flag_control_acceleration_enabled))
        
        self.vehicle_control_mode = msg
    
    def timesync_callback(self, msg):
        self.timesync_ = msg
        # self.get_logger().info("Timesync timestamp: {}".format(msg.timestamp))
    
    def odometry_callback(self, msg):
        self.vehicle_odometry = msg
        # self.get_logger().info("Odometry timestamp: {}".format(msg.timestamp))

def main(args=None):
    rclpy.init(args=args)
    basic_control = BasicControl()
    # Parrallel thread to handle user input
    thread = threading.Thread(target=handle_user_input, args=(basic_control,))
    thread.start()
    rclpy.spin(basic_control)
    # Stop the thread
    thread.join()
    basic_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()