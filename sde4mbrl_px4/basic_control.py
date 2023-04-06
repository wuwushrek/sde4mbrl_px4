#!/usr/bin/env python3

import rospy
import numpy as np

# Import the messages we're interested in
from mavros_msgs.msg import State, ParamValue, DebugValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.srv import CommandBool, SetMode, ParamSet
from sensor_msgs.msg import BatteryState

from sde4mbrl_px4.srv import LoadTrajAndParams, LoadTrajAndParamsRequest
from sde4mbrl_px4.srv import FollowTraj, FollowTrajRequest

# Contain the command line interface
from input_command import handle_user_input

import threading
import tf

# TODO: Some system status -> Not sure if this is all
SYS_STATUS = {
    0 : "Uninitialized",
    1 : "Booting",
    2 : "Calibrating",
    3 : "Standby",
    4 : "Active",
    5 : "Critical",
    6 : "Emergency",
    7 : "Poweroff",
    8 : "Flight Termination",
}

MPC_STATUS = {
    -1 : "MPC OFF | NOT INITIALIZED",
     0 : "MPC OFF | NOT INITIALIZED",
     1 : "MPC ON | TEST",
     2 : "MPC OFF -> MPC timeout [Motor msg] delay > 20ms",
     3 : "MPC OFF -> FCU time >= MPC horizon -> MPC too slow",
     4 : "MPC OFF -> FCU time < MPC -> Shouldn't happen"
}

# Create a logger class with info, debug, warn, error, and fatal methods
class Logger:
    def __init__(self, name=""):
        self.name = name

    def info(self, msg):
        rospy.loginfo(self.name + ": " + msg)

    def debug(self, msg):
        rospy.logdebug(self.name + ": " + msg)

    def warn(self, msg):
        rospy.logwarn(self.name + ": " + msg)

    def error(self, msg):
        rospy.logerr(self.name + ": " + msg)

    def fatal(self, msg):
        rospy.logfatal(self.name + ": " + msg)
    

class BasicControl:
    """ A basic control class that can be used to control the drone """
    def __init__(self):
        self.logger = Logger("BasicControl")
        # Log out that the class has been initialized
        self.get_logger().info("BasicControl node has been started")

        ######### Internal variables of the node #########
        # [TODO] Parameterize these variables
        wait_freq = 5 # Hz, frequency of waiting for the command
        wait_time = 5. # seconds
        #  Number of iterations to wait for the command based on the wait_freq
        self.wait_iter = int(wait_time * wait_freq)
        # Rate for waiting for the command
        self.rate = rospy.Rate(wait_freq) # Parameterize later
        self.last_mpc_state = -1

        # Is there a command/service to send?
        self.command_to_send = False

        # Has the command been sent?
        self.command_succeed = lambda : False
        self.command_function = lambda : None

        # Has an action been completed?
        self.action_completed = lambda : False

        # Stop offboard mode when needed
        self.stop_offboard_mode = True

        # Check if a controller is on or not -> Geometric or MPC or other
        self.ctrl_on = False

        # Check if the vehicle must satify some BOX constraints when operating
        self.security_check = False
        self.not_safe = False

        ######### Add services for arming and set mode #########
        # First we need to wait for the services to be available
        self.get_logger().warn("Waiting for services to be available")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.get_logger().warn("Services arming and set mode are detected")

        # Now we can create the service proxy
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.get_logger().warn("Arming and set mode services are ready")

        # Service for setting vehicle parameters
        self.set_param_client = rospy.ServiceProxy("/mavros/param/set", ParamSet)

        # Service for loading trajectory and parameters
        self.load_traj_and_params_client = rospy.ServiceProxy("set_trajectory_and_params", LoadTrajAndParams)

        # Service for starting the trajectory controller
        self.start_traj_controller_client = rospy.ServiceProxy("start_trajectory", FollowTraj)

        ######### Store subscribed/published messages #########
        self.state = State()
        self.odom = Odometry()
        # self.imu = Imu()
        self.battery = BatteryState()
        self.setpoint = PoseStamped()

        ######### Create subscribers for state, odometry, imu, battery #########
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
        # self.battery_sub = rospy.Subscriber("/mavros/battery", BatteryState, self.battery_callback)
        self.mpc_debug_sub = rospy.Subscriber("/mavros/debug_value/named_value_float", DebugValue, self.mpc_debug_callback)
        self.get_logger().warn("Subscribers are ready")

        ######### Create publisher for setpoint_position/local #########
        self.setpoint_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.get_logger().warn("Publisher [setpoint_position] is ready")

        ######### A timer for publishing the setpoint and main loop #########
        offboard_freq = 50. # In Hz -> [TODO] Parameterize later
        self.setpoint_timer = rospy.Timer(rospy.Duration(1./offboard_freq), self.offboard_loop)

        ## Set COM_RCL_EXCEPT to 4 for disabling failsafe without RC
        # [TODO] Check if this is needed
        for _ in range(5):
            self.set_param_client("COM_RCL_EXCEPT", ParamValue(integer=4))
            rospy.sleep(0.1)

    def wait_for_command(self):
        """ Wait for the command to be sent """
        for _ in range(self.wait_iter):
            if not self.command_to_send:
                return True
            self.rate.sleep()
        return False
    
    def arm(self):
        """ Arm the drone """
        self.get_logger().warn("Arming the motors...")
        self.command_function = lambda : self.arming_client(True)
        def arm_callback_and_msg():
            done = self.state.armed
            if done:
                self.get_logger().warn("Motors armed")
            return done
        self.command_succeed = arm_callback_and_msg
        self.command_to_send = True
    
    def arm_nooffboard(self):
        """ Arm the drone and disable offboard mode in the process"""
        self.stop_offboard_mode = True
        self.get_logger().warn("Arming the motors...")
        self.command_function = lambda : self.arming_client(True)
        def arm_callback_and_msg():
            done = self.state.armed
            if done:
                self.get_logger().warn("Motors armed")
            return done
        self.command_succeed = arm_callback_and_msg
        self.command_to_send = True
    
    def disarm(self):
        """ Disarm the drone  and disarm offboard mode in the process"""
        self.get_logger().warn("Disarming the motors...")
        self.stop_offboard_mode = True
        self.command_function = lambda : self.arming_client(False)
        def disarm_callback_and_msg():
            done = not self.state.armed
            if done:
                self.get_logger().warn("Motors disarmed")
            return done
        self.command_succeed = disarm_callback_and_msg
        self.command_to_send = True
    
    def offboard(self):
        """ Set the drone to offboard mode """
        self.stop_offboard_mode = False
        # Initialize the setpoint at the current position
        self.setpoint_msg()
        self.get_logger().warn("Setting offboard mode...")
        if self.state.mode == "OFFBOARD":
            self.get_logger().warn("Already in offboard mode")
            return
        self.command_function = lambda : self.set_mode_client(custom_mode="OFFBOARD")
        def offboard_callback_and_msg():
            done = self.state.mode == "OFFBOARD"
            if done:
                self.get_logger().warn("Offboard mode set")
            return done
        self.command_succeed = offboard_callback_and_msg
        self.command_to_send = True
    
    def takeoff(self, z=1.0, yaw=None, use_ctrl=False):
        """ Takeoff to a certain altitude
        Args:
            z: Altitude to takeoff to
            yaw: Yaw angle
        """
        # Arm the motors first if needed and same for offboard mode
        self.arm_and_offboard_if_needed()

        # Convert odometry quaternion to yaw via tf
        q = self.odom.pose.pose.orientation
        roll, pitch, curr_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_sp = curr_yaw if yaw is None else yaw
        q_target = tf.transformations.quaternion_from_euler(roll, pitch, yaw_sp)

        # Log warn takeoff and z position and yaw value
        self.get_logger().warn("Takeoff to {}m, yaw={}".format(z, yaw_sp))

        # Set the setpoint based on current x, y, z, yaw
        self.setpoint_msg(x=self.odom.pose.pose.position.x,
                            y=self.odom.pose.pose.position.y,
                            z=z,
                            qx=q_target[0],
                            qy=q_target[1],
                            qz=q_target[2],
                            qw=q_target[3])

        # Callback function to check if takeoff is completed
        def takeoff_callback_and_msg():
            done = abs(self.odom.pose.pose.position.z - z) <= 0.1 # [TODO] Parameterize
            if done:
                self.get_logger().warn("Takeoff completed")
            return done
        self.action_completed = takeoff_callback_and_msg
        
        # Set the setpoint to the controller instead of the PID controller onboard
        # And exit when that is done
        if use_ctrl:
            self.ctrl_cmd_posestamped()
            return
        
        # If the controller is on, send an off request to the node
        if self.ctrl_on:
            self.controller_off()
    
    def ctrl_takeoff(self, z=1.0, yaw=None):
        return self.takeoff(z, yaw, use_ctrl=True)
    
    def pos(self, x=None, y=None, z=None, yaw=None, use_ctrl=False):
        """Move to a certain position 
            Args:
                x: X position
                y: Y position
                z: Z position
                yaw: Yaw angle
        """
        # Arm the motors first if needed and same for offboard mode
        self.arm_and_offboard_if_needed()

        # Convert odometry quaternion to yaw via tf
        q = self.odom.pose.pose.orientation
        roll, pitch, curr_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_sp = curr_yaw if yaw is None else yaw
        q_target = tf.transformations.quaternion_from_euler(roll, pitch, yaw_sp)

        # Log warn takeoff and z position and yaw value
        self.get_logger().warn("Move to {}, {}, {}, yaw={}".format(x, y, z, yaw_sp))

        # Assign x,y,z values if not None else assign current position
        x_sp = x if x is not None else self.odom.pose.pose.position.x
        y_sp = y if y is not None else self.odom.pose.pose.position.y
        z_sp = z if z is not None else self.odom.pose.pose.position.z

        self.get_logger().warn("Position to x={}, y={}, z={}, yaw={}".format(x_sp, y_sp, z_sp, yaw_sp))

        # Set the setpoint based on current x, y, z, yaw
        self.setpoint_msg(x=x_sp,
                        y=y_sp,
                        z=z_sp,
                        qx=q_target[0],
                        qy=q_target[1],
                        qz=q_target[2],
                        qw=q_target[3])
        
        # Callback function to check if pos is completed
        def pos_callback_and_msg():
            # norm between current position and setpoint x,y,z
            done = np.linalg.norm([self.odom.pose.pose.position.x - x_sp,
                                    self.odom.pose.pose.position.y - y_sp,
                                    self.odom.pose.pose.position.z - z_sp]) <= 0.1
            if done:
                self.get_logger().warn("Position reached")
            return done
        self.action_completed = pos_callback_and_msg
        
        if use_ctrl:
            self.ctrl_cmd_posestamped()
            return
        
        if self.ctrl_on:
            self.controller_off()
    
    def ctrl_pos(self, x=None, y=None, z=None, yaw=None):
        return self.pos(x, y, z, yaw, use_ctrl=True)
    
    def relpos(self, dx=0, dy=0, dz=0, dyaw=0, use_ctrl=False):
        """
        Move to a certain position relative to the current position (In ENU frame)
        """
        # Arm the motors first if needed and same for offboard mode
        self.arm_and_offboard_if_needed()

        # Convert odometry quaternion to yaw via tf
        q = self.odom.pose.pose.orientation
        _, _, curr_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_sp = curr_yaw + dyaw
        x_sp = self.odom.pose.pose.position.x + dx
        y_sp = self.odom.pose.pose.position.y + dy
        z_sp = self.odom.pose.pose.position.z + dz
        self.pos(x_sp, y_sp, z_sp, yaw_sp, use_ctrl)
    
    def ctrl_relpos(self, dx=0, dy=0, dz=0, dyaw=0):
        return self.relpos(dx, dy, dz, dyaw, use_ctrl=True)
    
    def land(self):
        """ Land the drone via set_mode_client"""
        self.stop_offboard_mode = True
        if self.ctrl_on:
            self.controller_off()
        self.get_logger().warn("Landing...")
        self.command_function = lambda : self.set_mode_client(custom_mode="AUTO.LAND")
        def land_callback_and_msg():
            done = abs(self.odom.pose.pose.position.z) <= 0.1
            if done:
                self.get_logger().warn("Landing completed")
            return done
        self.command_succeed = land_callback_and_msg
        self.command_to_send = True
    

    def offboard_loop(self, _):
        """ The main loop for publishing the setpoint """
        # Check if a command has to be sent  and send it
        if self.command_to_send:
            self.command_function()
        
        # Check if the command has been sent or need to be sent again
        if self.command_to_send and self.command_succeed():
            self.command_to_send = False
            self.command_function = lambda : None
            self.command_succeed = lambda : False
        
        # Check if the action has been completed
        if self.action_completed():
            self.action_completed = lambda : False
        
        # Check the security box only in offboard mode
        if self.security_check and self.state.mode == "OFFBOARD":
            not_safe = self.check_unsafe_box()
            if not_safe and not self.not_safe: # First time we hit not safe
                # This already go offboard while staying still
                self.controller_off()
                self.not_safe = True
                self.get_logger().warn("Not safe, staying still")
            elif not_safe and self.not_safe:
                # Get back at the center of the safe box
                self.pos(self.center_point[0], self.center_point[1], self.center_point[2])
                self.not_safe = False
            else:
                self.not_safe = False
        
        if self.stop_offboard_mode:
            return
        
        # Update the time stamp and seq of the pose
        self.setpoint.header.stamp = rospy.Time.now()
        next_seq = self.setpoint.header.seq + 1
        self.setpoint.header.seq = next_seq
        # Publish the setpoint
        self.setpoint_pub.publish(self.setpoint)
    
    def set_box(self, x=0.2, y=0.2, z=0.2):
        # Get current position
        x0 = self.odom.pose.pose.position.x
        y0 = self.odom.pose.pose.position.y
        z0 = self.odom.pose.pose.position.z
        # Set the safe box
        self.safe_box = np.array([x0-x, x0+x, y0-y, y0+y, z0-z, z0+z])
        self.center_point = np.array([x0, y0, z0])
        self.security_check = True
        self.not_safe = False
    
    def rm_box(self):
        self.security_check = False
        self.safe_box = None
        self.center_point = None
        self.not_safe = False
        
    def check_unsafe_box(self):
        # Check if the drone is within the safe box
        drone_x = self.odom.pose.pose.position.x
        drone_y = self.odom.pose.pose.position.y
        drone_z = self.odom.pose.pose.position.z
        if drone_x < self.safe_box[0] or drone_x > self.safe_box[1] or \
            drone_y < self.safe_box[2] or drone_y > self.safe_box[3] or \
            drone_z < self.safe_box[4] or drone_z > self.safe_box[5]:
            return True
        return False

    def state_callback(self, msg):
        """ Callback for state """
        # Print out the state that changed since the last callback
        if msg.armed != self.state.armed:
            self.get_logger().warn("Armed: " + str(msg.armed))
        if msg.connected != self.state.connected:
            self.get_logger().warn("Connected: " + str(msg.connected))
        if msg.mode != self.state.mode:
            self.get_logger().warn("Mode: " + str(msg.mode))
        if msg.system_status != self.state.system_status:
            # A readable system status
            self.get_logger().warn("System status: " + SYS_STATUS[msg.system_status])
        self.state = msg
    
    def mpc_debug_callback(self, msg):
        """ Callback for mpc debug """
        # First check if the key of the message match mpc_state
        if msg.name != "mpc":
            return
        mpc_state = int(msg.value_float)
        
        if mpc_state != self.last_mpc_state:
            self.get_logger().warn("MPC state: " + MPC_STATUS[mpc_state])
            # Check if we go from mpc_on to off
            if self.last_mpc_state == 1 and mpc_state <= 0:
                # Switch to position control
                self.get_logger().warn("Switching to position control")
                self.pos()

        self.last_mpc_state = mpc_state


    def odom_callback(self, msg):
        """ Callback for odometry """
        self.odom = msg
    
    def get_logger(self):
        """ Returns the logger for this class """
        return self.logger
    
    def setpoint_msg(self, **args):
        """ 
            Set up a position setpoint message based on args and store it
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.odom.header.frame_id
        msg.header.seq = self.setpoint.header.seq
        msg.pose.position.x = args.get("x", self.odom.pose.pose.position.x)
        msg.pose.position.y = args.get("y", self.odom.pose.pose.position.y)
        msg.pose.position.z = args.get("z", self.odom.pose.pose.position.z)
        msg.pose.orientation.x = args.get("qx", self.odom.pose.pose.orientation.x)
        msg.pose.orientation.y = args.get("qy", self.odom.pose.pose.orientation.y)
        msg.pose.orientation.z = args.get("qz", self.odom.pose.pose.orientation.z)
        msg.pose.orientation.w = args.get("qw", self.odom.pose.pose.orientation.w)
        self.setpoint = msg
    
    def battery_callback(self, msg):
        """ Callback for battery """
        self.battery = msg
        # TODO: Add a parameter for the battery warning level
        # Warn out the battery level if low
        if self.battery.percentage < 0.4:
            self.get_logger().warn("Battery level is low: " + str(self.battery.percentage))
    
    def arm_and_offboard_if_needed(self):
        """ 
            Arm the motors and switch to offboard mode if needed
        """
        if not self.state.mode == "OFFBOARD":
            self.offboard()
            self.wait_for_command()
            
        if not self.state.armed:
            self.arm()
            self.wait_for_command()

    
    def controller_init(self, config_name=""):
        """
            controller_init test_traj_gen.csv gm_iris.yaml
            Initialize the controller. The controller is advertised as a service.
            The name of the service is set_trajectory_and_params
            config_name: The path to the controller parameters file
        """
        try:
            # Wait for the set_trajectory_and_params
            rospy.wait_for_service("set_trajectory_and_params", timeout=0.5) # [TODO] Make this a parameter

            # Create the request
            req = LoadTrajAndParamsRequest()
            req.controller_param_yaml = config_name

            # Call the service
            res = self.load_traj_and_params_client(req)
            if not res.success:
                self.get_logger().warn("Failed to load the trajectory and the parameters")
                return
            self.get_logger().info("Loaded the trajectory and the parameters")
        except rospy.ServiceException as e:
            self.get_logger().error("Failed to call set_trajectory_and_params: " + str(e))
    
    def controller_set_mode(self, mode, wmotors=110):
        """
            Set the controller mode
            mode: The mode to set
        """
        try:
            # Wait for the set_mode
            try:
                rospy.wait_for_service("start_trajectory", timeout=0.1) # Only wait for 0.1s
            except rospy.ROSException as e:
                self.get_logger().warn("Failed to wait for start_trajectory: " + str(e))
                self.ctrl_on = False
                return

            # Create the request
            req = FollowTrajRequest()
            req.state_controller = mode
            req.weight_motors = wmotors

            # Modify the quaternion of the target to have 0 roll, 0 pitch, and the yaw of the current position
            q = self.setpoint.pose.orientation
            _, _, curr_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            q_target = tf.transformations.quaternion_from_euler(0, 0, curr_yaw)
            self.setpoint.pose.orientation.x = q_target[0]
            self.setpoint.pose.orientation.y = q_target[1]
            self.setpoint.pose.orientation.z = q_target[2]
            self.setpoint.pose.orientation.w = q_target[3]
            req.target_pose = self.setpoint

            # Call the service
            res = self.start_traj_controller_client(req)

            if not res.success:
                self.get_logger().warn("Failed to set controller mode to " + str(mode))
                return
            
            if wmotors >=0 and wmotors <= 100:
                self.get_logger().warn("Setting MPC weight motors to: " + str(wmotors))
                return
            
            if mode != req.CTRL_INACTIVE and mode != req.CTRL_TEST:
                self.get_logger().warn("Controller mode set to: " + 'CTRL_TRAJ_ACTIVE' if mode == req.CTRL_TRAJ_ACTIVE else 'CTRL_POSE_ACTIVE' if mode == req.CTRL_POSE_ACTIVE else "CTRL_TRAJ_IDLE")
                self.ctrl_on = True
            else:
                self.get_logger().warn("Controller mode set to: " + 'CTRL_INACTIVE' if mode == req.CTRL_INACTIVE else 'CTRL_TEST')
                self.ctrl_on = False

        except rospy.ServiceException as e:
            self.get_logger().warn("Failed to call set_mode: " + str(e))
            # Add exception with timeout of the service

    
    def controller_off(self, nooffboard=False):
        """
            Turn off the controller
        """
        # Switch to offboard mode -> Essentially just send the current position as a setpoint
        # if not nooffboard:
        self.offboard()
        self.controller_set_mode(FollowTrajRequest.CTRL_INACTIVE)
    
    def controller_on(self):
        """
            Turn on the controller
        """
        self.controller_set_mode(FollowTrajRequest.CTRL_TRAJ_ACTIVE)
        self.stop_offboard_mode = True
    
    def controller_test(self):
        """
            Turn on the controller
        """
        # Make setpoint the current position
        self.setpoint_msg()
        self.controller_set_mode(FollowTrajRequest.CTRL_TEST)
        
    
    def controller_idle(self):
        """
            Reset the controller
        """
        self.controller_set_mode(FollowTrajRequest.CTRL_TRAJ_IDLE)
        self.stop_offboard_mode = True
    
    def weight_motors(self, wmotors):
        """
            Set the weight of the motors
        """
        # Check if the input is between 0 and 100 if not return
        if wmotors < 0 or wmotors > 100:
            self.get_logger().warn("Weight motors must be between 0 and 100")
            return
        self.controller_set_mode(FollowTrajRequest.CTRL_TEST, wmotors)
    
    def ctrl_cmd_posestamped(self):
        """
            Callback for the controller command
        """
        self.controller_set_mode(FollowTrajRequest.CTRL_POSE_ACTIVE)
        self.stop_offboard_mode = True

def main():
    """Create node, parse arguments if present, and spin"""
    rospy.init_node("BasicControl")

    basicControl = BasicControl()

    # Parrallel thred to handle user input
    input_thread = threading.Thread(target=handle_user_input, args=(basicControl,))
    input_thread.start()

    # Spin the node
    rospy.spin()

    # Stop the input thread
    input_thread.join()
    
    # Shutdown
    rospy.signal_shutdown("Basic offboard control node is terminated")

if __name__ == '__main__':
    main()