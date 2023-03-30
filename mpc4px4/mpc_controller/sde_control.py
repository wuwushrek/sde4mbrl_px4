#!/usr/bin/env python

import rospy
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp

import numpy as np

from sde4mbrlExamples.rotor_uav.sde_mpc_design import load_mpc_from_cfgfile
from sde4mbrlExamples.rotor_uav.utils import enu2ned

from mpc4px4.srv import FollowTraj, FollowTrajRequest, FollowTrajResponse
from mpc4px4.srv import LoadTrajAndParams, LoadTrajAndParamsResponse
from mpc4px4.msg import OptMPCState


from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import MPCMotorsCMD, MPCFullState

import pymavlink.mavutil as mavutil

import time

import threading

import multiprocessing
from multiprocessing import shared_memory

import setproctitle

# # Profiler
# with jax.profiler.trace("trace_mpc_t", create_perfetto_link=True):
#     _,self._uopt, _, self.opt_state = self._apg_mpc(self._dummy_x, init_rng, self.opt_state, self._dummy_xref)
#     self._uopt.block_until_ready()

# # Profiler
# with jax.profiler.trace("trace_mpc", create_perfetto_link=True):
#     self.opt_state = self._reset_fn(self._dummy_x)
#     self.opt_state.num_steps.block_until_ready()

class SDEControlROS:
    """ A ROS node to control the quadcopter using my sde-based approach
    """
    def __init__(self,):
        """ Constructor """

        # Info that the controller started
        rospy.loginfo("Starting the SDE controller")

        # Constants used below
        # the control automata has 3 states: 'idle', 'pos', 'traj'. idle performs pistion control to reach the initial state of the trajectory
        self.control_state_dict = {'idle': 0, 'pos': 1, 'traj': 2, 'none' : -1}
        self.name_control_state_dict = {v : k for k, v in self.control_state_dict.items()}
        self.last_traj_time = 0.0
        self.dt_state_callback = 0.0
        self.last_time_state_info = None
        self._target_x = self.dummy_state()

        # Trajectory and position control parameters
        self._run_trajectory = False
        self._trajec_time = -1.0
        self._pos_control = False
        self._test_mode = False
        self.mpc_on = MPCMotorsCMD.MPC_OFF
        self.reset_done = False
        self._index = 0

        # Store the parameters of this node
        self.init_node_params()

        # Load the mpc solver fromc config file
        self.load_mpc_models()

        # First define the shareable variables
        self.multi_process_shared_variables()

        # Start the mpc process after loading the mpc
        self.start_mpc_process()

        # [TODO] Create a mavlink connection -> Do this at last maybe
        self.init_mavlink_connection()

        # Publisher for the setpoint
        self._setpoint_pub = rospy.Publisher("mavros/desired_setpoint", PoseStamped, queue_size=10)

        self._opt_state_pub = rospy.Publisher("mavros/mpc_opt_state/state", OptMPCState, queue_size=10)

        # Service to set the trajectory
        self._set_traj_srv = rospy.Service("set_trajectory_and_params", LoadTrajAndParams, self.initialize_mpc_callback)

        # Service to start the trajectory
        self._start_traj_srv = rospy.Service("start_trajectory", FollowTraj, self.start_trajectory_callback)

        # Create a timer to publish on _opt_state_pub
        self._timer = rospy.Timer(rospy.Duration(self.mpc_report_dt), self.publish_opt_state)


    def init_node_params(self):
        """ Initialize the node parameters from launch file param
        """
        self.seed = rospy.get_param("~seed", 0)
        self.config_dir = rospy.get_param("~config_dir", "")
        self.traj_ctrl_dir = rospy.get_param("~traj_ctrl", "")
        self.sp_ctrl_dir = rospy.get_param("~sp_ctrl", "")
        self.mpc_report_dt = rospy.get_param("~mpc_report_dt", 0.2)
        self.mav_addr = rospy.get_param("~addr_mavlink_state_msg", "localhost:14998")
        # Pretty print the parameters
        rospy.loginfo("Node parameters:")
        rospy.loginfo("Seed: {}".format(self.seed))
        rospy.loginfo("Config directory: {}".format(self.config_dir))
        rospy.loginfo("Trajectory controller directory: {}".format(self.traj_ctrl_dir))
        rospy.loginfo("Setpoint controller directory: {}".format(self.sp_ctrl_dir))
        rospy.loginfo("MPC logging dt: {}".format(self.mpc_report_dt))
        rospy.loginfo("Mavlink address: {}".format(self.mav_addr))

    def init_mavlink_connection(self):
        """ Initialize the mavlink connection and start the thread to listen to the messages
        """
        rospy.logwarn('Initializing the mavlink connection')
        self.mav = mavutil.mavlink_connection('udpin:'+self.mav_addr)
        rospy.logwarn('Waiting for the first MPC message')
        _msg = self.mav.recv_match(blocking=True, timeout=1.0)
        if _msg is None:
            rospy.logwarn('No message received, make sure the mavlink router is running')
            self.last_time_state_info = _msg.time_usec
        else:
            self.last_time_state_info = None
            rospy.logwarn('First MPC State message received')
            # Log warn the message
            rospy.logwarn(_msg)

        # Start the thread to listen to the messages
        self.mpc_state_thread = threading.Thread(target=self.handle_mpc_state_msg)
        self.mpc_state_thread.start()

    def handle_mpc_state_msg(self):
        """ Listen on node.mav_addr to obtain the MPC_STATE message.
            Then, use the MPC_STATE to compute the next control input and immediately send it to the drone.
            Save the setpoint that is going to be sent over ros.
            This function is called in a separate thread at startup.
        """
        # TODO: Parametrize the timeout
        recv_time_out = 0.1
        # Loop to receive messages
        while not rospy.is_shutdown():
            # Get the state
            msg = self.mav.recv_match(blocking=True, timeout=recv_time_out)
            # If the message is not None, then call the callback
            # The call back essentially notify the mpc solver then extract the next sequences of control inputs from
            # the last solved mpc problem
            if msg is not None:
                self.mpc_state_callback(msg)

    def load_mpc_models(self):
        """ Load from given configuration file the MPC solvers
        """
        rospy.logwarn("################################################################")
        rospy.logwarn("Trajectory provided --> MPC will be used as a trajectory tracker")
        traj_ctrl_path = self.config_dir + "/" + self.traj_ctrl_dir
        self.state_from_traj, self.reset_traj_mpc, self.mpc_traj_solver, self.traj_uopt, \
                self.default_traj_opt_state, self.traj_cfg_dict = self.load_single_mpc(traj_ctrl_path)
        assert self.state_from_traj is not None, "The state_from_traj function should be not None, Have you provided a trajectory?"

        # Stire the dt time step
        # TODO: rEMOVE THESE AS THEY ARE USELESS NOW
        self._dt_usec = self.traj_cfg_dict['_time_steps'][0] * 1e6 # in usec
        self.tn = float(self.traj_cfg_dict['cost_params']['uref'][0])

        rospy.logwarn("################################################################")
        rospy.logwarn("Position controller provided --> MPC will be used as a position controller")
        pos_ctrl_path = self.config_dir + "/" + self.sp_ctrl_dir
        _state_from_pos, self.reset_pos_mpc, self.mpc_pos_solver, self.pos_uopt,\
                    self.default_pos_opt_state, self.pos_cfg_dict = self.load_single_mpc(pos_ctrl_path)
        assert _state_from_pos is None, "The state_from_pos function should be None, Have you provided a position controller?"


    def control_automata(self):
        """ Define the ways the control switch between trajectory and position controller
        """
        # Check about
        # dt is the time step since the last call of this function
        if self._pos_control:
            self._target_x = np.array([self._target_sp.pose.position.x, self._target_sp.pose.position.y, self._target_sp.pose.position.z,
                                      0., 0., 0.,
                                      self._target_sp.pose.orientation.w, self._target_sp.pose.orientation.x, self._target_sp.pose.orientation.y, self._target_sp.pose.orientation.z,
                                      0., 0., 0.],
                                    dtype=np.float32
                                )
            return self.control_state_dict['pos']

        if self._trajec_time < 0.0:
            # We are not running the trajectory
            return self.control_state_dict['none']

        # If we get there it means current trajec_time >= 0 but the trajectory is not running
        if not self._run_trajectory: # If we are not running the trajectory
            # Get the initial state
            self._trajec_time = 0
            self._target_x = np.array(self.state_from_traj(self._trajec_time), dtype=np.float32)
            return self.control_state_dict['idle']

        # We are running the trajectory
        _current_time = time.time()
        if self._trajec_time == 0: # If we are starting the trajectory
            # Start measuring the time for the trajectory
            self.last_traj_time = _current_time
            self._trajec_time = 0.0000001 # Small value so that next time we don't enter this if
        else:
            # Save the time elapsed since the beginning of the trajectory
            self._trajec_time = (_current_time - self.last_traj_time)
        return self.control_state_dict['traj']


    def mpc_state_callback(self, msg):
        """ Callback for the full state of the quadcopter.
            This is also where the main control computation is going to be done.
            So we compute control when we receive new position
                Args:
                    msg (MPCFullState): The full state of the quadcopter in ned coordinate
        """
        # Let's proceed to publish the control action
        self.curr_time = rospy.Time.now()

        # Get the current time
        _curr_time = time.time()

        # Time elapsed since the last call -> meaning last mpc state info
        self.dt_state_info = _curr_time - self.last_time_state_info if self.last_time_state_info is not None else 0.0
        # Update the last time received state info
        self.last_time_state_info = _curr_time

        # Transform the state into numpy array
        curr_state = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz, msg.qw, msg.qx, msg.qy, msg.qz, msg.wx, msg.wy, msg.wz], dtype=np.float32)

        # Get the time the sample was taken
        self.sample_time = msg.time_usec

        # Get the current control state
        self._control_state = self.control_automata()

        # Store information needed by the mpc process in the shared memory
        # Maybe Lock before modifying these information
        with self._curr_state_lock:
            # Store the current state
            self.curr_state_shr[:] = curr_state
            # Store the sample time for the current state
            self.info_mpc_pre_shr[self.key2index_pre['sample_time_prempc']] = msg.time_usec
            # Store the current control state
            self.info_mpc_pre_shr[self.key2index_pre['ctrl_state']] = self._control_state
            # Store the duration used in trajectory optimization
            self.info_mpc_pre_shr[self.key2index_pre['duration']] = self._trajec_time
            if self._control_state != self.control_state_dict['traj']:
                # Store the target setpoint
                self.target_setpoint_shr[:] = self._target_x


        # Send a signal to wake up the mpc process
        self._mpc_event.set()

        # # Wait for the mpc response
        # self.sync_event.wait()
        # self.sync_event.clear()

        # Proceed to Check from the shared memory the optimal control action and current time
        # Extract the frist index to start
        with self._u_opt_lock:
            # Extract the latest computed control
            _u_opt = np.array(self.u_opt_shr)
            # Extract the lates angular rate desired values
            _w_opt = np.array(self.w_opt_shr)
            # Extract the optimization info
            self._optimizer_info = np.array(self.opt_info_shr)

        # Now proceed to parse and publish the control action
        tsample_mpc = self._optimizer_info[self.key2index_info_mpc['sample_time_posmpc']]
        if tsample_mpc <= 0:
            self.dt_state_callback = time.time() - _curr_time
            # Ros log warn
            rospy.logwarn("The sample time for the mpc is not valid -> No MPC solution computed yet")
            return

        # Find the index of the control action to use
        _index = int((self.sample_time - tsample_mpc) / self._dt_usec)
        if _index >= _u_opt.shape[0]:
            # log ros error message
            rospy.logerr("The index of the control action is greater than the size of the control action array")
            # Pick the last control
            _index = _u_opt.shape[0] - 1

        # self._uopt = _u_opt[_index:(_index + MPCMotorsCMD.HORIZON_MPC)]
        self._uopt = _u_opt[_index, :]
        # Complete _uopt with zeros values if its dimension is only 4
        if self._uopt.shape[0] < 6:
            self._uopt = np.concatenate((self._uopt, np.zeros((6 - self._uopt.shape[0],))))
        # Obtain the angular rate desired
        self._wopt = _w_opt[_index, :]
        # Store the index for printing
        self._index = _index

        # if self._uopt.shape[0] < MPCMotorsCMD.HORIZON_MPC:
        #     # We duplicate the last control action and append it to match the horizon
        #     self._uopt = np.concatenate((self._uopt, np.tile(self._uopt[-1], (MPCMotorsCMD.HORIZON_MPC - self._uopt.shape[0], 1))), axis=0)

        # Check if there is a need to publish the control action
        if self._control_state == self.control_state_dict['none']:
            # No need to publish the control action
            self.dt_state_callback = time.time() - _curr_time
            return

        # Publish the control action
        self.pub_cmd_setpoint(self.curr_time, self.sample_time)

        # # # Publish the target setpoint
        # self.pub_reference_pose(_curr_ros_time)

        # COmpute the time elapsed in this function
        self.dt_state_callback = time.time() - _curr_time


    def mpc_process_fn(self):
        """ Main loop in the MPC process
        """
        # Set the process name
        setproctitle.setproctitle("mpc_process")

        # Rospy warn
        rospy.logwarn("Starting and Setting up the MPC process")

        # Initialize random key generator
        rng_ctrl = jax.random.PRNGKey(self.seed)

        # Split the rng key into 3 keys
        rng_ctrl, rng_traj, rng_pos = jax.random.split(rng_ctrl, 3)

        # Reset both the trajectory and position mpc
        x0 = self.dummy_state()
        opt_state_traj = self.reset_traj_mpc(x=self.dummy_state(), rng=rng_traj, xdes=x0)
        opt_state_pos = self.reset_pos_mpc(x=x0, rng=rng_pos, xdes=x0)

        # COmpute the control solution at least once
        self.mpc_traj_solver(x0, rng_traj, opt_state_traj, curr_t=0.0, xdes=x0)
        self.mpc_pos_solver(x0, rng_pos, opt_state_pos, curr_t=0.0, xdes=x0)

        # Warn that the MPC process is ready
        rospy.logwarn("MPC process is ready, looping")

        # A boolean to check if we need to wait for the event
        wait_event = True

        # THe current controller
        _curr_ctrl = None

        # _ctime = time.time()

        # Loop until the node is killed
        while True:
            # Wait for the event to be set with a timeout
            if wait_event:
                self._mpc_event.wait()
                self._mpc_event.clear()

            # Extract the useful information from the shared memory
            with self._curr_state_lock:
                # Store the current state
                curr_state = np.array(self.curr_state_shr)
                # Store the sample time for the current state
                sample_time = self.info_mpc_pre_shr[self.key2index_pre['sample_time_prempc']]
                # Store the current control state
                _control_state = self.info_mpc_pre_shr[self.key2index_pre['ctrl_state']]
                # Store the duration used in trajectory optimization
                _trajec_time = self.info_mpc_pre_shr[self.key2index_pre['duration']]
                if _control_state != self.control_state_dict['traj']:
                    # Store the target setpoint
                    _target_x = np.array(self.target_setpoint_shr)

            # Check if the control state is set to None
            _perf_time = time.time()
            if _curr_ctrl is None or (_curr_ctrl == 'none' and _control_state != self.control_state_dict['none']):
                # Reset both controllers
                opt_state_traj = self.reset_traj_mpc(x=curr_state, rng=rng_traj, xdes=curr_state)
                opt_state_pos = self.reset_pos_mpc(x=curr_state, rng=rng_pos, xdes=curr_state)

            if _control_state == self.control_state_dict['none']:
                _curr_ctrl = 'none'
                _uopt, opt_state_pos, rng_pos, _pos_evol = self.mpc_pos_solver(curr_state, rng_pos, opt_state_pos, curr_t=0.0, xdes=enu2ned(curr_state, np))
            elif _control_state == self.control_state_dict['traj']:
                _curr_ctrl = 'traj'
                _uopt, opt_state_traj, rng_traj, _pos_evol = self.mpc_traj_solver(curr_state, rng_traj, opt_state_traj, curr_t=_trajec_time, xdes=curr_state)
            elif _control_state == self.control_state_dict['pos']:
                _curr_ctrl = 'pos'
                _uopt, opt_state_pos, rng_pos, _pos_evol = self.mpc_pos_solver(curr_state, rng_pos, opt_state_pos, curr_t=0.0, xdes=_target_x)
            else:
                raise ValueError("Unknown control state: {}".format(_control_state))
            _uopt.block_until_ready()
            # # Let simulate some delay in the computation
            # time.sleep(0.1)

            _perf_time = time.time() - _perf_time

            # Do some processing before saving the control
            _uopt = np.array(_uopt)
            # TODO: Something more elegant here based on the Mier inside P4
            # Compute the thrust value -> This function assumes that all motors are operational and identical
            thrust_evol = np.sum(_uopt, axis=1) / _uopt.shape[1]
            _wopt = np.array([thrust_evol, _pos_evol[1:,10], _pos_evol[1:,11], _pos_evol[1:,12] ] ).T
            # _pos_evol = np.array(_pos_evol)
            # _uopt = np.array([_uopt[:,0], _pos_evol[1:,10], _pos_evol[1:,11], _pos_evol[1:,12] ] ).T

            _opt_state = opt_state_pos if (_curr_ctrl == 'pos' or _curr_ctrl == 'none')  else opt_state_traj

            # if time.time() - _ctime > 2.0:
            #     rospy.logwarn("MPC process: {} control, solve time: {:.3f} ms".format(_curr_ctrl, _perf_time*1000.0))
            #     _ctime = time.time()

            with self._u_opt_lock:
                # Store the control
                self.u_opt_shr[:] = _uopt
                # Store the angular rate evolution
                self.w_opt_shr[:] = _wopt
                self.opt_info_shr[self.key2index_info_mpc['sample_time_posmpc']] = sample_time
                self.opt_info_shr[self.key2index_info_mpc['solveTime']] = _perf_time
                self.opt_info_shr[self.key2index_info_mpc['avg_linesearch']] = float (_opt_state.avg_linesearch)
                self.opt_info_shr[self.key2index_info_mpc['stepsize']] = float (_opt_state.stepsize)
                self.opt_info_shr[self.key2index_info_mpc['num_steps']] = float (_opt_state.num_steps)
                self.opt_info_shr[self.key2index_info_mpc['grad_norm']] = float (_opt_state.grad_sqr)
                self.opt_info_shr[self.key2index_info_mpc['avg_stepsize']] = float (_opt_state.avg_stepsize)
                self.opt_info_shr[self.key2index_info_mpc['cost0']] = float (_opt_state.init_cost)
                self.opt_info_shr[self.key2index_info_mpc['costT']] = float (_opt_state.opt_cost)

            # self.sync_event.set()


    def initialize_mpc_callback(self, req):
        """ Service callback to set the trajectory """
        res = LoadTrajAndParamsResponse()

        if self._run_trajectory or self._pos_control:
            # warn the user of the problem
            rospy.logwarn("Cannot set the trajectory because the controller is running")
            res.success = False
            return res

        # Controller initialized
        self.mpc_on = MPCMotorsCMD.MPC_RESET

        # Send this message 5 times to make sure that the controller is reset
        for _ in range(5):
            self.pub_cmd_setpoint(self.curr_time, self.sample_time)
            rospy.sleep(0.01)
        # Log that the reset onboard is done
        rospy.loginfo("Reset the controller done")

        self.reset_done = True
        res.success = True
        return res


    def start_trajectory_callback(self, req):
        """ Service callback to start the trajectory """
        res = FollowTrajResponse()
        # If the controller is not reset yet, we cannot
        if not self.reset_done and req.state_controller != FollowTrajRequest.CTRL_INACTIVE:
            rospy.logwarn("Cannot start the trajectory because the controller is not reset: Do controller_init before!")
            res.success = False
            return res

        # Get the requested mode
        mode = req.state_controller
        self._target_sp = req.target_pose

        if mode == FollowTrajRequest.CTRL_TEST:
            # We are in test mode
            self.mpc_on = MPCMotorsCMD.MPC_TEST
            self._test_mode = True
            self._pos_control = True
            self._run_trajectory = False
            self._trajec_time = -1.0
            # ros warn the user
            rospy.logwarn("Test mode activated")
            res.success = True
            return res

        # Check if position control is requested
        if mode == FollowTrajRequest.CTRL_POSE_ACTIVE:
            self.mpc_on = MPCMotorsCMD.MPC_ON
            self._pos_control = True
            self._test_mode = False
            self._run_trajectory = False
            self._trajec_time = -1.0
            # ros warn the user
            rospy.logwarn("Position control activated")
            res.success = True
            return res

        if mode == FollowTrajRequest.CTRL_INACTIVE:
            self.reset_done = False
            self.mpc_on = MPCMotorsCMD.MPC_OFF
            self._test_mode = False
            self._pos_control = False
            self._run_trajectory = False
            self._trajec_time = -1.0

            # Send off mode to the onboard controller
            for _ in range(5):
                self.pub_cmd_setpoint(self.curr_time, self.sample_time)
                rospy.sleep(0.01)

            # ros warn the user
            rospy.logwarn("Controller deactivated")
            res.success = True
            return res

        # Check if rtrajectory is already running
        if self._run_trajectory and mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE:
            rospy.logerr("The trajectory is already running")
            res.success = False
            return res

        self._pos_control = False
        self._test_mode = False
        self._run_trajectory = mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE
        self._trajec_time = 0.0 if (mode == FollowTrajRequest.CTRL_TRAJ_IDLE or mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE) else -1.0
        self.mpc_on = MPCMotorsCMD.MPC_ON

        # ros warn the user
        rospy.logwarn("run_trajectory_ = {}, trajec_time_ = {}".format(self._run_trajectory, self._trajec_time))
        res.success = True
        return res

    def publish_opt_state(self, _):
        """ The callback to publish the state of the MPC
        """
        # We extract information from self._optimizer_info
        # Build the message
        msg = OptMPCState()
        msg.stamp = rospy.Time.now()
        msg.avg_linesearch = self._optimizer_info[self.key2index_info_mpc['avg_linesearch']]
        msg.avg_stepsize = self._optimizer_info[self.key2index_info_mpc['avg_stepsize']]
        msg.stepsize = self._optimizer_info[self.key2index_info_mpc['stepsize']]
        msg.num_steps = int(self._optimizer_info[self.key2index_info_mpc['num_steps']])
        msg.grad_norm = self._optimizer_info[self.key2index_info_mpc['grad_norm']]
        msg.cost_init = self._optimizer_info[self.key2index_info_mpc['cost0']]
        msg.opt_cost = self._optimizer_info[self.key2index_info_mpc['costT']]
        msg.solve_time = self._optimizer_info[self.key2index_info_mpc['solveTime']]
        msg.callback_dt = self.dt_state_callback
        msg.state_dt = self.dt_state_info
        msg.ctrl_state = self.name_control_state_dict[self._control_state]
        msg.mpc_indx = self._index

        # Publish the message
        self._opt_state_pub.publish(msg)

    # def pub_reference_pose(self, tpub):
    #     """ Publish the reference pose """
    #     # Construct the pose message
    #     pose = PoseStamped()
    #     pose.header.stamp = tpub + rospy.Duration(self._dt_usec*1e-6)
    #     pose.header.frame_id = "map"
    #     pose.pose.position.x = self._target_x[0]
    #     pose.pose.position.y = self._target_x[1]
    #     pose.pose.position.z = self._target_x[2]
    #     # Fill in the orientation
    #     pose.pose.orientation.w = self._target_x[6]
    #     pose.pose.orientation.x = self._target_x[7]
    #     pose.pose.orientation.y = self._target_x[8]
    #     pose.pose.orientation.z = self._target_x[9]

    #     # Publish the message
    #     self._setpoint_pub.publish(pose)

    def pub_cmd_setpoint(self, curr_time, sample_time):
        """ Publish the command setpoint """
        self.mav.mav.mpc_motors_cmd_send(
            time_usec = int(curr_time.to_nsec() / 1000),
            motor_val_des = self._uopt,
            thrust_and_angrate_des = self._wopt,
            mpc_on = self.mpc_on,
            weight_motors = 0
        )


    def multi_process_shared_variables(self):
        """ Define and store the variables tht are going to be saved between this process and
            the mpc process
        """
        state = self.dummy_state()
        # Create a shared array for the current state
        self._curr_state_shr = shared_memory.SharedMemory(create=True, size=state.nbytes)
        self.curr_state_shr = np.ndarray(state.shape, dtype=state.dtype, buffer=self._curr_state_shr.buf)

        # # Create a numpy array from the shared memory
        # Create a shared array for the control returned for the mpc
        size_u = max(self.traj_uopt.nbytes, self.pos_uopt.nbytes)
        u_zero = self.traj_uopt if self.traj_uopt.nbytes > self.pos_uopt.nbytes else self.pos_uopt
        self._u_opt_shr = shared_memory.SharedMemory(create=True, size=size_u)
        self.u_opt_shr = np.ndarray(u_zero.shape, dtype=u_zero.dtype, buffer=self._u_opt_shr.buf)

        # Create shared memory for the angular rate and thrust setpoint
        # First input is the thrust and the last three are the angular rates
        wopt_zero = np.zeros((u_zero.shape[0], 4))
        self._w_opt_shr = shared_memory.SharedMemory(create=True, size=wopt_zero.nbytes)
        self.w_opt_shr = np.ndarray(wopt_zero.shape, dtype=wopt_zero.dtype, buffer=self._w_opt_shr.buf)

        # Data needed for the mpc solver
        info_mpc_pre = np.array([0.0, 0.0, self.control_state_dict['none']])
        self.key2index_pre = {'sample_time_prempc': 0, 'duration': 1, 'ctrl_state': 2}
        self._info_mpc_pre_shr = shared_memory.SharedMemory(create=True, size=info_mpc_pre.nbytes)
        self.info_mpc_pre_shr = np.ndarray(info_mpc_pre.shape, dtype=info_mpc_pre.dtype, buffer=self._info_mpc_pre_shr.buf)

        # Data needed after the mpc solver -> information about solving the mpc
        # Create a shared array for storing optimization info from the mpc
        opt_state = self.default_traj_opt_state
        info_mpc = np.array([-1.0, opt_state.avg_linesearch, opt_state.stepsize, opt_state.num_steps, opt_state.grad_sqr,
                                opt_state.avg_stepsize, opt_state.init_cost, opt_state.opt_cost, 0.0], dtype=np.float32)
        self.key2index_info_mpc = {'sample_time_posmpc': 0, 'avg_linesearch': 1, 'stepsize': 2,
                                    'num_steps': 3, 'grad_norm': 4, 'avg_stepsize': 5, 'cost0': 6, 'costT': 7, 'solveTime': 8}
        self._opt_info_shr = shared_memory.SharedMemory(create=True, size=info_mpc.nbytes)
        self.opt_info_shr = np.ndarray(info_mpc.shape, dtype=info_mpc.dtype, buffer=self._opt_info_shr.buf)
        self._optimizer_info = info_mpc

        # Create a shared memory for the target setpoint
        self._target_setpoint_shr = shared_memory.SharedMemory(create=True, size=state.nbytes)
        self.target_setpoint_shr = np.ndarray(state.shape, dtype=state.dtype, buffer=self._target_setpoint_shr.buf)

        # Create the Locks
        self._curr_state_lock = multiprocessing.Lock()
        self._u_opt_lock = multiprocessing.Lock()

        # Create the event to notify the mpc process
        self._mpc_event = multiprocessing.Event()
        self.sync_event = multiprocessing.Event()

    def clean_shared_variables(self):
        """ Delete the shared variables
        """
        self._curr_state_shr.close()
        self._curr_state_shr.unlink()
        self._u_opt_shr.close()
        self._u_opt_shr.unlink()
        self._w_opt_shr.close()
        self._w_opt_shr.unlink()
        self._opt_info_shr.close()
        self._opt_info_shr.unlink()
        self._target_setpoint_shr.close()
        self._target_setpoint_shr.unlink()
        self._info_mpc_pre_shr.close()
        self._info_mpc_pre_shr.unlink()


    def load_single_mpc(self, mpc_dir):
        """ Load a single MPC model from the given directory
        """
        x0 = self.dummy_state()
        cfg_dict, (m_reset, m_mpc), state_from_traj, _ = load_mpc_from_cfgfile(mpc_dir, convert_to_enu=True)

        # Now let's jit and lower state_from_traj
        _state_from_traj = None
        t_init = 0.01
        if state_from_traj is not None:
            rospy.logwarn("Compiling the state_from_traj function")
            start_time = time.time()
            # Define the reset function
            _state_from_traj = jax.jit(state_from_traj).lower(t_init).compile()
            rospy.logwarn("Compilation of the reset function took {} seconds".format(time.time() - start_time))

        # Now let's jit and lower the reset function
        rng_seed = jax.random.PRNGKey(self.seed)
        rospy.logwarn("Compiling the reset function")
        start_time = time.time()
        # Define the reset function
        _reset_traj_mpc = jax.jit(m_reset).lower(x=x0, rng=rng_seed, xdes = x0).compile()
        rospy.logwarn("Compilation of the reset function took {} seconds".format(time.time() - start_time))
        # Proceed to a first evaluation of the reset function and get the comutation time
        start_time = time.time()
        _default_traj_opt_state = _reset_traj_mpc(x=x0, rng=rng_seed, xdes = x0)
        _default_traj_opt_state.yk.block_until_ready()
        rospy.logwarn("First evaluation of the reset function took {} seconds".format(time.time() - start_time))

        # Now let's jit and lower the mpc function
        rospy.logwarn("Compiling the mpc function")
        start_time = time.time()
        _mpc_traj_solver = jax.jit(m_mpc).lower(x0, rng_seed, _default_traj_opt_state, curr_t=t_init, xdes = x0).compile()
        rospy.logwarn("Compilation of the mpc function took {} seconds".format(time.time() - start_time))
        # Proceed to a first evaluation of the mpc function and get the comutation time
        start_time = time.time()
        traj_uopt, _, _, _ = _mpc_traj_solver(x0, rng_seed, _default_traj_opt_state, curr_t=t_init, xdes = x0)
        traj_uopt.block_until_ready()
        _traj_uopt = np.array(traj_uopt)
        rospy.logwarn("First evaluation of the mpc function took {} seconds".format(time.time() - start_time))
        return (_state_from_traj, _reset_traj_mpc, _mpc_traj_solver, _traj_uopt, _default_traj_opt_state, cfg_dict)

    def start_mpc_process(self):
        """ Start the mpc process """
        # Create the process
        self.mpc_process = multiprocessing.Process(target=self.mpc_process_fn, name='mpc_process')
        # Start the process
        self.mpc_process.start()

    def stop_mpc_process(self):
        """ Stop the mpc process """
        # Terminate the process
        self.mpc_process.terminate()
        # Join the process
        self.mpc_process.join()
        # Clean the shared memory
        self.clean_shared_variables()

    def stop_state_callback_thread(self):
        """ Close the mavlink connection
        """
        # self.mpc_state_thread.terminate()
        self.mpc_state_thread.join()

    def dummy_state(self):
        """ Return a dummy state """
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


if __name__ == '__main__':
    import os

    # Initialize the node
    rospy.init_node('sde_control', anonymous=True)

    # Create the MPC node
    mpc_solver = SDEControlROS()

    # Spin
    rospy.spin()

    # Terminate and join the thread
    mpc_solver.stop_state_callback_thread()

    # Stop the mpc process
    mpc_solver.stop_mpc_process()

    # Shutdown
    rospy.signal_shutdown("MPC control node is terminated")

