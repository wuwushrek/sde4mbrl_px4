import rospy

from functools import partial
import numpy as np

import os

# MPC seems to be faster on cpu because of the loop
# TODO: check if this is still true, and investiage how to make it faster on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax 
import jax.numpy as jnp

from mpc4px4.srv import FollowTraj, FollowTrajRequest, FollowTrajResponse
from mpc4px4.srv import LoadTrajAndParams, LoadTrajAndParamsRequest, LoadTrajAndParamsResponse
from mpc4px4.msg import OptMPCState

from mpc4px4.helpers import load_trajectory, quatmult, quatinv
from mpc4px4.modelling.sde_quad_model import load_predictor_function, load_mpc_solver

# Accelerated proximal gradient import
from sde4mbrl.apg import init_apg, apg, init_opt_state
from sde4mbrl.nsde import compute_timesteps

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import MPCFullState, MPCMotorsCMD

import time


def parse_trajectory(_traj):
    """ Return the array of time and concatenate the other states
        _traj: a dictionary with the keys: t, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz
    """
    # List of states in order
    states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    time_val = jnp.array(_traj['t'])
    # stack the states
    state_val = jnp.stack([_traj[state] for state in states], axis=1)
    return time_val, state_val

def extract_next_targets_from_trajectory( _indx_start, _curr_t, _time_evol, _state_evol, time_steps, time_steps_spacing):
    """ Extract the next targets from the trajectory
        _indx_start:            An index guaranteed to be prior of the current time step
        _curr_t:                The current time step
        _time_evol:             The time evolution of the trajectory
        _state_evol:            The state evolution of the trajectory
        time_steps:             The array of time steps along the control horizon
        time_steps_spacing:     The spacing equivalent of time_steps wrt to the trajectory
    """
    assert _time_evol.shape[0] >= 2, "The trajectory must have at least 2 points"
    # Get the index of the next time step
    _curr_time_range = jnp.array([ _time_evol[jnp.minimum(i+_indx_start, _time_evol.shape[0]-1)] for i in range(time_steps_spacing[-1]+1)])
    indx_next = _indx_start + jnp.searchsorted(_curr_time_range, _curr_t+time_steps[0])
    # Obtain the next time steps
    _tnext_array = _curr_t + time_steps
    _indx_array = indx_next - 1 + time_steps_spacing
    # Clipping the indexes and time steps
    _tnext_array = jnp.clip(_tnext_array, _time_evol[0], _time_evol[-1])
    _indx_array = jnp.clip(_indx_array, 0, _time_evol.shape[0]-2)
    # A function to extrapolate the next state
    extr_fn = lambda _i, _t: _state_evol[_i] + (_state_evol[_i+1] - _state_evol[_i]) * (_t - _time_evol[_i]) / (_time_evol[_i+1] - _time_evol[_i])
    return jax.vmap(extr_fn)(_indx_array, _tnext_array), jnp.clip(indx_next-1, 0, _time_evol.shape[0]-1)


def extract_convenience(_time_evol, time_steps):
    """ From the time evolution of the trajectory, extract the time steps spacing that matches the one
        imposed by the control horizon and the mpc controller
    """
    _dt_traj = np.mean(np.diff(_time_evol))
    _time_steps_spacing = np.round(time_steps / _dt_traj).astype(jnp.int32)
    # _time_steps_spacing = _time_steps_spacing.at[0].set(0)
    _time_steps_spacing[0] = 0
    return np.cumsum(_time_steps_spacing), _dt_traj


# Define a cost function
# The goal is to takeoff to 3.0m of altitude and stay there
def cost_f(x, u, extra_args, wcoeffs):
    """ x is a (N, 13) array
    u is a (N, 4) array
    The cost function is a scalar
    """
    xref = extra_args
    # Compute the error on the quaternion
    qerr = jnp.sum(jnp.square(quatmult(x[6:10], quatinv(xref[6:10], jnp), jnp)[1:]) * jnp.array(wcoeffs['qerr']))
    # Compute the error on the rest of the states
    perr = jnp.sum(jnp.square(x[:6] - xref[:6]) * jnp.array(wcoeffs['perr']))
    # Compute the error on the angular velocity
    werr = jnp.sum(jnp.square(x[10:] - xref[10:]) * jnp.array(wcoeffs['werr']))
    # Compute the cost
    uerr = jnp.sum(jnp.square(u - jnp.array(wcoeffs['uref'])) * jnp.array(wcoeffs['uerr']))
    return perr + qerr + werr + uerr

def reset_apg(y, cfg_dict, construct_opt_params, pol_fn=None):
    """ Reset the optimizer based on the current value of the quadrotor state
    """
    if pol_fn is not None:
        u_init = pol_fn(y)
    else:
        u_init = jnp.ones(4) * jnp.array(cfg_dict['cost_params']['uref'])
    opt_init = construct_opt_params(u=u_init)
    opt_state = init_opt_state(opt_init, cfg_dict['apg_mpc'])
    return opt_state

def apg_mpc(xt, rng, past_solution, target_x, multi_cost_sampling, proximal_fn, cfg_dict, _sde_learned):
    """ Run one step of the MPC controller
        Return the state of the controller at the next time step
    """
    # Split the key
    rng_next, rng = jax.random.split(rng)

    red_cost_f = lambda _x, _u, extra_args: cost_f(_x, _u, extra_args, cfg_dict['cost_params'])
    cost_xt = lambda u_params: multi_cost_sampling(_sde_learned, xt, u_params, rng, 
                                    red_cost_f, extra_cost_args=target_x)[0]

    # Shift the solution so that it can be exploited
    opt_state = init_apg(past_solution.x_opt, cost_xt, 
                        cfg_dict['apg_mpc'],
                        momentum=past_solution.momentum,
                        stepsize=past_solution.avg_stepsize)

    new_opt_state = apg(opt_state, cost_xt, cfg_dict['apg_mpc'],
                            proximal_fn=proximal_fn)

    # Compute the next states of the system
    _, vehicle_states = multi_cost_sampling(_sde_learned, xt, new_opt_state.x_opt, rng, red_cost_f, extra_cost_args=target_x)
    vehicle_states = vehicle_states[0] # Only first sample of vehicle states

    uopt = new_opt_state.x_opt[:, :4]
    # project uopt -> No need here
    uopt = jnp.clip(uopt, 0.0, 1.0)
    # More memory effiicient by not returning the full state
    new_solution = new_opt_state._replace(x_opt=new_opt_state.x_opt.at[:-1].set(new_opt_state.x_opt[1:]))
    # new_solution = new_opt_state
    # We rescale back the engine torque data
    return rng_next, uopt, vehicle_states, new_solution


class SDEControlROS:
    """ A ROS node to control the quadcopter using my sde-based approach
    """
    def __init__(self, dir_sde_config, seed=0, nominal_model=False, report_dt=0.2):
        """ Constructor """
        # Info that the controller started
        rospy.loginfo("Starting the SDE controller")

        # Define the parameters
        self._cfg_dict =None
        self._sde_learned = None

        # Random number generator
        self._rng = jax.random.PRNGKey(seed)

        # Load the model and the mpc solver
        self.load_model(dir_sde_config, self._rng, nominal_model)

        # Current trajectory information
        self._traj = None
        self._time_evol = None
        self._time_steps_spacing = None
        self.extract_ts = None

        # # TODO: Clean this
        # self.last_sampled_time = None

        # State evolution of the trajectory
        self._current_stage = 0
        self._run_trajectory = False
        self._trajec_time = -1.0

        # Target state
        self._target_sp = None
        self._target_x = self.dummy_state()
        self._pos_control = False

        # # The current state of the quadcopter
        self._curr_state = None
        self._last_time = rospy.Time.now()
        # self._last_time_display = rospy.Time.now()

        # Subscriber for the full state of the quadcopter
        self._state_sub = rospy.Subscriber("mavros/mpc_full_state/state", MPCFullState, self.mpc_state_callback, queue_size=1)

        # Publisher for the setpoint
        self._setpoint_pub = rospy.Publisher("mavros/desired_setpoint", PoseStamped, queue_size=10)
        # Publisher for the control input
        self._control_pub = rospy.Publisher("mavros/mpc_motors_cmd/cmd", MPCMotorsCMD, queue_size=1)
        # Publisher for the mpc state
        self._opt_state_pub = rospy.Publisher("mavros/mpc_opt_state/state", OptMPCState, queue_size=10)

        # Service to set the trajectory
        self._set_traj_srv = rospy.Service("set_trajectory_and_params", LoadTrajAndParams, self.set_trajectory_callback)
        # Service to start the trajectory
        self._start_traj_srv = rospy.Service("start_trajectory", FollowTraj, self.start_trajectory_callback)

        # Create a timer to publish on _opt_state_pub
        self._timer = rospy.Timer(rospy.Duration(report_dt), self.publish_opt_state)
    
    def set_trajectory_callback(self, req):
        """ Service callback to set the trajectory """
        res = LoadTrajAndParamsResponse()
        # We only update this controller when this controller is not on
        # TODO: Load this in a separate thread?
        if self._run_trajectory or self._pos_control:
            # warn the user of the problem
            rospy.logwarn("Cannot set the trajectory because the controller is running")
            res.success = False
            return res
        
        if req.controller_param_yaml != "":
            # We have to realod the model and moc solver
            try:
                self.load_model(req.controller_param_yaml, self._rng, nominal_model=self._nominal_model)
            except Exception as e:
                rospy.logerr("Cannot load the model and the MPC solver")
                rospy.logerr(e)
                res.success = False
                return res

        if req.traj_dir_csv != "":
            try:
                traj_dict = load_trajectory(req.traj_dir_csv)
                self._time_evol, self._traj = parse_trajectory(traj_dict)
                self._traj_last_time = float(self._time_evol[-1])
                self._traj_last_stage = len(self._time_evol) - 1
                self._init_traj_pts = np.array(self._traj[0])
                self._end_traj_pts = np.array(self._traj[-1])
                # We need to compute other necessary information
                self._time_steps_spacing, _ = extract_convenience(self._time_evol, self._time_steps)
                self._time_steps_spacing = np.array(self._time_steps_spacing)   
                self._extract_ts = np.cumsum(np.array(self._time_steps))
                # Lower and compile the function to extract the next states
                start = time.time()
                self.extract_targets_jit = jax.jit(lambda idx, t: extract_next_targets_from_trajectory(idx, t, self._time_evol, self._traj, self._extract_ts, self._time_steps_spacing)).lower(self._current_stage, 0.0).compile()
                end = time.time()
                rospy.loginfo("Time to compile the trajectory extraction function: {}".format(end-start))
                start = time.time()
                states, _ = self.extract_targets_jit(self._current_stage, 0.2)
                states.block_until_ready()
                end = time.time()
                rospy.loginfo("Time to run the trajectory extraction function: {}".format(end-start))
            except Exception as e:
                rospy.logerr("Cannot load the trajectory: {}".format(e))
                res.success = False
                return res
        
        res.success = True
        return res
    
    def start_trajectory_callback(self, req):
        """ Service callback to start the trajectory """
        res = FollowTrajResponse()
        mode = req.state_controller
        self._target_sp = req.target_pose
        # Check if position control is requested
        if mode == FollowTrajRequest.CTRL_POSE_ACTIVE:
            self._pos_control = True
            self._run_trajectory = False
            self._trajec_time = -1.0
            self._current_stage = 0
            # ros warn the user
            rospy.logwarn("Position control activated")
            res.success = True
            return res

        if mode == FollowTrajRequest.CTRL_INACTIVE:
            self._pos_control = False
            self._run_trajectory = False
            self._trajec_time = -1.0
            self._current_stage = 0
            # ros warn the user
            rospy.logwarn("Controller deactivated")
            res.success = True
            return res
        
        # Check if the trajectory is set
        if self._traj is None or self._time_evol is None:
            rospy.logerr("The trajectory is not set")
            res.success = False
            return res
        
        # Check if rtrajectory is already running
        if self._run_trajectory and mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE:
            rospy.logerr("The trajectory is already running")
            res.success = False
            return res
        
        self._pos_control = False
        self._run_trajectory = mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE
        self._trajec_time = 0.0 if (mode == FollowTrajRequest.CTRL_TRAJ_IDLE or mode == FollowTrajRequest.CTRL_TRAJ_ACTIVE) else -1.0
        self._current_stage = 0
        # ros warn the user
        rospy.logwarn("run_trajectory_ = {}, trajec_time_ = {}, current_stage_ = {}".format(self._run_trajectory, self._trajec_time, self._current_stage))
        res.success = True
        return res

    def trajectory_automata(self, dt):
        """ This function is called when the trajectory is running """
        if self._trajec_time < 0.0:
            return False
        
        # What is the current time in trajectory
        current_time_val = self._trajec_time
        next_time_val = current_time_val + dt

        if next_time_val >= self._traj_last_time:
            # We are done and will send the last state
            self._trajec_time = self._traj_last_time
            self._current_stage = self._traj_last_stage
            self._target_x = self._end_traj_pts
            self.target_states = self.jit_setpoints_from_state(self._target_x)
            return True
        
        if not self._run_trajectory:
            self._target_x = self._init_traj_pts
            self.target_states = self.jit_setpoints_from_state(self._target_x)
            # We are not running the trajectory
            self._trajec_time = 0.0
            self._current_stage = 0
            return True
        
        # We are running the trajectory -> extract the next targets
        self.target_states,  next_stage = self.extract_targets_jit(self._current_stage, current_time_val)
        self._target_x = np.array(self.target_states[0, :])

        self._current_stage = int(next_stage)
        self._trajec_time = next_time_val
        return True
    
    def display_optimizer_state(self):
        m_dict = {'avg_linesearch': self.opt_state.avg_linesearch, 
                'stepsize' : self.opt_state.stepsize,
                'num_steps' : self.opt_state.num_steps, 
                'grad_norm': self.opt_state.grad_sqr,
                'avg_stepsize': self.opt_state.avg_stepsize, 
                'cost0': self.opt_state.init_cost, 
                'costT': self.opt_state.opt_cost,
                'solveTime': self.mpc_solve_time}
        rospy.loginfo(' | '.join([ '{} : {:.3e}'.format(k,v) for k, v in m_dict.items() ]))
        
    def publish_opt_state(self, _):
        """ The callback to publish the state of the MPC
        """
        # Build the message 
        msg = OptMPCState()
        msg.stamp = rospy.Time.now()
        msg.avg_linesearch = self.opt_state.avg_linesearch
        msg.avg_stepsize = self.opt_state.avg_stepsize
        msg.avg_momentum = self.opt_state.avg_momentum
        msg.stepsize = self.opt_state.stepsize
        msg.num_steps = int(self.opt_state.num_steps)
        msg.grad_norm = self.opt_state.grad_sqr
        msg.cost_init = self.opt_state.init_cost
        msg.opt_cost = self.opt_state.opt_cost
        msg.solve_time = self.mpc_solve_time
        msg.cost_final = self.opt_state.curr_cost
        # Publish the message
        self._opt_state_pub.publish(msg)

    def mpc_state_callback(self, msg):
        """ Callback for the full state of the quadcopter.
            This is also where the main control computation is going to be done.
            So we compute control when we receive new position
        """
        _perf_time = time.time()

        # Get the current time
        curr_time = rospy.Time.now()
        
        # COmpute the delta time
        dt = (curr_time - self._last_time).to_sec()

        # Update the last time
        self._last_time = curr_time

        # Get the time the sample was taken
        sample_time = msg.time_usec

        # Get the current state from the msg
        self._curr_state = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz, msg.qw, msg.qx, msg.qy, msg.qz, msg.wx, msg.wy, msg.wz], dtype=np.float32)

        # Check if we need to send cmd point to the quadcopter
        success = False

        if self._pos_control:
            # We are in position control mode
            self._target_x = np.array([self._target_sp.pose.position.x, self._target_sp.pose.position.y, self._target_sp.pose.position.z,
                                      0., 0., 0., 
                                      self._target_sp.pose.orientation.w, self._target_sp.pose.orientation.x, self._target_sp.pose.orientation.y, self._target_sp.pose.orientation.z,
                                      0., 0., 0.], dtype=np.float32)
            # Get the set of next setpoints
            self.target_states = self.jit_setpoints_from_state(self._target_x)
            success = True
        else:
            success = self.trajectory_automata(dt)

        if not success:
            # Reset the mpc controller
            # self.opt_state = self._reset_fn(self._target_x)
            return
        
        # Compute the optimal control
        self._rng, self._uopt, self.states, self.opt_state = self._apg_mpc(self._curr_state, self._rng, self.opt_state, self.target_states)
        self._uopt.block_until_ready()
        self._uopt = np.array(self._uopt)

         # Total performance
        self.mpc_solve_time = time.time() - _perf_time

        # Publish the control
        self.pub_cmd_setpoint(curr_time, sample_time)

        # Publish the target setpoint
        self.pub_reference_pose(curr_time)
        
        # _perf_total = time.time() - _perf_time
        # if _perf_total > 0.015:
        #     rospy.logwarn("MPC is not running in real time: {} > 0.015".format(_perf_total))
    
    def pub_reference_pose(self, tpub):
        """ Publish the reference pose """
        # Construct the pose message
        pose = PoseStamped()
        pose.header.stamp = tpub + rospy.Duration(self._dt_usec*1e-6)
        pose.header.frame_id = "map"
        pose.pose.position.x = self._target_x[0]
        pose.pose.position.y = self._target_x[1]
        pose.pose.position.z = self._target_x[2]
        # Fill in the orientation
        pose.pose.orientation.w = self._target_x[6]
        pose.pose.orientation.x = self._target_x[7]
        pose.pose.orientation.y = self._target_x[8]
        pose.pose.orientation.z = self._target_x[9]

        # Publish the message
        self._setpoint_pub.publish(pose)
    
    def pub_cmd_setpoint(self, curr_time, sample_time):
        """ Publish the command setpoint """
        # Construct the motors cmd message
        cmd_v = MPCMotorsCMD()
        cmd_v.time_usec = int(curr_time.to_nsec() / 1000)
        cmd_v.time_init = sample_time
        cmd_v.dt = self._dt_usec
        cmd_v.m1 = self._uopt[:cmd_v.HORIZON_MPC,0]
        cmd_v.m2 = self._uopt[:cmd_v.HORIZON_MPC,1]
        cmd_v.m3 = self._uopt[:cmd_v.HORIZON_MPC,2]
        cmd_v.m4 = self._uopt[:cmd_v.HORIZON_MPC,3]

        # Publish the message
        self._control_pub.publish(cmd_v)
    
    def dummy_state(self):
        """ Return a dummy state """
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def setpoints_from_state(self, state):
        """ Return the setpoints from the state """
        # Duplicate the state over the time horizon of the MPC
        xref = jnp.tile(state, (self._cfg_dict['horizon'], 1))
        return xref

    def load_model(self, dir_sde_config, init_rng, nominal_model=False):
        """ Load the model and the mpc solver.
            Compile the necessary jax functions to be used later for control
        """
        # Load the configuration
        # Path configuration file
        (_sde_learned, cfg_dict), multi_cost_sampling, vmapped_prox, construct_opt_params = \
            load_mpc_solver(dir_sde_config, modified_params={}, nominal_model=nominal_model)
        
        # Proximal function -> projection on the bound constraints
        proximal_fn = None if vmapped_prox is None else lambda x, stepsize=None: vmapped_prox(x)
        pol_fn = None # Used later for the MPC
        self._nominal_model = nominal_model

        # Print the configuration file
        rospy.logwarn("Configuration file:")
        print(cfg_dict)

        # COmpute and save the time step
        self._time_steps = np.array(compute_timesteps(cfg_dict['model']), dtype=np.float32)
        # TODO: Change the type to be uint16_t or uint8_t
        self._dt_usec = self._time_steps[0] * 1e6
        # ROSWARN the time step
        rospy.logwarn("Time step: {}".format(self._time_steps))

        # Save the configuration
        self._cfg_dict = cfg_dict
        self._sde_learned = _sde_learned

        # We need to compile these functions ahead on time on the CPU
        # Otherwise, the first call to the MPC will be very slow
        # We use the dummy state to compile the functions
        self._dummy_x = self.dummy_state()
        self._dummy_xref = self.setpoints_from_state(self._dummy_x)
        # ROS log the size of the reference trajectory
        rospy.loginfo("Size of the reference trajectory: {}".format(self._dummy_xref.shape))
        # We compile the reset function

        rospy.logwarn("Compiling the reset function")
        start_time = time.time()
        # Define the reset function
        self._reset_fn = jax.jit(lambda y: reset_apg(y, self._cfg_dict, construct_opt_params, pol_fn=pol_fn)).lower(self._dummy_x).compile()
        rospy.logwarn("Compilation of the reset function took {} seconds".format(time.time() - start_time))

        # Get the initial state of the optimizer
        start_time = time.time()
        self.opt_state = self._reset_fn(self._dummy_x)
        self.opt_state.num_steps.block_until_ready()
        rospy.logwarn("Reset of the optimizer took {} seconds".format(time.time() - start_time))

        # Log warn the compilation of the MPC function
        rospy.logwarn("Compiling the MPC function")
        start_time = time.time()
        # Define the apg_mpc
        self._apg_mpc = jax.jit(lambda xt, rng, past_solution, target_x: apg_mpc(xt, rng, past_solution, target_x, multi_cost_sampling, proximal_fn, self._cfg_dict, self._sde_learned)).lower(self._dummy_x, init_rng, self.opt_state, self._dummy_xref).compile()
        rospy.logwarn("Compilation of the MPC function took {} seconds".format(time.time() - start_time))

        start_time = time.time()
        _,_, _, self.opt_state = self._apg_mpc(self._dummy_x, init_rng, self.opt_state, self._dummy_xref)
        self.opt_state.num_steps.block_until_ready()
        self.mpc_solve_time = time.time() - start_time
        rospy.logwarn("First call to the MPC took {} seconds".format(self.mpc_solve_time))
        self.display_optimizer_state()

        # Jit the function to get the setpoint
        self.jit_setpoints_from_state = jax.jit(self.setpoints_from_state).lower(self._dummy_x).compile()
        self.jit_setpoints_from_state(self._dummy_x)


if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('apg_mpc_node', anonymous=True)
    # Extract the parameters
    dir_sde_config = rospy.get_param('dir_sde', "/home/franckdjeumou/catkin_ws/src/mpc4px4/launch/iris_mpc_sde.yaml")
    # Seed number
    seed = rospy.get_param('seed', 0)
    # Nominal model
    nominal_model = rospy.get_param('nominal_model', False)
    # Looging freq of the MPC
    mpc_logging_freq = rospy.get_param('mpc_log_dt', 0.2)

    # Create the MPC node
    mpc_solver = SDEControlROS(dir_sde_config, seed, nominal_model, mpc_logging_freq)
    rospy.spin()

