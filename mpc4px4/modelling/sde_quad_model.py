import jax
import jax.numpy as jnp

import os

import numpy as np

import haiku as hk

import pickle

from mpc4px4.helpers import parse_ulog, load_yaml, update_params, apply_fn_to_allleaf

from mpc4px4.modelling.quad_model import * 

from mpc4px4.modelling.train_static_model import load_vector_field_from_file

from sde4mbrl.nsde import ControlledSDE
from sde4mbrl.nsde import create_sampling_fn, create_online_cost_sampling_fn

class SDEQuadModel(ControlledSDE):
    """SDE model of the quadrotor
    """
    def __init__(self, params, physics_informed_nominal, motor_speed_fn, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name=name)
        # This is a function that was learnt in static condition
        # and provide the vector field as F(x,pwm_in, Fresidual, Mresidual
        self.physics_informed_vfield = physics_informed_nominal
        self.motor_speed_fn = motor_speed_fn
        self.init_residual_networks()

    # Define a projection function specifically for the quaternions
    def projection_fn(self, x):
        # TODO: Differentiation issues here???
        quat_val = get_quat(x)
        norm_q = quat_val / jnp.linalg.norm(quat_val)
        return set_quat(x, norm_q)
    
    def prior_diffusion(self, x, extra_args=None):
        """Diffusion of the prior dynamics
        """
        # Noise magnitude
        # Position and orientation has no noise in the known model
        # Velocity and angular velocity has noise that depend on the noise model
        mag_noise = jnp.array([0., 0., 0.,  # Position -> to be ignored
                                1e-3, 1e-3, 1e-3, # Velocity
                                0., 0., 0., 0., # Quaternion noise, to be ignored
                                1e-2, 1e-2, 1e-2]) # Angular velocity
        # Now if can include more complex noise mode if requested
        if self.params['diffusion_type'] == 'constant':
            noise_vel = self.params['amp_noise_vel']
            noise_ang_vel = self.params['amp_noise_angular_vel']
            mag_noise = set_vel(mag_noise, jnp.array([noise_vel, noise_vel, noise_vel]))
            mag_noise = set_ang_vel(mag_noise, jnp.array([noise_ang_vel, noise_ang_vel, noise_ang_vel]))
        elif self.params['diffusion_type'] == 'zero':
            mag_noise = set_vel(mag_noise, jnp.zeros(3))
            mag_noise = set_ang_vel(mag_noise, jnp.zeros(3))
        else:
            raise ValueError('Unknown diffusion type')
        # Now we can return the diffusion term
        return mag_noise
    
    def aero_drag(self, v_b):
        """Aerodynamic drag prior model
        """
        # Now we can compute the drag force
        # We use the linear drag force assumption -> Initial estimate close to zero
        kdx = hk.get_parameter('kdx', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kdy = hk.get_parameter('kdy', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kdz = hk.get_parameter('kdz', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kh = hk.get_parameter('kh', shape=(), init=hk.initializers.RandomUniform(0., 0.00001))
        return jnp.array([-kdx*v_b[0], -kdy*v_b[1], -kdz*v_b[2] + kh*(v_b[0]**2 + v_b[1]**2)] )
        
    def ground_effect(self, x):
        """Ground effect
        """
        # TODO: Implement the ground effect
        # Project the position in the local frame
        # Then the results is dependent on the projected height


        # This is a hack for the thrust response to see if the static learning worked well
        k3 = hk.get_parameter('Ft', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
        return (1.0+k3) * jnp.ones(4)

    def compute_force_residual(self, x, v_b):
        """Compute the residual
        """
        # Create an MLP to compute the residual
        # The parameters are store in params dictionary under residual_forces
        Fres = jnp.zeros(3)
        # The residual terms are function of the velocity and angular velocity in the body frame
        if 'residual_forces' in self.params:
            Fres += self.residual_forces(jnp.array([v_b[0], v_b[1], v_b[2], x[10], x[11], x[12]]))
        if self.params.get('aero_drag_effect', False):
            Fres += self.aero_drag(v_b)
        return Fres
    
    def compute_moment_residual(self, x, v_b):
        """Compute the residual
        """
        # Create an MLP to compute the residual
        # The parameters are store in params dictionary under residual_forces
        Mres = jnp.array([0., 0., 0., 1., 1., 1.])
        if self.params['residual_moments'].get('resisdual', False):
            # Mres += self.residual_moments(jnp.array([v_b[0], v_b[1], v_b[2], x[10], x[11], x[12]]))
            Mres = Mres.at[:3].set(self.residual_moments(jnp.array([v_b[0], v_b[1], v_b[2], x[10], x[11], x[12]])))
        if self.params['residual_moments'].get('mot_coeff', False):
            # These extra coefficients are used to adjust the motor coefficients in case static learning is not good
            # The values should be close to 1, so Mx, My, Mz are close to 0
            # Create some parameters
            k1 = hk.get_parameter('Mx', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
            k2 = hk.get_parameter('My', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
            k3 = hk.get_parameter('Mz', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
            Mres = Mres.at[3:].set(1.0 + jnp.array([k1, k2, k3]))
        return Mres
    
    def init_residual_networks(self):
        """Initialize the residual deep neural networks
        """
        # Create the residual MLP
        # The parameters are store in params dictionary under residual_forces
        # The residual MLP is a function of the state and the control
        if 'residual_forces' in self.params:
            _act_fn = self.params['residual_forces']['activation_fn']
            self.residual_forces = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 3],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-0.001, 0.001),
                                                name = 'res_forces')

        # The residual MLP is a function of the state and the control
        if 'residual_moments' in self.params:
            _act_fn = self.params['residual_moments']['activation_fn']
            self.residual_moments = hk.nets.MLP([*self.params['residual_moments']['hidden_layers'], 3],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-0.001, 0.001),
                                                name = 'res_moments')


    def prior_drift(self, x, u, extra_args=None):
        """Drift of the prior dynamics
        """
        return self.physics_informed_vfield(x, u)

    
    def posterior_drift(self, x, u, extra_args=None):
        """Drift of the posterior dynamics
        """
        # We need to build the residual terms and ext_thrust terms
        # First we need to rotate the velocity vector in the body frame
        v_b = quat_rotatevectorinv(get_quat(x), get_vel(x))

        # Now we can compute the residual forces
        if 'residual_forces' not in self.params and not self.params.get('aero_drag_effect', False):
            Fres = None
        else:
            Fres = self.compute_force_residual(x, v_b)
        
        # Now we can compute the residual moments
        if 'residual_moments' not in self.params:
            Mres = None
        else:
            Mres = self.compute_moment_residual(x, v_b)
        
        # Now we can compute the external thrust
        if self.params.get('ground_effect', False):
            ext_thrust = self.ground_effect(x)
        else:
            ext_thrust = jnp.array([1.,1.,1.,1.])
        
        # Now we can compute the drift
        return self.physics_informed_vfield(x, u, Fres, Mres, ext_thrust)

def load_trajectory(log_dir):
    """Load the trajectories from the file
    """
    log_dir = os.path.expanduser(log_dir)
    # Extract the current directory without the filename
    log_dir_dir = log_dir[:log_dir.rfind('/')]
    # Extract the file name without the .ulog extension
    log_name = log_dir[log_dir.rfind('/')+1:].replace('.ulog','')
    # Try to load the data if it is already parsed
    try:
        with open(log_dir_dir + '/' + log_name + '_parsed.pkl', 'rb') as f:
            data = pickle.load(f)
            # Print that the data was loaded from the filtered file
            tqdm.write('Data loaded from already parsed file')
            return data
    except:
        pass

    log_data = parse_ulog(log_dir)
    # Ordered state names
    name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    name_controls = ['m1', 'm2', 'm3', 'm4']
    # Extract the states and controls
    x = np.stack([log_data[_name] for _name in name_states], axis=1)
    # Build the control action ndarray
    u = np.stack([log_data[_name] for _name in name_controls], axis=1)
    # Return the data
    m_res = {'y' : x, 'u' : u}
    with open(log_dir_dir + '/' + log_name + '_parsed.pkl', 'wb') as f:
        pickle.dump(m_res, f)
    return m_res

def load_nominal_model(learned_params_dir, modified_params ={}):
    """ Create a function to integrate the nominal model (No noise and using static learned parameters)
    """
    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)
    # vehicle parameters
    _model_params = learned_params['nominal']
    _static_params = learned_params['nominal']['learned_nominal']
    # SDE learned parameters
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # Load the vector field from file
    _, (vector_field_fn, motor_model_fn, *_) = load_vector_field_from_file(_static_params)
    # Load the sde model given the sde learned and the vector field utils
    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)
    # Set the zero noise setting
    params_model['diffusion_type'] = 'zero'
    # Create the model
    _, m_sampling = create_sampling_fn(params_model, sde_constr=SDEQuadModel, prior_sampling=True, 
                        physics_informed_nominal=vector_field_fn, motor_speed_fn=motor_model_fn)
    return lambda *x : m_sampling(_sde_learned, *x)

def load_predictor_function(learned_params_dir, prior_dist=False, modified_params ={}):
    """ Create a function to sample from the prior distribution or
        to sample from the posterior distribution
    """
    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)
    # vehicle parameters
    _model_params = learned_params['nominal']
    _static_params = learned_params['nominal']['learned_nominal']
    # SDE learned parameters
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # Load the vector field from file
    _, (vector_field_fn, motor_model_fn, *_) = load_vector_field_from_file(_static_params)
    # Load the sde model given the sde learned and the vector field utils
    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)
    # Create the model
    _, m_sampling = create_sampling_fn(params_model, sde_constr=SDEQuadModel, prior_sampling=prior_dist, 
                        physics_informed_nominal=vector_field_fn, motor_speed_fn=motor_model_fn)
    return lambda *x : m_sampling(_sde_learned, *x)

def load_mpc_solver(mpc_config_dir, modified_params ={}, nominal_model = False):
    """ Create an MPC solver that can be used at each time step for control
    """
    # Load the yaml configuration file
    _mpc_params = load_yaml(mpc_config_dir)
    # Get the path to the model parameters
    mpc_params = _mpc_params
    learned_params_dir = mpc_params['learned_model_params']
    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)
    # vehicle parameters
    _model_params = learned_params['nominal']
    _static_params = learned_params['nominal']['learned_nominal']
    # SDE learned parameters
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # Load the vector field from file
    _, (vector_field_fn, motor_model_fn, *_) = load_vector_field_from_file(_static_params)
    # Load the sde model given the sde learned and the vector field utils
    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)
    if nominal_model:
        # Set the zero noise setting
        params_model['diffusion_type'] = 'zero'
    # Create the function that defines the cost of MPC and integration
    # This function modifies params_model
    _, _multi_cost_sampling, vmapped_prox, _, construct_opt_params = create_online_cost_sampling_fn(params_model, mpc_params, sde_constr=SDEQuadModel,
                                    physics_informed_nominal=vector_field_fn, motor_speed_fn=motor_model_fn)
    # multi_cost_sampling =  lambda *x : _multi_cost_sampling(_sde_learned, *x)
    # Set the actual model params
    _mpc_params['model'] = params_model
    return (_sde_learned, _mpc_params), _multi_cost_sampling, vmapped_prox, construct_opt_params

def main_train_sde(yaml_cfg_file, output_file=None):
    """ Main function to train the SDE
    """
    # Load the yaml file
    cfg_train = load_yaml(yaml_cfg_file)

    # Obtain the path to the ulog files
    logs_dir = cfg_train['logs_dir']
    print('\nPath to the ulog files')

    # Load the data from the ulog file
    full_data = list()
    for log_dir in logs_dir:
        # Pretty print the path to the ulog files
        tqdm.write('\t - {}'.format(log_dir))
        _data = load_trajectory(log_dir)
        full_data.append(_data)
    
    # Load the test trajectory data
    test_traj_dir = cfg_train['test_trajectory']
    test_traj_data = load_trajectory(test_traj_dir)

    # Construct the extra arguments for SDE
    nominal_params_path = os.path.expanduser(cfg_train['vehicle_dir'] + '/my_models/' + cfg_train['model']['learned_nominal'])
    _model_params, (vector_field_fn, motor_model_fn, *_) = load_vector_field_from_file(nominal_params_path)
    cfg_train['model']['learned_nominal'] = _model_params

    # # Additonal parametrs
    # if output_file is not None:
    #     m_file_path = os.path.dirname(os.path.realpath(__file__))
    #     # Output directory
    #     output_dir = '{}/learned_models/'.format(m_file_path)
    #     output_file = output_dir+ output_file # Get the path the directory of this file
    output_file = os.path.expanduser(cfg_train['vehicle_dir'] + '/my_models/' + output_file)

    # TODO: Improve this stopping criteria
    def _improv_cond(opt_var, test_res, train_res):
        """Improvement condition
        """
        return opt_var['logprob'] > test_res['logprob']
    # Train the model
    train_model(cfg_train, full_data, test_traj_data, output_file, _improv_cond, SDEQuadModel, 
                physics_informed_nominal=vector_field_fn,
                motor_speed_fn=motor_model_fn)

if __name__ == '__main__':
    from tqdm.auto import tqdm
    from sde4mbrl.train_sde import train_model
    import argparse
    # Argument parser
    parser = argparse.ArgumentParser(description='Train the SDE model')
    parser.add_argument('--cfg', type=str, help='Path to the yaml training configuration file')
    parser.add_argument('--output', type=str, help='Path to the output file')
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function
    main_train_sde(args.cfg, args.output)
