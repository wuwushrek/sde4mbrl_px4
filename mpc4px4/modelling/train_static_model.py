import jax
import jax.numpy as jnp

import numpy as np

import haiku as hk
import optax

from jaxopt import PolyakSGD
from jaxopt import ArmijoSGD

from mpc4px4.modelling.quad_model import QuadDynamics
from mpc4px4.modelling.smoothing_data import filter_data
from mpc4px4.helpers import parse_ulog, load_yaml

import time, datetime
import pandas as pd

from tqdm.auto import tqdm

# Import yaml
import yaml
import argparse

import pickle

from jax.tree_util import tree_flatten
import os

import copy

def apply_fn_to_allleaf(fn_to_apply, dict_val):
    """Apply a function to all the leaf of a dictionary
    """
    res_dict = {}
    for k, v in dict_val.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = apply_fn_to_allleaf(fn_to_apply, v)
        else:
            res_dict[k] = fn_to_apply(v)
    return res_dict

def evaluate_loss_fn(loss_fn, m_params, data_eval, test_batch_size):
    """Compute the metrics for evaluation accross the data set

    Args:
        loss_fn (TYPE): A loss function lambda m_params, data : scalar
        m_params (dict): The parameters of the neural network model
        data_eval (iterator): The dataset considered for the loss computation
        num_iter (int): The number of iteration over the data set

    Returns:
        TYPE: Returns loss metrics
    """
    result_dict ={}

    num_test_batches = data_eval['x'].shape[0] // test_batch_size
    # Iterate over the test batches
    for n_i in tqdm(range(num_test_batches), leave=False):
        # Get the batch
        batch = {k : v[n_i*test_batch_size:(n_i+1)*test_batch_size] for k, v in data_eval.items()}
        # Infer the next state values of the system
        curr_time = time.time()
        # Compute the loss
        lossval, extra_dict = loss_fn(m_params, batch)
        lossval.block_until_ready()

        diff_time  = time.time() - curr_time
        extra_dict = {**extra_dict, 'Pred. Time' : diff_time}

        if len(result_dict) == 0:
            result_dict = {_key : np.zeros(num_test_batches) for _key in extra_dict}

        # Save the data for logging
        for _key, v in extra_dict.items():
            result_dict[_key][n_i] = v

    return {_k : np.mean(v) for _k, v in result_dict.items()}

def load_vector_field_from_file(data):
    """Load the vector field from a file

    Args:
        dat (str or dict): The path to the file or the dictionary containing 
            the parameters of the vector field

    Returns:
        TYPE: The vector field
    """
    # Open the yaml file containing the vehicle parmas
    if isinstance(data, str):
        with open(os.path.expanduser(data), 'r') as stream:
            try:
                vehicle_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        assert isinstance(data, dict), "The data must be a dictionary or a path to a yaml dict"
        vehicle_dict = data
    # Convert the dictionary to a jax array
    learned_params = apply_fn_to_allleaf(jnp.array, vehicle_dict['learned'])

    # Define the vector field
    vector_field_pred = lambda x, u, Fres=None, Mres=None, ext_thrust=jnp.array([1.,1.,1.,1.]) : \
                                QuadDynamics(vehicle_dict['prior']).vector_field(x, u, Fres, Mres, ext_thrust)
    # Define the motor model function
    motor_model = lambda pwm_in : QuadDynamics(vehicle_dict['prior']).get_single_motor_vel(pwm_in)
    # Define the the thrust function and moment function
    body_moment_from_actuator = lambda Oga_vect, x=None, ext_thrust=jnp.array([1.,1.,1.,1.]): QuadDynamics(vehicle_dict['prior']).body_moment_from_actuator(Oga_vect, x, ext_thrust)
    body_thrust_from_actuator = lambda Oga_vect, ext_thrust=jnp.array([1.,1.,1.,1.]): QuadDynamics(vehicle_dict['prior']).body_thrust_from_actuator(Oga_vect, ext_thrust)
    # Now transform all these functions into Haiku functions
    _vector_field_pred = hk.without_apply_rng(hk.transform(vector_field_pred))
    _motor_model = hk.without_apply_rng(hk.transform(motor_model))
    _body_moment_from_actuator = hk.without_apply_rng(hk.transform(body_moment_from_actuator))
    _body_thrust_from_actuator = hk.without_apply_rng(hk.transform(body_thrust_from_actuator))
    # Initialize the parameters of each of these functions
    _vector_field_pred.init(0, np.zeros((13,)), np.zeros((4,)))
    _motor_model.init(0, np.array(0.))
    _body_moment_from_actuator.init(0, np.zeros((4,)))
    _body_thrust_from_actuator.init(0, np.zeros((4,)))
    __vector_field_pred = lambda x, u, Fres=None, Mres=None, ext_thrust=jnp.array([1.,1.,1.,1.]): \
                                _vector_field_pred.apply(learned_params, x, u, Fres, Mres, ext_thrust)
    __motor_model = lambda pwm_in : _motor_model.apply(learned_params, pwm_in)
    # print(__motor_model(jnp.array(0.)))
    __body_moment_from_actuator = lambda Oga_vect, x=None, ext_thrust=jnp.array([1.,1.,1.,1.]) : \
                                    _body_moment_from_actuator.apply(learned_params, Oga_vect, x, ext_thrust)
    __body_thrust_from_actuator = lambda Oga_vect, ext_thrust=jnp.array([1.,1.,1.,1.]): \
                                    _body_thrust_from_actuator.apply(learned_params, Oga_vect, ext_thrust)
    return vehicle_dict, (__vector_field_pred, __motor_model, __body_moment_from_actuator, __body_thrust_from_actuator)
    

def init_data(log_dir, cutoff_freqs, force_filtering=False):
    """Load the data from the ulog file and return the data as a dictionary.

    Args:
        dataset_dir (str): The path to the ulog file

    Returns:
        dict: The data dictionary
    """
    # Extract the current directory without the filename
    log_dir_dir = log_dir[:log_dir.rfind('/')]
    # Extract the file name without the .ulog extension
    log_name = log_dir[log_dir.rfind('/')+1:].replace('.ulg','')

    # Check if a filtered version of the data already exists in the log directory
    if not force_filtering:
        # Check if the filtered data already exists
        try:
            with open(log_dir_dir + '/' + log_name + '_filtered.pkl', 'rb') as f:
                data = pickle.load(f)
                # Print that the data was loaded from the filtered file
                tqdm.write('Data loaded from filtered file')
                return data
        except:
            pass

    # Load the data from the ulog file
    tqdm.write('Loading data from the ulog file..')
    # In static condition we want to avoid ground effect
    outlier_cond = lambda d: d['z'] > 0.5
    log_data = parse_ulog(log_dir, outlier_cond=outlier_cond)
    # Ordered state names
    name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    name_controls = ['m1', 'm2', 'm3', 'm4']
    _reduced_cutoff_freqs = {k : cutoff_freqs[k] for k in name_states}
    # Filter the data as described in filter_data_analysis.ipynb
    tqdm.write('Filtering the data...')
    log_data_dot = filter_data(log_data, _reduced_cutoff_freqs, 
                    suffix_der='_dot', save_old_state=False, include_finite_diff = False, 
                    state_names=list(_reduced_cutoff_freqs.keys()))
    # Check if any nan values are present in the data
    for k, v in log_data_dot.items():
        if np.any(np.isnan(v)):
            raise ValueError('Nan values present in the data: {}'.format(k))
    # Check if any inf values are present in the data
    for k, v in log_data_dot.items():
        if np.any(np.isinf(v)):
            raise ValueError('Inf values present in the data: {}'.format(k))
    # Build the ndarray of derivative of states [smooth derivatives]
    x_dot = np.stack([log_data_dot['{}_dot'.format(_name)] for _name in name_states], axis=1)
    # Build the ndarray of states [smoothed states]
    x = np.stack([log_data_dot[_name] for _name in name_states], axis=1)
    # Build the control action ndarray
    u = np.stack([log_data[_name] for _name in name_controls], axis=1)
    # Print the size of the data
    tqdm.write('Size of the data : {}'.format(x.shape))
    m_res = {'x_dot' : x_dot, 'x' : x, 'u' : u}
    # Save the filtered data
    with open(log_dir_dir + '/' + log_name + '_filtered.pkl', 'wb') as f:
        pickle.dump(m_res, f)
    return m_res


def create_loss_function(prior_model_params, loss_weights, seed=0):
    """ Create the learning loss function and the parameters of the 
        approximation
    """
    # Predictor for the vector field
    vector_field_pred = lambda x, u : QuadDynamics(prior_model_params).vector_field(x, u)
    # Motor constraint predictor
    def motor_constraint_pred():
        qdyn = QuadDynamics(prior_model_params)
        mconstr = qdyn.motor_model_constraints()
        param_dev = qdyn.param_deviation_from_init()
        return mconstr, param_dev

    # Transform these functions into Haiku modules
    vector_field_pred_fn = hk.without_apply_rng(hk.transform(vector_field_pred))
    motor_constraint_pred_fn = hk.without_apply_rng(hk.transform(motor_constraint_pred))

    # Initialize the parameters of the vector field predictor
    vector_field_pred_params = vector_field_pred_fn.init(seed, np.zeros((13,)), np.zeros((4,)))
    # Initialize the parameters of the motor constraint predictor
    _ = motor_constraint_pred_fn.init(seed)

    # Create the loss function
    def _loss_fun(est_params, batch_xdot, batch_x, batch_u):
        """The loss function"""
        # Compute the vector field prediction
        xdot_pred = jax.vmap(lambda x, u : vector_field_pred_fn.apply(est_params, x, u), in_axes=(0,0))(batch_x, batch_u)
        # Compute the motor constraint prediction
        motor_constr_val, param_constr = motor_constraint_pred_fn.apply(est_params)
        # Compute the loss
        # only get the speed and angular velocity
        diff_x = xdot_pred - batch_xdot
        # loss_xdot = jnp.mean(jnp.square(diff_x))
        loss_xdot = jnp.mean(jnp.square(jnp.concatenate([diff_x[:,3:6], diff_x[:,10:13]], axis=0)))
        # err_vx_dot = diff_x[:,3:6]
        # err_w_dot = diff_x[:,10:13]
        # loss_xdot = jnp.mean(jnp.square(err_vx_dot) + jnp.mean(jnp.square(err_w_dot))

        loss_motor_constr = motor_constr_val # jnp.mean(jnp.square(motor_constr_val))
        loss_param_constr = param_constr

        # Compute the weights regularization
        loss_params = jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(jnp.square, est_params))

        # Compute the deviation of est_params from prior_model_params
        total_loss = loss_xdot * loss_weights['pen_xdot'] + loss_motor_constr * loss_weights['pen_motor_constr'] + loss_param_constr * loss_weights['pen_init_dev']
        total_loss += loss_weights['pen_params'] * loss_params

        return total_loss, {'total_loss' : total_loss,
                            'loss_xdot' : loss_xdot, 
                            'loss_mot' : loss_motor_constr, 
                            'loss_par' : loss_param_constr,
                            'norm_params' : loss_params}
    
    return vector_field_pred_params, _loss_fun, vector_field_pred_fn

def convert_dict_jnp_to_dict_list(d):
    """Convert a dictionary with jnp arrays to a dictionary with lists"""
    res_dict = {}
    for k, v in d.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = convert_dict_jnp_to_dict_list(v)
        else:
            list_value = v.tolist()
            res_dict[k] = list_value 
    return res_dict

def train_static_model(yaml_cfg_file, output_file=None, motor_model=None):
    """Train the static model

    Args:
        yaml_cfg_file (str): The path to the yaml configuration file
    """
    # Open the yaml file containing the configuration to train the model
    cfg_train = load_yaml(yaml_cfg_file)

    # Obtain the cutoff frequencies for the data filtering
    cutoff_freqs = cfg_train['cutoff_freqs']
    # Pretty print the cutoff frequencies
    print('\nCutoff frequencies for the data filtering')
    for k, v in cutoff_freqs.items():
        print('\t - {} : {}'.format(k, v))

    # Obtain the path to the ulog files
    logs_dir = cfg_train['logs_dir']
    print('\nPath to the ulog files')
    # Load the data from the ulog file
    full_data = list()
    for log_dir in tqdm(logs_dir):
        # Pretty print the path to the ulog files
        tqdm.write('\t - {}'.format(log_dir))
        _data = init_data(log_dir, cutoff_freqs, force_filtering=cfg_train['force_filtering'])
        full_data.append(_data)
    
    # Load the test trajectory data
    test_traj_dir = cfg_train['test_trajectory']
    test_traj_data = init_data(test_traj_dir, cutoff_freqs, force_filtering=cfg_train['force_filtering'])

    
    # Random number generator for numpy variables
    seed = cfg_train['seed']
    # Numpy random number generator
    m_numpy_rng = np.random.default_rng(seed)
    # Generate the JAX random key generator
    # train_rng = jax.random.PRNGKey(seed)

    # Get the path the directory of this file
    m_file_path = cfg_train['vehicle_dir']

    # Load the prior model parameters
    prior_model_params = load_yaml(m_file_path+'/'+cfg_train['prior_file'])
    if motor_model is not None:
        prior_model_params['motor_model'] = motor_model
    motor_model = prior_model_params['motor_model']

    # Prettu print the prior model parameters
    print('\nPrior model parameters')
    for k, v in prior_model_params.items():
        print('\t - {} : {}'.format(k, v))
    
    # Create the hk parameters and the loss function
    pb_params, loss_fun, _ = \
        create_loss_function(prior_model_params, cfg_train['loss'], seed=seed)
    # Pretty print the loss weights
    print('\nLoss weights')
    for k, v in cfg_train['loss'].items():
        print('\t - {} : {}'.format(k, v))
    # Pretty print the initial parameters of the vector field predictor
    print('\nInitial parameters of the vector field predictor')
    for k, v in pb_params.items():
        # Check if the parameter is a dictionary
        if isinstance(v, dict):
            # Print the key first 
            print('\t - {}'.format(k))
            # Print the subkeys
            for k2, v2 in v.items():
                print('\t\t - {} : {}'.format(k2, v2))
        else:
            print('\t - {} : {}'.format(k, v))
    
    # Define the multi_trajectory loss
    @jax.jit
    def actual_loss(est_params, data):
        """The actual loss function"""
        # Get the batch of data
        batch_xdot = data['x_dot']
        batch_x = data['x']
        batch_u = data['u']
        # Compute the loss
        loss, loss_dict = loss_fun(est_params, batch_xdot, batch_x, batch_u)
        return loss, loss_dict
    
    # Define the evaluation function
    eval_test_fn = lambda est_params: evaluate_loss_fn(actual_loss, est_params, test_traj_data, cfg_train['training']['test_batch_size'])

    # Create the optimizer
    # Customize the gradient descent algorithm
    print('\nInitialize the optimizer')
    optim = cfg_train['optimizer']
    special_solver = False
    if type(optim) is list:
        chain_list = []
        for elem in optim:
            m_fn = getattr(optax, elem['name'])
            m_params = elem.get('params', {})
            print('Function : {} | params : {}'.format(elem['name'], m_params))
            if elem.get('scheduler', False):
                m_params = m_fn(**m_params)
                chain_list.append(optax.scale_by_schedule(m_params))
            else:
                chain_list.append(m_fn(**m_params))
        # Build the optimizer to be initialized later
        opt = optax.chain(*chain_list)
        opt_state = opt.init(pb_params)
    else:
        # Create the optimizer
        if optim['name'] == 'PolyakSGD':
            opt_fun = PolyakSGD
        else:
            opt_fun = ArmijoSGD
        # Specify that the optimizer is a special solver (PolyakSGD or ArmijoSGD)
        special_solver = True
        # Initialize the optimizer
        opt = opt_fun(actual_loss, has_aux=True, jit=True, **optim['params'])
        opt_state = opt.init_state(pb_params, {k : v[:10,:] for k, v in full_data[0].items() })
    # Initialize the parameters of the neural network
    init_nn_params = pb_params

    @jax.jit
    def projection(paramns, data):
        """Project the parameters onto non-negative values and compute the loss"""
        return jax.tree_map(lambda x : jnp.maximum(x, 1e-6), paramns), actual_loss(paramns, data)[1]

    # Define the update function that will be used with no special solver
    @jax.jit
    def update(paramns, opt_state, data):
        """Update the parameters of the neural network"""
        # Compute the gradients
        grads, loss_dict = jax.grad(actual_loss, has_aux=True)(paramns, data)
        # Update the parameters
        updates, opt_state = opt.update(grads, opt_state, paramns)
        # Update the parameters
        paramns = optax.apply_updates(paramns, updates)
        # Do the projection
        paramns, loss_dict = projection(paramns, data)
        return paramns, opt_state, loss_dict
    
    # Utility function for printing / displaying loss evolution
    def fill_dict(m_dict, c_dict, inner_name, fstring):
        for k, v in c_dict.items():
            if k not in m_dict:
                m_dict[k] = {}
            m_dict[k][inner_name] = fstring.format(v)

    # Print the dictionary of values as a table in the console
    subset_key = cfg_train.get('key_to_show', None)
    pretty_dict = lambda d : pd.DataFrame({_k : d[_k] for _k in subset_key} \
                                            if subset_key is not None else d
                                          ).__str__()
    
    # Save the number of iteration
    itr_count = 0
    count_epochs_no_improv = 0

    # Save the loss evolution and other useful quantities
    opt_params_dict = init_nn_params
    opt_variables = {}
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    log_data_list = []
    parameter_evolution = []

    # Save all the parameters of this function
    m_parameters_dict = {'params' : cfg_train, 'seed' : seed}

    # Output directory
    output_dir = '{}/my_models/'.format(m_file_path)

    # Output file and if None, use the current data and time
    out_data_file = output_file+'_'+motor_model if output_file is not None else \
        'static_model_{}_{}'.format(motor_model, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # Open the info file to save the parameters information
    outfile = open(output_dir+out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('\n////// Command line messages \n\n')
    outfile.close()

    # Save the initial parameters in a yaml file
    with open(output_dir+out_data_file+'.yaml', 'w') as params_outfile:
        converted_params = convert_dict_jnp_to_dict_list(pb_params)
        # Add the prior parameters
        save_dict = {'learned' : converted_params, 'prior' : prior_model_params}
        yaml.dump(save_dict, params_outfile)
    
    # Iterate through the epochs
    training_params = cfg_train['training']

    # Extract the batch size
    batch_size = training_params['train_batch_per_traj']
    # Find the number of evals per epoch
    num_evals_per_epoch = np.max([ _data['x'].shape[0] // batch_size for _data in full_data ])

    for epoch in tqdm(range(training_params['nepochs'])):
        # Counts the number of epochs until cost does not imrpove anymore
        count_epochs_no_improv += 1

        # Iterate through the number of total batches
        for i in tqdm(range(num_evals_per_epoch), leave=False):
            log_data = dict()

            # Generate a bunch of random batch indexes for each trajectory in fulldata
            batch_idx = [ m_numpy_rng.choice(_data['x'].shape[0], batch_size, replace=False) \
                            for _data in full_data ]
            # Extract the data from the batch indexes
            batch_data = [ {k : v[batch_idx_ind,:] for k, v in _data.items()} \
                            for batch_idx_ind, _data in zip(batch_idx, full_data) ]
            # Concatenate the data
            batch_data = {k : np.concatenate([_data[k] for _data in batch_data], axis=0) \
                            for k in batch_data[0].keys()}
            
            if itr_count == 0:
                _train_dict_init = eval_test_fn(init_nn_params)
                _test_dict_init = copy.deepcopy(_train_dict_init)
            
            # Increment the iteration count
            itr_count += 1

            # Start the timer
            update_start = time.time()
            # Update the parameters
            if special_solver:
                # Update the parameters with the special solver
                pb_params, opt_state = opt.update(pb_params, opt_state, batch_data)
                tree_flatten(opt_state)[0][0].block_until_ready()
                # Projection onto non-negative values
                pb_params, _train_res = projection(pb_params, batch_data)
                # Get the stepsize of the optimizer from the opt_state
                # curr_stepsize = opt_state.stepsize
            else:
                # Update the parameters with the standard solver
                pb_params, opt_state, _train_res = update(pb_params, opt_state, batch_data)
                tree_flatten(opt_state)[0][0].block_until_ready()
                # Get the stepsize of the optimizer from the opt_state
                # curr_stepsize = -1.0
                
            update_end = time.time() - update_start
            # Include time in _train_res for uniformity with test dataset
            _train_res['Pred. Time'] = update_end
            # _train_res['Step Size'] = curr_stepsize

            # Total elapsed compute time for update only
            if itr_count >= 5: # Remove the first few steps due to jit compilation
                update_time_average = (itr_count * update_time_average + update_end) / (itr_count + 1)
                compute_time_update.append(update_end)
                total_time += update_end
            else:
                update_time_average = update_end

            # Check if it is time to compute the metrics for evaluation
            if itr_count % training_params['test_freq'] == 0 or itr_count == 1:
                # Print the logging information
                print_str_test = '----------------------------- Eval on Test Data [Iteration count = {} | Epoch = {}] -----------------------------\n'.format(itr_count, epoch)
                tqdm.write(print_str_test)

                # # Do some printing for result visualization
                # if itr_count == 1:
                #     _test_dict_init = copy.deepcopy(_train_res)
                
                # Compute the metrics on the test dataset
                _test_res = eval_test_fn(pb_params)
                # _test_res['']

                # First time we have a value for the loss function
                if itr_count == 1 or (opt_variables['total_loss'] >= _test_res['total_loss']):
                    opt_params_dict = pb_params
                    opt_variables = _test_res
                    count_epochs_no_improv = 0

                fill_dict(log_data, _train_res, 'Train', '{:.3e}')
                fill_dict(log_data, _test_res, 'Test', '{:.3e}')
                log_data_copy = copy.deepcopy(log_data)
                fill_dict(log_data_copy, opt_variables, 'Opt. Test', '{:.3e}')
                fill_dict(log_data_copy, _train_dict_init, 'Init Train', '{:.3e}')
                fill_dict(log_data_copy, _test_dict_init, 'Init Test', '{:.3e}')
                parameter_evolution.append(opt_params_dict)

                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += pretty_dict(log_data_copy)
                print_str += '\n Number epochs without improvement  = {}'.format(count_epochs_no_improv)
                print_str += '\n'
                # tqdm.write(print_str)

                # Pretty print the parameters of the model
                print_str += '----------------------------- Model Parameters -----------------------------\n'
                # Print keys adn values
                for key, value in opt_params_dict.items():
                    if isinstance(value, dict):
                        print_str += '{}: \n'.format(key)
                        for key2, value2 in value.items():
                            print_str += '    {}: \t OPT= {} \t CURR={} \n'.format(key2, value2, pb_params[key][key2])
                    else:
                        print_str += '{}: {} \n'.format(key, value)
                print_str += '\n'
                tqdm.write(print_str)

                # Save all the obtained data
                log_data_list.append(log_data)

                # Save these info of the console in a text file
                outfile = open(output_dir+out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            last_iteration = (epoch == training_params['nepochs']-1 and i == num_evals_per_epoch-1)
            last_iteration |= (count_epochs_no_improv > training_params['patience'])

            if itr_count % training_params['save_freq'] == 0 or last_iteration:
                m_dict_res = {'last_params' : pb_params,
                                'best_params' : opt_params_dict,
                                'total_time' : total_time,
                                'opt_values' : opt_variables,
                                'compute_time_update' : compute_time_update,
                                'log_data' : log_data_list,
                                'init_losses' : (_train_dict_init, _test_dict_init),
                                'training_parameters' : m_parameters_dict,
                                'parameter_evolution' : parameter_evolution}
                outfile = open(output_dir+out_data_file+'.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()

                # Save the initial parameters in a yaml file
                with open(output_dir+out_data_file+'.yaml', 'w') as params_outfile:
                    converted_params = convert_dict_jnp_to_dict_list(opt_params_dict)
                    # Add the prior parameters
                    save_dict = {'learned' : converted_params, 'prior' : prior_model_params}
                    yaml.dump(save_dict, params_outfile)

            if last_iteration:
                break
        if last_iteration:
            break


if __name__ == '__main__':
    # Parse the arguments
    # train_static_model.py --cfg cfg_sitl_iris.yaml --output_file test --motor_model cubic
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the yaml configuration file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to the output file')
    # Parse the motor model
    parser.add_argument('--motor_model', type=str, default='', help='Motor model to use: linear, quadratic, cubic, sigmoid_linear, sigmoid_quad')
    args = parser.parse_args()

    # Train the static model
    train_static_model(args.cfg, output_file=args.output_file, motor_model=args.motor_model if len(args.motor_model) > 0 else None)