import copy
import numpy as np

import pynumdiff
import pynumdiff.optimize

from tqdm.auto import tqdm

def fft_signal(x, dt):
    '''Compute the fourrier transform of a signal --> cutoff_frequency analysis'''
    n = x.size
    fhat = np.fft.fft(x,n)
    PSD = np.real(fhat * np.conjugate(fhat)/n)
    freq = (1.0/(dt*n)) * np.arange(n)
    L = np.arange(1, np.floor(n/2), dtype='int')
    return freq[L], 20*np.log10(PSD[L])

def fft_data(dict_data, state_names = None, suffix_fft='_fft'):
    """Compute the frequency domain of data specified in state_names
       The resulting frequency domain is used for estimating the cutoff_freq
       required by filter_data function

    Args:
        dict_data (dict): Log from the vehicle
        state_names (List, optional): The states we want to do frequency analysis
        suffix_fft (str, optional): The suffix to add for the new variable

    Returns:
        TYPE: Description
    """
    new_dict_data = copy.deepcopy(dict_data)
    if state_names is None:
        state_names = [key for key in dict_data]
    _t = dict_data["t"]
    _dt = np.mean(_t[1:] - _t[:-1])
    for name in tqdm(state_names, leave=False):
        vVaue = dict_data[name]
        freq, ff_vValue = fft_signal(vVaue, _dt)
        new_dict_data[name+suffix_fft] = ff_vValue
        new_dict_data[name+suffix_fft+'_freq'] = freq
    return new_dict_data

def filter_data(dict_data, cutoff_freqs,
                state_names = None, suffix_der='dot',
                include_finite_diff = False,
                save_old_state = False,
                filter_method=(pynumdiff.optimize.smooth_finite_difference.friedrichsdiff,
                                pynumdiff.smooth_finite_difference.friedrichsdiff)
               ):
    """Filter the state  model according to the cutoff_freq as described by
        the framework of https://github.com/florisvb/PyNumDiff
        [TODO] : Add an example

    Args:
        dict_data (dict): Log from the vehicle
        cutoff_freqs (dict): cutoff frequency for each state in the log
        state_names (List, optional): The states we want to filter
        suffix_der (str, optional): The suffix to add for the new variable
        include_finite_diff (bool, optional): Save finite difference of state
        save_old_state (bool, optional): Save the old state
        filter_method (TYPE, optional): The method employed for filtering

    Returns:
        TYPE: A new dictionary of data with filtered data in addition
                to the old data. The filtered data are named according
                to the old data + a suffix given by suffix_der
    """
    # state_names = ['r', 'V', 'beta', 'delta']
    new_dict_data = copy.deepcopy(dict_data)
    if state_names is None:
        state_names = [key for key in dict_data]
    _t = dict_data["t"]
    _dt = np.mean(_t[1:] - _t[:-1])
    if _dt <= 0: # Sanity check -> No filter apply if dt=0 in the data
        return {}
    for name in tqdm(state_names, leave=False):
        vVaue = dict_data[name]
        # [TODO] The value '50' is the highest frequency to cut if not given
        # cutoff_freq estimated by by (a) counting real # peaks per second in the data
        # or (b) look at power spectra and choose cutoff.
        log_gamma = -1.6*np.log(cutoff_freqs.get(name, 50.)) -0.71*np.log(_dt) - 5.1
        tvgamma = np.exp(log_gamma)
        # Save also smoothing by doing finite difference
        if include_finite_diff:
            _, new_dict_data[name+'_fd'] = \
                    pynumdiff.finite_difference.first_order(vVaue, _dt)
        if save_old_state:
            new_dict_data[name+'_old'] = dict_data[name]

        # Filter parameters
        filter_params, _ = filter_method[0](vVaue, _dt, tvgamma=tvgamma)
        new_dict_data[name], new_dict_data[name+suffix_der] = \
                                    filter_method[1](vVaue, _dt, filter_params)
    return new_dict_data

def find_consecutive_true(metrics, min_length=-1):
    """Return the set of indices of minimum length 'min_length' for which
    the boolean array 'metrics' is True.
    Example: Typically, this can be used to identify the sideslip angle offset
             or where the model is actually invertible

    Args:
        metrics (TYPE): Description
        min_length (TYPE, optional): Description

    Returns:
        TYPE: Description
    """
    full_auto = metrics
    section_inds_ = np.split(np.r_[:len(full_auto)], np.where(np.diff(full_auto) != 0)[0]+1)

    full_auto_inds = []

    for inds_ in section_inds_:
        if full_auto[inds_[0]] and len(inds_) > min_length:
            full_auto_inds.append(inds_)
    return full_auto_inds