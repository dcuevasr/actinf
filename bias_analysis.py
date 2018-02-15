"""Plots and stuff for the bias thingy."""

import pickle

import numpy as np

import invert_parameters as invp

SUBJECTS = range(35)


def best_model(subjects=None, shapes=None, filename=None, force_bias=None,
               force_alpha=None, force_rule=True, **kwargs):
    """Returns, for each subject in --subjects--, the best model, including
    the bias parameter, limited by those shapes included in --shapes--.

    Parameters
    ----------
    shapes : list
    Shapes to consider. See invp.__rds__() for available shapes. Even when
    a single shape is used, it must be in a list; e.g. ['exponential'].

    filename : str
    If provided, it is used as the base for the filenames where the log-
    likelihoods are stored. It is assumed that different shapes are stored
    in different files, appended by the shape name.

    Returns
    -------
    best_pars : dict
    A key for each subject in --subjects--. Each element has the same
    confusing shape as the one in pr.iterate_best_pars(), which is as follows:
    ([logli], [alpha, bias, [shape, p1, p2, ...]]). This confusing shape is kept
    for compatibility purposes, as the outputs from this function and that for
    pr.iterate_best_pars() can be used interchangeably in all functions that
    do not require the bias element.
    """
    shape_pars_all = invp.__rds__()
    par_nums = {shape_pars[0]: [len(par) for par in shape_pars[1:]] for
                shape_pars in shape_pars_all}
    if subjects is None:
        subjects = SUBJECTS

    if shapes is None:
        shapes = ['exponential']

    if filename is None:
        filename = 'logli_bias_%s.pi'
    best_logli = {subject:-500 for subject in subjects}
    best_pars = {subject: (0,) for subject in subjects}
    for shape in shapes:
        shape_pars = invp.__rds__(shape)
        with open(filename % shape, 'rb') as mafi:
            logli = pickle.load(mafi)
        for key in logli.keys():
            c_sub = key[0]
            if c_sub not in best_logli.keys():
                continue
            if not(force_bias is None) and force_bias != key[2]:
                continue
            if not(force_alpha is None) and force_alpha != key[1]:
                continue
            c_max_logli = logli[key].max()
            if c_max_logli > best_logli[c_sub]:

                ix_ix = np.unravel_index(logli[key].argmax(), par_nums[shape])
                par_values = [shape_pars[c_ix + 1][ix_ix[c_ix]]
                              for c_ix in range(len(ix_ix))]
                if force_rule:
                    c_shape_pars = [shape] + par_values
                    if not invp.rules_for_shapes(c_shape_pars):
                        continue
                best_logli[c_sub] = c_max_logli
                alpha = float(int(key[1] * 10)) / 10
                best_pars[c_sub] = (np.array([c_max_logli]),
                                    [[alpha, key[2], [shape] +
                                      [par for par in par_values]]])
    return best_pars

def concatenate_bias_files(shape=None, output_file=None, input_file=None):
    """Takes the individual files created by script_bias.py and saves them to a
    single file."""

    if shape is None:
        shape = 'exponential'

    if input_file is None:
        input_file = './data/bias_logli_%s_%s.pi' % ('%d', shape)

    if output_file is None:
        output_file = 'logli_bias_%s.pi' % shape
    logli = {}
    ix_file = 0
    while True:
        try:
            with open(input_file % ix_file, 'rb') as mafi:
                logli.update(pickle.load(mafi))
        except FileNotFoundError:
            break
        ix_file += 1
    with open(output_file, 'wb') as mafi:
        pickle.dump(logli, mafi)


def max_logli_single_bias(subjects=None, shapes=None, verbose=False):
    """Finds the value of the bias that has the highest likelihood, assuming 
    the same value for all subjects.

    This is for model comparison, to determine whether a single bias value is
    a better BIC-fit to the data than the full, free-bias thing.
    """
    if subjects is None:
        subjects = SUBJECTS
    
    bias_vec = invp.__rds__(bias=True)

    best_bias = 0
    best_logli = -np.inf

    habemus_bias = set(bias_vec)
    
    for bias in bias_vec:
        best_pars = best_model(subjects, shapes=shapes, force_bias=bias)
        if best_pars[subjects[0]] == (0, ):
            habemus_bias.remove(bias)
            continue
        
        c_logli = 0
        for subject in subjects:
            c_logli += best_pars[subject][0][0]
        if c_logli > best_logli:
            best_logli = c_logli
            best_bias = bias

    if verbose:
        print('Considered biases: ', habemus_bias)
    return best_bias, best_logli


def fix_bias_names(filename=None, backup=True):
    """Fix the stupid numpy float-to-dict_key problem."""
    if filename is None:
        filename = './logli_bias_exponential.pi'

    filename_bak = filename[:-3] + '_bk.pi'
        
    with open(filename, 'rb') as mafi:
        logli = pickle.load(mafi)

    logli_new = {}

    for key in logli.keys():
        new_key = []
        for keyval in key:
            new_key.append(int(keyval * 10) / 10)
        logli_new[tuple(new_key)] = logli[key]

    if backup:
        with open(filename_bak, 'wb') as mafi:
            pickle.dump(logli, mafi)
        
    with open(filename, 'wb') as mafi:
        pickle.dump(logli_new, mafi)
