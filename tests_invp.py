""" Battery of tests for invert_parameters and its generated files."""
import itertools
import pickle
import os
import re


import numpy as np
from tqdm import tqdm

import invert_parameters as invp
import import_data as imda


LOGS_PATH = '/home/dario/Proj_ActiveInference/results/logs_qs/'
OUTS_PATH = '/home/dario/Proj_ActiveInference/results/posta_qs/'


def likelihood_infer_parameters(max_prob=1, data_flat=None):
    """ Generate a set of data and the best model that fits the data, by making
    the posterior over actions equal one for the action that was chosen at each
    observation.

    Expected result: the loglikelihood should be 0.
    Assumptions:
        1) import_data works at it should
        2) The data handling procedure to divide by 10 and shift works well.
    """

    min_prob = 1 - max_prob
    if data_flat is None:
        _, data_flat = imda.main()
    data = data_flat[0]
    deci, trial, state, thres, _, _, indices = invp.preprocess_data(data)

    as_seen = {}
    count = 0
    non_repeated_obs = []
    for t in range(len(deci)):
        key = (1, 1, state[t], trial[t], thres[t])
        if key in as_seen.keys():
            count += 1
        else:
            non_repeated_obs.append(t)
            tmp = np.array([0, 0], dtype=float)
            tmp[deci[t]] = max_prob
            tmp[tmp == 0] = min_prob
            tmp /= tmp.sum()
            as_seen[key] = [tmp, 0.9]
    for field in ['choice', 'trial', 'threshold', 'obs', 'reihe']:
        data[field] = data[field][indices]
        data[field] = data[field][non_repeated_obs]
    print('Number of data points used: ', len(deci))
    logli, _, _, _, _, _ = invp.infer_parameters(as_seen=as_seen,
                                                 shape_pars=[
                                                     'unimodal_s', [1], [1]],
                                                 data_flat=data)
    print('Log-likelihood: ', logli)
    print('Repeated obs: ', count)
    raise Exception


def find_all_pars(subject=0, shape='unimodal_s', force_positive=False):
    """Goes through the results of inference (using the q files) and counts
    how many times each parameter set is found. Ideally, they should all be found
    384 times or a bit less. If anything less, there's a problem.

    If the Q file is not found, a small dict is returned which signals
    iterate_find_all_pars() that a problem was found.
    """
    try:
        q_seen = invp.load_or_calculate_q_file(subject, shape, create=False)
    except FileNotFoundError:
        return {1: 100, 2: 0}
    shape_pars = invp. __rds__(shape)

    big_index = itertools.product(*shape_pars[1:])
    num_pars = len(shape_pars[1:])
    found_pars = {}
    for key in q_seen.keys():
        par_key = key[:num_pars]
        if par_key in found_pars:
            found_pars[par_key] += 1
        else:
            found_pars[par_key] = 1

    found_pars_forced = {}
    for par_index in big_index:
        if par_index not in found_pars.keys():
            found_pars[par_index] = 0
            found_pars_forced[par_index] = 0
        else:
            found_pars_forced[par_index] = found_pars[par_index]
    if force_positive:
        return found_pars_forced
    else:
        return found_pars


def iterate_find_all_pars(subjects=None, shapes=None):
    """Iterates over find_all_pars() for all subjects, all shapes and reports
    which subject-shape combinations have problems.

    Parameters
    ----------
    subjects: int or list of ints. Default: first 20 subjects
        Subjects to use. 
    shapes: string or list of strings. Default: all in __rds__()
        Shapes to look for. 

    Returns
    ------
    flag: dict
        Dictionary whose keys are (subject, shape) pairs whose q files need
        some looking-into, because (possibly) not all values of the parameters
        were calculated.
    """
    if subjects is None:
        subjects = list(range(35))
    shape_pars_all = invp.__rds__()
    shapes = [shape_pars[0] for shape_pars in shape_pars_all]
    flag = set([])
    for subject in subjects:
        for shape in shapes:
            found_pars = find_all_pars(subject, shape, force_positive=True)
            max_pars = -np.inf
            for key in found_pars:
                max_pars = max(max_pars, found_pars[key])

            for key in found_pars.keys():
                if found_pars[key] < max_pars:
                    flag.add((subject, shape))
    return flag


def find_overwritten_files(subject, shape):
    """Finds those out/qut files that have more than one log file pointing to them. This
    indicates that the file was overwritten at some point.
    """
    logs_path = LOGS_PATH
    outs_path = OUTS_PATH

    num_pars = {'unimodal_s': 5, 'exponential': 4, 'sigmoid_s': 5}[shape]

    out_files, qut_files = invp.find_files_subject(
        subject, shape, LOGS_PATH, outs_path, outs_path)

    bad_files = [set(), set()]

    for ix_list, file_list in enumerate([out_files, qut_files]):
        for file in file_list:
            try:
                with open(outs_path + file, 'rb') as mafi:
                    one_out = pickle.load(mafi)
            except FileNotFoundError:
                bad_files[ix_list].add(file)
                continue
            for key in one_out.keys():
                if len(key) != num_pars:
                    bad_files[ix_list].add(file)
    return bad_files


def iterate_find_overwritten_files(subjects=range(35), shapes=None):
    """Iterates find_overwritten_files() for the given subjects and shapes."""

    if shapes is None:
        shapes = [shape_pars[0] for shape_pars in invp.__rds__()]

    bad_files = {}
    for subject in subjects:
        for shape in shapes:
            bad_files[(subject, shape)] = find_overwritten_files(
                subject, shape)
    return bad_files


def check_files_thorough(write_to_file=False, logs_path=None, outs_path=None, logs=None):
    """Goes through each log file and checks that its corresponding qut files
    contain the information they should.

    Returns a list of shape_pars and log files that need to be redone.
    """
    if logs_path is None or outs_path is None:
        logs_path = LOGS_PATH
        outs_path = OUTS_PATH

    bad_files = []
    if logs is None:
        logs = os.listdir(logs_path)

    for file in tqdm(logs):
        if file[-3:] != 'out':
            continue
        out_file, qut_file, shape_pars, subject = _find_shape_pars_from_file(
            file, logs_path)
        out_file = outs_path + out_file
        qut_file = outs_path + qut_file
        shape_pars = invp.switch_shape_pars(shape_pars)
        out_check = check_out_file(out_file, subject, shape_pars)
        qut_check = check_out_file(qut_file, subject, shape_pars)
        if not (out_check and qut_check):
            this_results = [file, out_file, qut_file, subject, shape_pars]
            bad_files.append(this_results)
    if write_to_file:
        with open('./output_thorough.pi', 'wb') as mafi:
            pickle.dump(bad_files, mafi)
    else:
        return bad_files


def check_out_file(out_file, subject, shape_pars, count=False):
    """Checks whether the out_file has all the posteriors corresponding to 
    the subject and the parameters in shape_pars.

    Takes the plain shape_pars.
    """
    _, data_flat = imda.main()
    data_flat = data_flat[subject]
    _, trial, state, thres, _, _, _ = invp.preprocess_data(data_flat)
    try:
        with open(out_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)
    except FileNotFoundError:
        return False
    return_flag = True
    k = 0
    for ix_trial, trial in enumerate(trial):
        index = shape_pars[1:]
        index.append(state[ix_trial])
        index.append(trial)
        index.append(thres[ix_trial])
        if tuple(index) not in as_seen:
            return_flag = False
            if count:
                k += 1
            else:
                break
    if count:
        return return_flag, k
    else:
        return return_flag


def _find_shape_pars_from_file(file, LOGS_PATH):
    """Extracts the information from the given file."""
    with open(LOGS_PATH + file, 'r') as mafi:
        text = mafi.read()
    # Get shape_pars
    re_shape_line = re.compile(
        r'Task and parameters: .*')  # \[[\'_A-Za-z0-9\[\],]*\]')
    re_shape_pars = re.compile(r'\[.*\]\]')
    shape_line = re_shape_line.findall(text)[0]
    shape_pars = re_shape_pars.findall(shape_line)
    shape_pars = eval(shape_pars[0])

    # Find subject number
    re_subject_line = re.compile(r'Subjects .*')
    re_subject = re.compile(r'[0-9][0-9]*')
    subject_line = re_subject_line.findall(text)[0]
    subject = int(re_subject.findall(subject_line)[0])

    # Find pid and create file names
    re_filename_line = re.compile(r'Saving posteriors .*')
    re_filename = re.compile(r'out_[0-9][0-9_]*\.pi')
    filename_line = re_filename_line.findall(text)[0]
    filename = re_filename.findall(filename_line)[0]
    out_file = filename
    qut_file = 'q' + filename[1:]
    return out_file, qut_file, shape_pars, subject


def find_log_for_out(out_file):
    """Finds all the log files which point to the --out_file--."""

    LOGS_PATH = '/home/dario/Proj_ActiveInference/results/logs_qs/'
    outs_path = '/home/dario/Proj_ActiveInference/results/posta_qs/'

    # In case it's full path:
    local_path = re.compile(r'(?:o|q)ut_.*\.pi')
    out_file = local_path.findall(out_file)[0]
    logs_found = []
    for log_file in os.listdir(LOGS_PATH):
        if log_file[-3:] != 'out':
            continue
        with open(LOGS_PATH + log_file, 'r') as mafi:
            text = mafi.read()
        find_out = re.compile(out_file)
        found_file = find_out.findall(text)
        if len(found_file) > 0:
            logs_found.append(log_file)
    return logs_found


def find_semi_bad_outs(load_from_file=True):
    """Finds those bad out_.pi and qut_.pi files which belong to more than one
    log file.

    The bad outs and quts are either taken from file or calculated with 
    find_log_for_out()
    """
    if load_from_file:
        with open('./bad_logs.pi', 'rb') as mafi:
            checks = pickle.load(mafi)
    else:
        checks = check_files_thorough()

    bad_logs = []
    for bad_file in checks:
        log_file, out_file, qut_file = bad_file[:3]
        log_out = find_log_for_out(out_file)
        #log_qut = find_log_for_out(qut_file)
        bad_logs.append(
            {'out': out_file, 'bad_log': log_file, 'other_logs': log_out})
    return bad_logs


def log_to_out():
    """Goes through each log file and sees which out file it points to. It builds a
    dictionary with {out: log} structure.
    """
    out_dict = {}
    for file in os.listdir(LOGS_PATH):
        if file[-3:] != 'out':
            continue
        out_file, _, _, _ = _find_shape_pars_from_file(file, LOGS_PATH)
        if out_file in out_dict:
            out_dict[out_file].append(file)
        else:
            out_dict[out_file] = [file]

    return out_dict
