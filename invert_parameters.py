#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:46:59 2017

@author: dario

Methods for sampling the entire parameter space for the Kolling task, in which
the goals are the one free parameter, which itself is parametrized in terms of
the sufficient statistics of a normal distribution.

Data is created for a number of games, with a 'true' parametrization for the
goals, which is not normal (maybe change this?). Then, runs are simulated with
the active inference agent for all the parameter values.

"""
from time import time
import betClass as bc
import numpy as np
import pickle
import atexit
import multiprocessing as mp
import itertools


def _remove_above_thres(deci, trial, state, thres, reihe, multiplier = 1.2):
    """Removes any observation that goes above the cutoff. """

    indices = state <= np.array(multiplier*thres, dtype=int)
    return deci[indices], trial[indices], state[indices], thres[indices], reihe[indices]


def infer_parameters(mu_range = None, sd_range = None, num_games = None,
                     data = None):
    """ Calculates the likelihood for every model provided.

    Each model is created with different values of the mean and sd of a
    Gaussian distribution for the goals of active inference (log-priors). These
    models are used for generating posterior over actions for all the
    observations contained in Data. These posteriors are then used to calculate
    the likelihood for each model.

    If no data is provided, it is simulated with betClass.py. [This has been
    deprecated (in principle) by the code in main() that simulates data before
    infer_parameters() is called.]

    The file ./data/posteriors.pi is used to store a dictionary whose keys are
    the tuples (mu, sd, obs, trial, threshold) and its elements are the
    posteriors over actions for that combination of model parameters and
    observations. During the execution, the data in that file is loaded to
    memory and before calculating a posterior with active inference, the
    dictionary is checked for the appropriate key. If the key is found, the
    data therein is used. This saves a lot of execution time. When this
    function is executed, the data inside is read, and if new observations are
    found in the data, they are added to the dictionary. At the end,
    _save_posteriors() is called and the possibly-enhanced dictionary is saved
    anew to the file.

    Note that the _save_posteriors() function is called with the module
    atexit, so that if the execution is halted (e.g. via KeyboardInterrupt) the
    preliminary results are saved.

    It is best to use the function main(), which will generate data using the
    given parameters or the import_data module, and run infer_parameters() as
    needed. But, you know, do whatever you want. I ain't gonna stop you.

    Parameters
    ----------
    mu_range: (int, int)
        Selects the range for the grid search on mu. It will search all
        integers in the range [int, int].
    sd_range: (int, int)
        Selects the range for the grid search on sd. It will search all
        integers in the range [int, int].
    num_games: int or tuple of ints
        Select which games to use with 'small' or 'simulated'. Defaults to all
        available games in the data, or 20 for 'simulated'.
    data: dict
        Contains the data (observations) of the subjects (or simulated). Should
        contain the following keys:
            'obs': np.array[nObs], dtype = int
                Observations for all trials. One element per observation.
            'trials': np.array[nObs], dtype = int
                Trial number for each observation.
            'threshold': np.array[nObs], dtype = int
                Threshold used for every observation.
            'choice': np.array[nObs], dtype = int
                Choice made by the subject at each observation.
            'reihe': np.array[nObs], dtype = int
                Action pair (see Kolling_2014) offered to the subject at each
                trial.
            'TargetLevels': np.array[(1)], dtype = int
                All distinct threshold in the trial. It's equivalent to
                np.unique(data['threshold']).


    Returns
    -------
    prob_data: np.array([nMu, nSd]), dtype = float
        Posterior over models (assuming flat priors)
    post_act: np.array([nMu, nSd, nObs, 2]), dtype = float
        Posteriors over actions for every model and every trial in the data.
    deci, trial, state: each a np.array([nObs]), dtype = int
        Decisions, trial numbers and observations in the data. #TODO: delete?
    mu_sigma: np.array([nMu, nSd, 2]), dtype = int
        Values of Mu and Sigma used to build the models.
    """

    print('Running...')
    flag_write_posta = False
    t_ini = time()


    if data is None:
        (mabes, deci, trial, state, thres, posta,
         preci, stateV, nD, nT) = simulate_data(num_games)
#        max_state = mabes.nS
#        Ns = mabes.Ns
        as_seen = {}
        flag_write_posta = False #Don't write to file if simulated data
    else:
        deci, trial, state, thres, reihe = (data['choice'], data['trial'],
                                            data['obs'], data['threshold'],
                                            data['reihe'])
        state = np.round(state/10).astype(int)
        thres = np.round(thres/10).astype(int)
        deci, trial, state, thres, reihe = _remove_above_thres(deci, trial,
                                                        state, thres, reihe)
        target_levels = np.round(data['TargetLevels']/10)
#        return 1, 1, 1, trial, state, thres

        _shift_states(state, thres, reihe)
#        nD = data['NumberGames']
#        nT = data['NumberTrials']
        NuData = thres.size
        data_file = './data/posteriors.pi'
        try:
            with open(data_file, 'rb') as mafi:
                as_seen = pickle.load(mafi)
                print('File opened; data loaded.')
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
                as_seen = {}
                print('%s file not found, or contains no valid data.' % data_file)
    # Read the file of already-encountered observations
    #%% Create mesh of parameter values
    if mu_range is None:
        min_mean = -10 # grid search values for thres + mu
        max_mean = 10
    else:
        min_mean, max_mean = mu_range
        max_mean += 1 #so the input mu_range=(mu1, mu2) is inclussive.
    if sd_range is None:
        min_sigma = 10 # >0
        max_sigma = 20 # Maximum value for the variance
    else:
        min_sigma, max_sigma = sd_range
        max_sigma += 1

    mu_size = max_mean - min_mean
    sd_size = max_sigma - min_sigma
    tini = time()

    mabed = {}
    for lvl in target_levels: #one actinf for every threshold
        mabed[lvl] = bc.betMDP(nT = 8, nS = np.round(lvl*1.2).astype(int),
                               thres = int(lvl))
    print('Calculating log-priors (goals)...', end='', flush = True)
    mu_sigma = np.zeros((mu_size, sd_size, 2))
#    arr_lnc = np.zeros((max_mean-min_mean, max_sigma-min_sigma, len(state), mabes.Ns))
    arr_lnc = {}
#    return 1, 1, 1, trial, state, thres
    for m, mu in enumerate(range(min_mean, max_mean)):
        for sd, sigma in enumerate(range(min_sigma,max_sigma)):
            for s in range(len(state)):
                mu_sigma[m, sd, :] = [mu, sigma]
                if (mu, sigma, state[s], trial[s], thres[s]) not in as_seen:
                    flag_write_posta = True # write to file if new stuff found
                    arr_lnc[(m, sd, s)] = np.log(mabed[thres[s]].set_prior_goals(
                                                     selectShape='unimodal',
                                                     Gmean = mu+thres[s],
                                                     Gscale= sigma,
                                                     just_return = True,
                                                     convolute = True,
                                                     cutoff = False)
                                                     +np.exp(-16))

    print('Took %d seconds.' % (time() - tini))

    # Do an atexit call in case I interrupt the python execution with ctrl+c,
    # it will save the calculations so far into the file, as it would normally
    # do when exiting:


    print('Calculating posteriors over actions for all observations...',
          end='', flush = True)
    tini = time()
    #%% Calculate data likelihood for all parameter values
    posta_inferred = simulate_posteriors(min_mean, max_mean, min_sigma,
                                         max_sigma, as_seen,
                                         state, trial, thres,
                                         arr_lnc, mabed, flag_write_posta, mu_sigma)



    print('Took: %d seconds.' % (time() - tini))
    atexit.unregister(_save_posteriors)
    if flag_write_posta is True:
        _save_posteriors(posta_inferred, trial, state, thres, mu_sigma)

    post_act = np.zeros((mu_size, sd_size,NuData, 2))
    for keys in posta_inferred.keys():
        post_act[keys[0], keys[1], keys[2],:] = posta_inferred[keys][0]

    #%%
    print('Calculating likelihoods...', end='', flush = True)
    tini = time()
    likelihood_model = _calculate_likelihood(deci, post_act)
    print('Took: %d seconds.' % (time() - tini))

    # Marginalize
    marginal_data = likelihood_model.sum()
    prob_data = likelihood_model/marginal_data

    print('The whole thing took: %d seconds.' % (time() - t_ini))
    smiley = np.random.choice([':)', 'XD', ':P', '8D'])
    print('Finished. Have a nice day %s\n' % smiley)
    return prob_data, post_act, deci, trial, state, mu_sigma

def simulate_posteriors(min_mean, max_mean, min_sigma, max_sigma, as_seen,
                        state, trial, thres, arr_lnc, mabed, wflag, mu_sigma):
    posta_inferred = {}
    from signal import signal, SIGTERM
    def _save_posteriors_local(signum, frame):
        import os
        """ Wrapper to send to _save_posteriors from signal.

        Note that posta_inferred, defined above, is mutable. So when this
        function is called, the current value of posta_inferred will be used,
        instead of the empty dictionary.
        """
        _save_posteriors(posta_inferred, trial, state, thres, mu_sigma)
        os._exit(0)


    if wflag is True:
        atexit.register(_save_posteriors, posta_inferred,
                        trial, state, thres, mu_sigma)
        signal(SIGTERM, _save_posteriors_local) # because background processes ignore SIGINT

    T = max(trial)+1 #need this +1 since they start at 0
    def get_Vs(trial):
        return np.array(list(itertools.product(range(0,2),repeat = T - trial)), dtype=int)
    le_V = {}
    for t in range(T):
        le_V[t] = get_Vs(t)

    for m, mu in enumerate(range(min_mean, max_mean)):
        for sd, sigma in enumerate(range(min_sigma,max_sigma)):
            for s in range(len(state)):
#                mabed.thres = thres[s]
                # If state>max_state, replace by max_state.
#                obs[cstate] = 1
                if (mu, sigma, state[s], trial[s], thres[s]) in as_seen:
                    posta_inferred[(m, sd, s)] = as_seen[(mu, sigma, state[s], trial[s], thres[s])]
                    assert posta_inferred[(m, sd, s)] != (), \
                              'as_seen returned empty for mu = %d, '\
                              'sd = %d and obs = %d, t = %d, thres = %d'\
                              % (m, sd, state[s], trial[s], thres[s])
                else:
                    mabed[thres[s]].lnC = arr_lnc[(m, sd, s)]
                    posta_inferred[(m, sd, s)] = mabed[thres[s]].posteriorOverStates(state[s],
                               trial[s], le_V[trial[s]], mabed[thres[s]].D, 0, 15)[1:3]
                    assert posta_inferred[(m, sd, s)] != (), \
                              'posteriorOverStates returned empty for mu = %d, '\
                              'sd = %d and obs = %d, t = %d, thres = %d'\
                              % (m, sd, state[s], trial[s], thres[s])

    return posta_inferred
def call_mabe(le_input):
    """ Wrapper to call mabe.posteriorOverStates. It needs to be on the main
    namespace to be pickable for pool.
    """
    mabe = le_input[0]
    ckey = le_input[1]
    return [ckey, mabe.posteriorOverStates(*le_input[2])[1:3]]
def simulate_posteriors_par(min_mean, max_mean, min_sigma, max_sigma, as_seen,
                        state, trial, thres, arr_lnc, mabed, dummy1, dummy2):
    class multipros(object):
        """ Methods for using with pool.eval_async below."""
        def __init__(self, posta):
            self.posta = posta

        def save_to_dict(self, le_input):
            """ Saves the output from posteriorOverStates (here received as
            input) into the dictionary self.posta.
            """
            self.posta[le_input[0]] = le_input[1]


    posta_inferred = {}
    mamu = multipros({})
    to_do = set()
    for m, mu in enumerate(range(min_mean, max_mean)):
        for sd, sigma in enumerate(range(min_sigma,max_sigma)):
            for s in range(len(state)):
#                mabed.thres = thres[s]
                # If state>max_state, replace by max_state.
#                obs[cstate] = 1
                if (mu, sigma, state[s], trial[s], thres[s]) in as_seen:
                    posta_inferred[(m, sd, s)] = as_seen[(mu, sigma, state[s], trial[s], thres[s])]
                else:
                    to_do.add((m, sd, s, mu, sigma))
    with mp.Pool(8) as pool:
        res = []
        for nums in to_do:
            m, sd, s, mu, sigma = nums
#            print(type(m), type(sd), type(s), type(thres), flush=True)
            mabed[thres[s]].lnC = arr_lnc[(m, sd, s)]
            le_input = [mabed[thres[s]], (m, sd, s), [state[s], trial[s], mabed[thres[s]].V,
                        mabed[thres[s]].D, 0, 15]]
            res.append(pool.apply_async(call_mabe, [le_input], callback = mamu.save_to_dict))
#            posta_inferred[(m, sd, s)] = mabed[thres[s]].posteriorOverStates(state[s],
#                       trial[s], mabed[thres[s]].V, mabed[thres[s]].D, 0, 15)[1:3]
        for r in res:
            r.get()
    posta_inferred.update(mamu.posta)
    return posta_inferred

def simulate_data(num_games = 10, paradigm = None, nS = 100, thres = None,
                  mu = None, sd = None):
    """ Simulates data for the kolling experiment using active inference.

    Parameters
    ----------
        num_games: int
            Number of games to simulate (each with 8 trials or so).
        paradigm: string
            Paradigm to use (see betClass.py for more on this).
        thres: int
            Threshold to use when calculating lnC for the agent
        nS: int
            Number of total (non-convoluted) states.
        mu, sd: int
            Hyperparameters for calculating lnC. When provided, the base
            lnC is overwritten by a normal with the given parameters. See
            betClass for more info. mu should be provided as a relative value,
            with respect to the threshold. For example, for a threshold of 10,
            mu=-1 would mean that the normal is centered at 10-1 = 9.

    Returns
    -------
    (nD = number of games; nT = number of trials per game; nO =
              number of observable states)
        mabes       {class} Instance of betClass used for the data.
        data        {nD*nT} Decisions made by the agent at every observation.
        trial       {nD*nT} Trial number for all observations.
        state       {nD*nT} Observed states.
        posta       {nD*nT, 2} Posterior over actions for each observation.
        preci       {nD*nT} Precision after each observation.
        stateV      {nD*nT, nO} Observed states in [0,0,...,1,0,...,0] form.
        nD
        nT
    """
    flag_lnc = False
    if mu + sd == 1:
        raise ValueError('mu and sd must both be provided')
    elif mu + sd == 2:
        flag_lnc = True

    mabes = bc.betMDP(nS = nS, thres = thres)
    if flag_lnc:
        mabes.set_prior_goals(selectShape='unimodal',
                              Gmean = mu+thres, Gscale= sd,
                              just_return = False, convolute = True,
                              cutoff = False)

    nD = num_games  # Number of games played by subjects
    nT = mabes.nT
    deci = np.zeros(nD*nT, dtype=np.float64)
    trial = np.zeros(nD*nT, dtype=np.int64)
    state = np.zeros(nD*nT, dtype=np.int64)
    posta = np.zeros((nD*nT, 2), dtype=np.float64)
    preci = np.zeros((nD*nT), dtype=np.float64)
    for d in range(nD):
        mabes.exampleFull()
        deci[d*nT:(d+1)*nT] = mabes.Example['Actions']
        trial[d*nT:(d+1)*nT] = range(nT)
        state[d*nT:(d+1)*nT] = mabes.Example['RealStates']
        posta[d*nT:(d+1)*nT,:] = mabes.Example['PostActions']
        preci[d*nT:(d+1)*nT] = mabes.Example['Precision']
    stateV = np.zeros((nD*nT, mabes.Ns))
    thres = np.tile(mabes.thres, trial.shape)
    for n, st in enumerate(state):
        stateV[n, st] = 1
    return mabes, deci, trial, state, thres, posta, preci, stateV, nD, nT

def _shift_states(states, thres, reihe, multiplier = 1.2):
    """ 'convolves' the states from the data with the 8 action pairs."""
#    raise Exception
    for s, sta in enumerate(states):
        states[s] = (sta + multiplier*thres[s]*(reihe[s]-1)).astype(int)

def _calculate_likelihood(deci, post_act):
    r""" Calculates the likelihood of the data given the model in post_act."""

    def likelihood(data, dist):
        return dist[:,0]**((data==0)*1)*dist[:,1]**((data==1)*1)
    sizes_mu_sd = post_act.shape[0:2]
    likelihood_model = np.zeros((sizes_mu_sd[0], sizes_mu_sd[1]))
    for mu, pa_mu in enumerate(post_act):
        for sd, pa_mu_sd in enumerate(pa_mu):
            likelihood_model[mu, sd] = np.prod(likelihood(deci, pa_mu_sd))
    return likelihood_model

def _save_posteriors(post_act, trial, state, thres, mu_sigma):
    """ Save the posterior over actions given the observations and other data.
    It first loads the data file again and adds to it whatever is in post_act;
    then it saves the file again with the added data.

    The idea is that every time the infer_parameters() is ran, the posteriors
    are saved in a file, which can then be used in future runs to read the
    previously-obtained posteriors instead of having to calculate them again.

    The file contains a single dictionary, with one key for every observation
    in the data. The keys are tuples (Mu, Sd, Observation, Trial Number,
    Threshold). Those parameters fully determine what the Active Inference
    posteriors over actions are, so the tuple is a good identifier.
    """
    import pickle
    import os
    print('Saving posteriors to file...', end=' ', flush=True)
    # read data from file
#    in_file  = './data/posteriors.pi'
    out_file = './data/out_%d.pi' % os.getpid()
    k = 0
    while os.path.isfile(out_file): #find file that doesn't exist. Stupid HPC
        out_file = './data/out_%d_%d.pi' % (os.getpid(), k)
        k += 1


#    try:
#        with open(in_file, 'rb') as mafi:
#            data = pickle.load(mafi)
#    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
#        data = {}
    data = {}
    nMu, nSd, _ = mu_sigma.shape
    nSt = state.shape[0]
    for m in range(nMu):
        for s in range(nSd):
            for st in range(nSt):
                try: #In case the function was called by atexit with partial data
                    data[(mu_sigma[m, s, 0], mu_sigma[m, s, 1],
                          state[st], trial[st], thres[st])] = post_act[(m,s,st)]
                except KeyError:
                    pass
    with open(out_file, 'wb') as mafi:
        pickle.dump(data, mafi)
    print('Posteriors saved.')


def concatenate_data(data_folder=None, out_file=None, old_files=None):
    r""" Concatenates all the out**.pi files into one dictionary and saves it
    to the specified file.

    Parameters
    ----------
        data_folder: string, default='posteriors.pi'
            Folder where the data files (.pi) are.
        out_file: string, default='./data'
            File where previous data is found and where new data is saved.
        old_files: {'move', 'delete'}, default=None
            What to do with the individual out**.pi files: move them to a
            backup folder, delete them or leave them where they are if no
            value is passed.
    """
    import os
    from import_data import cd
    import pickle
    from tqdm import tqdm

    if data_folder is None:
        data_folder = './data/'
    if out_file is None:
        out_file = 'posteriors.pi'

    with cd(data_folder):
        files = [file for file in os.listdir('.') if file[:3] == 'out' and file[-2:] == 'pi']
        try:
            with open('posteriors.pi', 'rb') as mafi:
                data = pickle.load(mafi)
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            data = {}
        print('Joining dictionaries...', flush=True)
        for file in tqdm(files):
            with open(file, 'rb') as mafi:
                data.update(pickle.load(mafi))
        print('Saving data...', flush=True)
        with open(out_file, 'wb') as mafi:
            pickle.dump(data, mafi)
        if old_files=='delete':
            for file in files:
                os.remove(file)
        elif old_files=='move':
            with cd('./old_data'):
                pass
            for file in files:
                os.rename(file, './old_data/'+file)
def find_saved_posteriors(subject_data):
    """ Finds out which of the observations in the data have already been
    simulated (and for which parameter values).

    DOES NOT YET WORK
    """
    import pickle
    from tqdm import tqdm
    # read data from file
    data_file = './data/posteriors.pi'

    les_noms = ['mu', 'sd']
    with open(data_file, 'rb') as mafi:
        data = pickle.load(mafi)
    state = subject_data['obs']
    trial = subject_data['trial']
    thres = subject_data['threshold']

    nObs = state.size
    seen = {}
    print('Begin checking data...', flush=True)
    for o in tqdm(range(nObs)):
        seen[(o,)] = {'mu':set(), 'sd':set()}
        for makey in data.keys():
            if makey[2]==state[o] and makey[3]==trial[o] and makey[4]==thres[o]:
                seen[(o,)]['mu'].add(makey[0])
                seen[(o,)]['sd'].add(makey[1])



    # Find the values for mu and sd that are there for this subject
    print('Begin finding ranges...', flush = True)
    temp_set = {'mu':set(), 'sd':set()}
    for o in tqdm(range(1,nObs)):
        for name in les_noms:
            temp_set[name] = seen[(o,)][name].intersection(seen[(o-1,)][name])

    return temp_set, seen

def check_data_file(subjects = None, trials = None,
                    mu_range = None, sd_range = None,
                    data_file = None):
    """ Checks whether the observations in the data have been simulated.

    It can be limited to subjects, trials and parameter values.

    Limiting by subjects and trials is not yet implemented.

    Parameters
    ----------
    subjects: tuple, defaults to all
    trials: tuple, defaults to all
    mu_range: [min, max], defaults to [-15, 45]
        Range of values for the parameter mu. The interval is inclussive on
        both ends.
    sd_range: [min, max], defaults to [1, 15]
        Range of values for the parameter sd. The interval is inclussive on
        both ends.
    data_file: string, defaults to './data/posteriors.pi'
        Path of the file where the simulated data is.

    Returns
    -------
    was_found: bool
        Returns True if all subjects/trials/parameters were found in the data
        file. False otherwise.
    """
    import import_data as imda
    if not (isinstance(subjects, int) or len(subjects)==1 or subjects is None):
        raise NotImplementedError('Only single subject is implemented')
    if subjects is None:
        subjects = 0
        print('Zero-th subject chosen')
        
    _, datas = imda.main() #discard data, rename data_flat to data.
    datas = [datas[subjects],]
    for d, data in enumerate(datas):
        deci, trial, state, thres, reihe = (data['choice'], data['trial'],
                                        data['obs'], data['threshold'],
                                        data['reihe'])

        state = np.round(state/10).astype(int)
        thres = np.round(thres/10).astype(int)
        deci, trial, state, thres, reihe = _remove_above_thres(deci, trial, state, thres, reihe)
        _shift_states(state, thres, reihe)

        if data_file is None:
            data_file = './data/posteriors.pi'

        with open(data_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)

        if mu_range is None:
            min_mean = -15
            max_mean = 46
        else:
            min_mean = mu_range[0]
            max_mean = mu_range[1] + 1

        if sd_range is None:
            min_sigma = 1
            max_sigma = 15
        else:
            min_sigma = sd_range[0]
            max_sigma = sd_range[1] + 1

#        flag_stop = 0
        bad_set = set()
        for m, mu in enumerate(range(min_mean, max_mean)):
            for sd, sigma in enumerate(range(min_sigma,max_sigma)):
                for s in range(len(state)):
                    if (mu, sigma, state[s], trial[s], thres[s]) not in as_seen:
                        if (mu, sigma, state[s], trial[s], thres[s]) not in bad_set:
                            bad_set.add((mu, sigma, state[s], trial[s], thres[s]))
#                        print(d, mu, sigma, state[s], trial[s], thres[s], flush=True)

#                        return False
    return bad_set

def main(data_type, mu_range, sd_range, subject = 0, threshold = None,
         games = 20, trials = None, sim_mu = None, sim_sd = None,
         return_results = True):
    r""" main(data_type, subject, mu_range, sd_range [, games] [, trials]
              [, return_results])

    Runs the invert_parameters routine with the selected data, for the selected
    subjects and parameters.

    Parameters
    ----------
    data_type: {'full', 'small','pruned', 'simulated', 'threshold'}
        Determines the type of data to be used. 'full' uses all the games and
        trials. 'small' uses the selected number of games, with all trials.
        'pruned' uses the selected trials, all games. Combining 'full' or
        'small' with 'pruned' is possible. 'simulated' simulates data using
        active inference itself. Defaults to 'simulated'. 'threshold' runs
        the simulations only for those trials (games) with the given threshold;
        the optional parameter threshold (see below) must be provided.
    threshold: int
        Used with data_type = 'threshold', indicates which threshold (as
        indexed in the data) will be used from the data.
    subjects: int or tuple of ints
        Selects the subjects to use. Numbers bigger than the available number
        of subjects will be quietly ignored. Defaults to 0.
    games: int or tuple of ints
        Select which games to use with 'small' or 'simulated'. Defaults to all
        available games in the data, or 20 for 'simulated'.
    trials: int or tuple of ints
        Select which trials to use with 'pruned' or 'simulated'.
    mu_range: (int, int)
        Selects the range for the grid search on mu. It will search all
        integers in the range [int, int].
    sd_range: (int, int)
        Selects the range for the grid search on sd. It will search all
        integers in the range [int, int].
    return_results: bool
        Whether to return the results to the caller (True; default) or not.

    Returns
    -------
    (nM = number of models; nMu = number of values of mu; nSd = number of
    values of sd; nObs = total number of observations per subject)

    Note 1:
    All outputs are dictionaries with a key for every subject in the data.

    Note 2:
    In case of return_results = False, nothing is returned.

    post_model: np.array[nM]
        Posteriors for all models specified by mu_range and sd_range.
    posta: np.array[nMu, nSd, nObs, 2], dtype = float
        Posterior over actions for all the observations, all the models.
    deci: np.array[nObs], dtype = bool
        Decisions made by the subjects (or agent, if 'simulated') for every
        observation in the data.
    trial: np.array[nObs], dtype = int
        Trial number for all observations.
    state: np.array[nObs], dtype = int
        Observations in the data.
    mu_sigma: np.array[nMu, nSd, 2], dtype = int
        All values of Mu and Sd used for the models.
    """

    path_to_data = None#'/home/dario/Proj_ActiveInference/Svens Data/logfiles/'
    import import_data as imda
    data = imda.import_kolling_data(path_to_data)
    data = imda.clean_data(data)
    imda.enhance_data(data)
    data_flat = imda.flatten_data(data)
    imda.add_initial_obs(data_flat)

    # Work (or generate) the data:
    if 'simulated' in data_type:
        if 'threshold' in data_type: #simulate data with the given thres
            if 'small' in data_type:
                games = max(12, games)
            else:
                games = 12
            nS = np.ceil(1.2*threshold)
            mabes, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
                              simulate_data(num_games = 12, thres = threshold,
                                            nS = nS, mu = sim_mu, sd = sim_sd))

    else:
        if 'small' in data_type:
            data_flat = imda.small_data(data_flat, nGames = games) # TODO: select which games
        if 'pruned' in data_type:
            if trials is None:
                raise ValueError('No trials were selected for pruning.')
            data_flat = imda.prune_trials(data_flat, trials)
        if 'threshold' in data_type:
            if threshold is None:
                raise ValueError('A value for the input -threshold- must be' +
                                  ' provided with data_type = ''threshold'' ')
            for datum in data_flat:
                target_thres = datum['TargetLevels'][threshold]
                indices = datum['threshold']==target_thres
                for field in ['obs','choice','reihe', 'trial', 'threshold']:
                    datum[field] = datum[field][indices]

    # Run the inversion:
    if isinstance(subject, int):
        subject = subject,
    post_model = {}
    posta = {}
    deci = {}
    trial = {}
    state = {}
    mu_sigma = {}
    for s in subject:
        (post_model[s], posta[s], deci[s], trial[s], state[s], mu_sigma[s]) = (
                                  infer_parameters(mu_range = mu_range,
                                  sd_range=sd_range, data=data_flat[s]))
    if return_results is True:
        return post_model, posta, deci, trial, state, mu_sigma



#    return data, data_flat
if __name__ == '__main__':
    import argparse
    from os import getpid

    print('pid: %s' % getpid())

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mu', nargs=2,
                        help='(mu1, mu2) Interval for mu', type=int)
    parser.add_argument('-s', '--sd', nargs=2,
                        help='(sd1, sd2) Interval for SD', type=int)
    parser.add_argument('subjects', nargs='+',
                        help='Subject number. Can be more than one.', type=int)
    parser.add_argument('-t', '--trials', nargs='+',
                        help='List of trials to use', type=int)
    parser.add_argument('-v', '--verbose',
                        help='Verbose flag', action='store_true')
    parser.add_argument('-i', '--index',
                        help='A single index to select parameter values', type=int)
    args = parser.parse_args()

    if args.trials is None:
        trials = [0,1,2,3,4,5,6,7,]
    else:
        trials = args.trials

    if args.index is not None:
        mu_vals = np.arange(-15, 45+1) #make it inclussive
        sd_vals = np.arange(1, 15+1) #make it inclussive
        indices = np.unravel_index(args.index, (mu_vals.size, sd_vals.size))
        one_mu = mu_vals[indices[0]]
        one_sd = sd_vals[indices[1]]
        mu_range = (one_mu, one_mu)
        sd_range = (one_sd, one_sd)
    else:
        mu_range = args.mu
        sd_range = args.sd
    # Print message stating what is to be calculated
    if args.verbose:
        print('Subjects to use: %s' %args.subjects)
        print('Mu and Sd intervals: (%d, %d), (%d, %d):'
              % (mu_range[0], mu_range[1], sd_range[0], sd_range[1]))
        print('Trials to use: %s' % trials)
    main(data_type = ['full','pruned'], mu_range = mu_range,
         sd_range = sd_range, subject = args.subjects,
         trials = trials, return_results = False)
