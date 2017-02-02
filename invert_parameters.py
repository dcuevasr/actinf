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

def _remove_above_thres(deci, trial, state, thres, multiplier = 1.2):
    """Removes any observation that goes above the cutoff. """

    indices = state <= np.array(multiplier*thres, dtype=int)
    return deci[indices], trial[indices], state[indices], thres[indices]

def infer_parameters(mu_range = None, sd_range = None, num_games = None,
                     data = None, write_file = True):
    """ Calculates the likelihood for every model provided.

    Each model is created with different values of the mean and sd of a
    Gaussian distribution for the goals of active inference (log-priors). These
    models are used for generating posterior over actions for all the
    observations contained in Data. These posteriors are then used to calculate
    the likelihood for each model.

    If no data is provided, it is simulated with betClass.py.

    All posteriors are saved into a file for future use. This is independent of
    the write_file parameter (see description below); write_file controls
    whether the results of the current execution are saved to a standalone
    file which is pretty much useless at this point. Might deprecate later.

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
        integers in the range [int, int).
    sd_range: (int, int)
        Selects the range for the grid search on sd. It will search all
        integers in the range [int, int).
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
    write_file: bool
        Whether or not write everything to a file.


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
    flag_write_posta = True
    t_ini = time()


    if data is None:
        (mabes, deci, trial, state, thres, posta,
         preci, stateV, nD, nT) = simulate_data(num_games)
#        max_state = mabes.nS
#        Ns = mabes.Ns
        as_seen = {}
        flag_write_posta = False #Don't write to file if simulated data
    else:
        deci, trial, state, thres, reihe = data['choice'], data['trial'], data['obs'], data['threshold'], data['reihe']
        state = np.round(state/10).astype(int)
        thres = np.round(thres/10).astype(int)
        deci, trial, state, thres = _remove_above_thres(deci, trial, state, thres)
        target_levels = np.round(data['TargetLevels']/10)
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
    if sd_range is None:
        min_sigma = 10 # >0
        max_sigma = 20 # Maximum value for the variance
    else:
        min_sigma, max_sigma = sd_range
    mu_size = max_mean - min_mean
    sd_size = max_sigma - min_sigma
    tini = time()

    mabed = {}
    for lvl in target_levels: #one actinf for every threshold
        mabed[lvl] = bc.betMDP(nT = 9, nS = np.round(lvl*1.2).astype(int), thres = int(lvl))
    print('Calculating log-priors (goals)...', end='', flush = True)
    mu_sigma = np.zeros((mu_size, sd_size, 2))
#    arr_lnc = np.zeros((max_mean-min_mean, max_sigma-min_sigma, len(state), mabes.Ns))
    arr_lnc = {}
    for m, mu in enumerate(range(min_mean, max_mean)):
        for sd, sigma in enumerate(range(min_sigma,max_sigma)):
            for s in range(len(state)):
                mu_sigma[m, sd, :] = [mu, sigma]
                if (mu, sigma, state[s], trial[s], thres[s]) not in as_seen:
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
    #%% Writing data to files
    if write_file is True:
        explanation = ('1: likelihood for all models. 2: posteriors over actions for'+
                       ' all values of the parameters. 3: (data) observed states. 4: '+
                      ' (data) decisions at every trial. 5: (data) trial number for' +
                      ' each observation. 6: all values of (mu, sigma) used to build' +
                      ' the models')
        now_date = str(round(time()))
        filename = './data/model_inversion_sim_'+ now_date + '.npy'
        np.save(filename, [likelihood_model,
                post_act, deci, trial, state, mu_sigma])
        with open('model_inversion_sim_KEY.txt', "w") as expl_file:
            expl_file.write(explanation)

    print('The whole thing took: %d seconds.' % (time() - t_ini))
    smiley = np.random.choice([':)', 'XD', ':P', '8D'])
    print('Finished. Have a nice day %s\n' % smiley)
    return prob_data, post_act, deci, trial, state, mu_sigma

def simulate_posteriors(min_mean, max_mean, min_sigma, max_sigma, as_seen,
                        state, trial, thres, arr_lnc, mabed, wflag, mu_sigma):
    posta_inferred = {}
    if wflag is True:
        atexit.register(_save_posteriors, posta_inferred,
                        trial, state, thres, mu_sigma)
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
                               trial[s], mabed[thres[s]].V, mabed[thres[s]].D, 0, 15)[1:3]
                    assert posta_inferred[(m, sd, s)] != (), \
                              'posteriorOverStates returned empty for mu = %d, '\
                              'sd = %d and obs = %d, t = %d, thres = %d'\
                              % (m, sd, state[s], trial[s], thres[s])

    return posta_inferred
def call_mabe(le_input):
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

def simulate_data(num_games = 10, paradigm = None):
    """ Simulates data for the kolling experiment using active inference.

    Inputs:
        num_games: {1} Number of games to simulate (each with 8 trials or so).
        paradigm: {string} Paradigm to use (see betClass.py for more on this).
    Outputs: (nD = number of games; nT = number of trials per game; nO =
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


    mabes = bc.betMDP(nS = 100)
    mabes.gamma = 1
    mabes.thres = 90
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

    The idea is that every time the infer_parameters() is ran, the posteriors
    are saved in a file, which can then be used in future runs to read the
    previously-obtained posteriors instead of having to calculate them again.

    The file contains a single dictionary, with one key for every observation
    in the data. The keys are tuples (Mu, Sd, Observation, Trial Number,
    Threshold). Those parameters fully determine what the Active Inference
    posteriors over actions are, so the tuple is a good identifier.
    """
    import pickle
    print('Saving posteriors to file...', end=' ')
    # read data from file
    data_file = './data/posteriors.pi'
    try:
        with open(data_file, 'rb') as mafi:
            data = pickle.load(mafi)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
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
    with open(data_file, 'wb') as mafi:
        pickle.dump(data, mafi)
    print('Posteriors saved.')
def _check_data(): # TODO: delete
    """ No idea what that is and it's never used... Probably a test that is no
    longer necessary.
    """

    import pickle
    import numpy as np
    # read data from file
    data_file = './data/posteriors.pi'
    try:
        with open(data_file, 'rb') as mafi:
            data = pickle.load(mafi)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        data = {}
    post = np.zeros((len(data.keys()), 2))
    for k, keys in enumerate(data.keys()):
        post[k, :] = data[keys][1]
    return post

def main(data_type, mu_range, sd_range, subject = 0,
         games = 20, trials = None, return_results = True):
    r""" main(data_type, subject, mu_range, sd_range [, games] [, trials]
              [, return_results])

    Runs the invert_parameters routine with the selected data, for the selected
    subjects and parameters.

    Parameters
    ----------
    data_type: {'full', 'small','pruned', 'simulated'}
        Determines the type of data to be used. 'full' uses all the games and
        trials. 'small' uses the selected number of games, with all trials.
        'pruned' uses the selected trials, all games. Combining 'full' or
        'small' with 'pruned' is possible. 'simulated' simulates data using
        active inference itself. Defaults to 'simulated'.
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
        integers in the range [int, int).
    sd_range: (int, int)
        Selects the range for the grid search on sd. It will search all
        integers in the range [int, int).
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

    path_to_data = None
    import import_data as imda
    data = imda.import_kolling_data(path_to_data)
    data = imda.clean_data(data)
    imda.enhance_data(data)
    data_flat = imda.flatten_data(data)
    imda.add_initial_obs(data_flat)
    if data_type == 'simulated':
        likeli, posta, deci, trial, state, mu_sigma = (
                           infer_parameters(mu_range = mu_range,
                           sd_range = sd_range,
                           data = None, num_games = games))
    if 'small' in data_type:
        data_flat = imda.small_data(data_flat, nGames = games) # TODO: select which games
    if 'pruned' in data_type:
        if trials is None:
            raise ValueError('No trials were selected for pruning.')
        data_flat = imda.prune_trials(data_flat, trials)
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
    main(data_type = ['full','pruned'], mu_range = (-15, 40), sd_range = (2,13),
         subject = (0,), trials = [0,1], return_results = False)