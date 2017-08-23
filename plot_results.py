#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bunch of mostly-disconnected functions to plot and visualize results.

Created on Tue Mar 21 09:09:19 2017

@author: dario
"""
import pickle
import itertools
import ipdb
from tqdm import tqdm
import os

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gs

import invert_parameters as invp
import import_data as imda
from utils import calc_subplots
import betClass as bc


def calc_risk_pressure(data, rmv_after_thres=True):
    """ Calculates the risk pressure (from Kolling_2014) for every point in
    the supplied data.

    Parameters
    ----------
    data: dict
        Contains the data points to be used. It is assumed that the first index
        is the subject number; if only one subject is included, it should still
        be supplied in a dict, such that data[0] has the data for the one
        subject. For example, data[0]['obs'] (as oppossed to data['obs']). Note
        that this is assumed to be data_flat, from import_data.main().

    Returns
    -------
    risk_p: list, double
        Contains the risk pressure for every data point in data, in the same
        format.

    """
    from invert_parameters import _remove_above_thres as rat

    risk_p = {}
    for d, datum in enumerate(data):
        state = datum['obs']
        thres = datum['threshold']
        reihe = datum['reihe']
        state = np.round(state / 10).astype(int)
        thres = np.round(thres / 10).astype(int)
        if rmv_after_thres:
            _, trial, obs, thres, reihe, _ = rat(datum['choice'], datum['trial'],
                                                 state, thres, reihe)
        else:
            obs = state
            trial = datum['trial']
        risk_p[d] = (thres - obs) / (8 - trial)
        risk_p[d][risk_p[d] < 0] = 0

    return risk_p


def plot_rp_pr(posta, mu, sd, fignum=2):
    """ Plots risk pressure (rp) vs. probability of chosing risky (pr).

    Parameters
    ----------
    posta: dict
        Contains the posteriors over actions for every subject and observation:
        posta[subjectNr][obsNr][1] should be the probability of the risky
        option for the given subject and observation. If only one subject,
        posta should still be a dict, with posta[0] as the data for the only
        subject.
    fignum: int or None
        If int, uses the number to create a new figure. If None, tries to plot
        in the current axes.
    """

    _, data = imda.main()

    rp = calc_risk_pressure(data)

    num_subs = len(posta)
    a1, a2 = calc_subplots(num_subs)

    if fignum is not None:
        plt.figure(fignum)
        plt.clf
    for s in range(num_subs):
        assert posta[s][mu, sd, :][:, 1].shape == rp[s].shape
        plt.plot(rp[s], posta[s][mu, sd, :][:, 1], '.')
    if fignum is not None:
        plt.title(r'Actinf''s p(risky) for $\mu = %d, \sigma = %d$' %
                  (np.arange(-15, 45)[mu], np.arange(1, 15)[sd]))
        plt.xlabel('Risk pressure (as per Kolling_2014)')
        plt.ylabel('p(risky) for ActInf')
    else:
        plt.title(r'$\mu = %d, \sigma = %d$' % (np.arange(-15, 45)[mu],
                                                np.arange(1, 15)[sd]))
        plt.xticks([], [])


def plot_all_pr(posta, mu_values, sd_values, fignum=3):
    """ Creates an array of plots for all mu_values and sd_values (one-to-one,
    not all combinations), using plot_rp_pr().
    """
    assert len(mu_values) == len(sd_values), ("mu_values and sd_values do " +
                                              "not have the same number of " +
                                              "elements")
    a1, a2 = calc_subplots(len(mu_values))
    plt.figure(fignum)
    plt.clf
    for i in range(len(mu_values)):
        plt.subplot(a1, a2, i + 1)
        plot_rp_pr(posta, mu_values[i], sd_values[i], None)

    plt.suptitle(r'Actinf''s posterior probability of risky for different' +
                 ' values of $\mu$ and $\sigma$')


def plot_by_thres(shape_pars=None, subjects=None, trials=None, fignum=4,
                  data_flat=None, priors=None, as_seen=None):
    """ Calculate and plot the likeli for the given subjects, separating the
    different thresholds into sub-subplots.

    The function can be called by specifying subjects and trial numbers to use
    or it can be given data as input. Giving data as input overrides the other
    parameters.

    Parameters
    ----------
    subjects: tuple of ints, or int
        Which subjects to include. The order is arbitrary.
    trials: tuple of ints, or int
        Which trials to use.
    fignum: int
        Figure number to plot to
    data_flat: dict
        Output of import_data.main() or similar. If none supplied, it will be
        taken from import_data.main().
    priors: dict of numpy arrays. Defaults to 1s.
        Prior distribution over model parameters. It should contain a key for
        each threshold level (usually 4). Each value should be a numpy array
        the size of the parameter space (60x15, usually).

    Returns
    -------
    likeli_out: dict
        Dictionary with the loglikelihoods for all (subject, threshold) pairs.
    """
    import matplotlib.gridspec as gridspec

    if shape_pars is None:
        shape_pars = invp.__rds__('unimodal_s')

    if data_flat is None:
        _, data_flat = imda.main()
    if subjects is None:
        subjects = range(len(data_flat))
    elif isinstance(subjects, int):
        subjects = subjects,

    if trials is None:
        trials = [0, 1, 2, 3, 4, 5, 6, 7]
    num_subs = len(subjects)
    nTL = len(data_flat[0]['TargetLevels'])
    if priors is None:
        priors = np.ones(4)  # TODO: what is this number?
    tl1, tl2 = calc_subplots(nTL)
    s2, s1 = calc_subplots(num_subs)
    fig = plt.figure(fignum)
    fig.clf()
    outer_grid = gridspec.GridSpec(s1, s2)
    if as_seen is None:
        data_file = './data/posteriors.pi'
        with open(data_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)
            print('File opened; data loaded.')
    plt.set_cmap('gray_r')
    if len(shape_pars) == 3:
        xlen = shape_pars[2].size
        xvec = shape_pars[2]
    elif len(shape_pars) == 2:
        xlen = 1
        xvec = [1]
    elif len(shape_pars) > 3 or len(shape_pars) < 1:
        raise NotImplementedError('Grob confused. What shape is this?')
    xticks = [0, int(xlen / 2), xlen]
    xticklabels = [xvec[0], xvec[xticks[1]], xvec[-1]]

    ylen = shape_pars[1].size
    yvec = shape_pars[1]
    yticks = [0, int(ylen / 2), ylen]
    yticklabels = [yvec[0], yvec[yticks[1]], yvec[-1]]
    likeli_out = {}
    for s in range(num_subs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1,
                                                      subplot_spec=outer_grid[s], hspace=0.0,
                                                      wspace=0)

        vTL = np.arange(4)[data_flat[s]['TargetLevels'] != 0]

        for th in vTL:
            ax = plt.Subplot(fig, inner_grid[th])
            likeli, _, _, _, _, _, _ = invp.main(data_type=['threshold', 'pruned'],
                                                 threshold=th, shape_pars=shape_pars,
                                                 subject=subjects[s],
                                                 data_flat=data_flat,
                                                 trials=trials, as_seen=as_seen,
                                                 return_results=True, normalize=False)
            likeli_out[(subjects[s], th)] = likeli[subjects[s]]
            if xlen == 1:
                for key in likeli.keys():
                    likeli[key] = np.expand_dims(likeli[key], 1)
            ax.imshow(np.exp(likeli[subjects[s]]) * priors[th], aspect=0.25,
                      interpolation='none')

            ax.tick_params(width=0.01)

            if ax.is_last_row() and ax.is_first_col():
                ax.set_xticks(xticks)
                ax.set_xlabel(r'$\sigma$')
                ax.set_xticklabels(xticklabels, fontsize=8)
                ax.set_yticks(yticks)
                ax.set_ylabel(r'$\mu$', fontsize=6)
                ax.set_yticklabels(yticklabels, fontsize=6)
            else:
                ax.set_yticks([])
                ax.set_xticks([])
            if ax.is_first_row():
                ax.set_title('Subject: %d' % (s), fontsize=6)
            else:
                ax.set_title('')

            fig.add_subplot(ax)
    plt.tight_layout()
    plt.show()
    return likeli_out


def select_by_rp(rp_lims, trials=None):
    """ Will select the obs that meet the requirement of having a so-and-so
    risk pressure.
    """
    rp_min, rp_max = rp_lims
    _, data_flat = imda.main()

    rp = calc_risk_pressure(data_flat, False)
    indices = {}
    for i in rp:
        tmp1 = rp[i] >= rp_min
        tmp2 = rp[i] <= rp_max
        indices[i] = np.logical_and(tmp1, tmp2)
    for d, datum in enumerate(data_flat):
        for field in ['threshold', 'trial', 'obs', 'reihe', 'choice']:
            datum[field] = datum[field][indices[d]]

    plot_by_thres(trials=trials, data_flat=data_flat)


def select_by_doability(rp_lims, trials, data_flat=None, fignum=5,
                        as_seen=None):
    """ Will select the obs that meet the requirement of having a so-and-so
    risk pressure. These trials will be sent to plot_by_thres for plotting.

    Parameters
    rp_lims: list of duples of floats
        Each element of the list, with its corresponding element of trials,
        determines the range of rp to be used.
    trials: list of ints
        See above.
    """
    if data_flat is None:
        _, data_flat = imda.main()
    if isinstance(rp_lims, tuple):
        rp_lims = [rp_lims, ]
    if isinstance(trials, int):
        trials = trials,

    rp_min = {}
    rp_max = {}
    for r, rl in enumerate(rp_lims):
        rp_min[r] = rl[0]
        rp_max[r] = rl[1]

    nC = len(rp_lims)
    if len(trials) != nC:
        raise TypeError('The two inputs must have the same length')

    # Keep only those observations where the RP is within the given limits
    rp = calc_risk_pressure(data_flat, False)
    indices = {}
    for s in range(len(data_flat)):
        indices[s] = [0] * len(data_flat[s]['trial'])
    for c in range(nC):
        for i in rp:  # one i for every subject in data_flat
            tmp1 = rp[i] >= rp_min[c]
            tmp2 = rp[i] <= rp_max[c]
            indices_rp = np.logical_and(tmp1, tmp2)
            indices_tr = data_flat[i]['trial'] == trials[c]
            indices_mx = np.logical_and(indices_rp, indices_tr)
            indices[i] = np.logical_or(indices[i], indices_mx)
    for d, datum in enumerate(data_flat):
        for field in ['threshold', 'trial', 'obs', 'reihe', 'choice']:
            datum[field] = datum[field][indices[d]]

    # For every subject, check whether there are no observations left for a
    # given threshold. This is needed for plot_by_thresholds():
    for datum in data_flat:
        for tl in range(4):  # target levels
            if datum['TargetLevels'][tl] not in datum['threshold']:
                datum['TargetLevels'][tl] = 0

    plot_by_thres(trials=trials, data_flat=data_flat, fignum=fignum,
                  as_seen=as_seen)


def prepare_inputs_doability(trials_left=1, as_seen=None):
    """ Little 'script' to prepare inputs for select_by_doability."""
    for c in range(1, trials_left + 1):
        rp_lims = []
        trials = range(7, 7 - c, -1)
        for t in trials:
            rp_lims.append((0.8 * c * 14, 1.2 * c * 20))
        select_by_doability(rp_lims, trials, fignum=c, as_seen=as_seen)


def concatenate_subjects():
    """ Concatenates the data from one subject into a single-subject-like
    structure to do empirical priors.

    TODO:
    NOTE: Doing this makes the likelihoods so small that the precision is not
    and everything goes to zero. To use this, I would need to fix that problem.
    """
    _, data_flat = imda.main()

    fields = set(data_flat[0].keys())
    fields.remove('TargetLevels')
    fields.remove('NumberGames')
    fields.remove('NumberTrials')

    data_out = {}
    for field in fields:
        cField = np.array([], dtype=np.int64)
        for datum in data_flat:
            cField = np.concatenate([cField, datum[field]])
        data_out[field] = cField

    for field in ['TargetLevels', 'NumberGames', 'NumberTrials']:
        data_out[field] = data_flat[0][field]
    return [data_out, ]


def concatenate_smart(plot=True, retorno=False, fignum=6):
    """ Will get the likelihood for the whole set of subjects in one go, while
    sidestepping the problem of numbers getting too small and getting lost in
    precision.
    """

    """
    Idea:
        1.- Get likelihoods for each subject-condition combination.
        2.- On a per-condition basis, transform to logarithms and add.
        3.- Normalize
        4.- Profit
    """

    with open('./data/posteriors.pi', 'rb') as mafi:
        as_seen = pickle.load(mafi)

    logli = {}
    for th in range(4):  # Target values
        likeli, _, _, _, _, _ = invp.main(data_type=['threshold'],
                                          threshold=th, mu_range=(-15, 45),
                                          subject=range(35),
                                          sd_range=(1, 15), as_seen=as_seen,
                                          return_results=True)
        logli[th] = 0
        for s in range(len(likeli)):
            logli[th] += np.log(likeli[s])

    if plot:
        plt.figure(fignum)
        for th in range(4):
            plt.subplot(2, 2, th + 1)
            plt.imshow(logli[th])
            plt.set_cmap('gray_r')
    if retorno:
        return logli


def plot_concatenated_linear(fignum=7):
    """ Gets loglikelihood from concatenate_smart, applies exponential and
    plots.
    """

    logli = concatenate_smart(plot=False, retorno=True)
    likeli = {}
    for key in logli.keys():
        likeli[key] = np.exp(logli[key] - logli[key].max())

    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        ax.imshow(likeli[i], aspect=0.2, interpolation='none')
        ax.tick_params(width=0.01)
        ax.set_title('Threshold: %d' % [595, 930, 1035, 1105][i])
    plt.suptitle(r'Likelihood map over all subjects')


def plot_animation_trials(subject=0, nDT=2, fignum=8, data_flat=None,
                          nA=3, as_seen=None):
    """ Creates a number of plots for a single subject with different trial
    numbers being taken into account.

    Parameters
    ----------
    subject: int
        Subject number to use.
    nA: int
        Number of trials for each plot. For example, if nDT = 3, then the
        first plot will use trials 1,2,3.
    nDT: int
        Number of plots to make.
    """
    import matplotlib.gridspec as gridspec

    if as_seen is None:
        data_file = './data/posteriors.pi'
        with open(data_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)
            print('File opened; data loaded.')

    target_levels = [595, 930, 1035, 1105]
    fig = plt.figure(fignum)
    fig.clf()
    nTL = 4  # number of Target Levels
    all_trials = range(8)
    t2, t1 = calc_subplots(nDT)
    outer_grid = gridspec.GridSpec(t1, t2)
    for i in range(nDT):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1,
                                                      subplot_spec=outer_grid[i],
                                                      wspace=0.0, hspace=0.0)
        trials = all_trials[i:(i + nA)]
        for th in range(nTL):
            shape_pars = ['unimodal_s',
                          np.arange(-15, 45 + 1), np.arange(1, 15 + 1)]
            loglikeli, _, _, _, _, _ = invp.main(data_type=['threshold', 'pruned'],
                                                 threshold=th, shape_pars=shape_pars,
                                                 subject=subject,
                                                 trials=trials, as_seen=as_seen,
                                                 data_flat=data_flat,
                                                 return_results=True)
            ax = plt.Subplot(fig, inner_grid[th])
            ax.imshow(np.exp(loglikeli[subject]),
                      aspect=0.2, interpolation='none')
            plt.set_cmap('gray_r')
            ax.tick_params(width=0.01)
            if ax.is_first_row():
                ax.set_title('Trials = %s, Thres = %d' % (list(trials), target_levels[th]),
                             fontsize=6)
            fig.add_subplot(ax)
    all_axes = fig.get_axes()

    for ax in all_axes:
        if ax.is_last_row():
            ax.set_xticks([1, 5, 10])
            ax.set_xticklabels([1, 5, 10], fontsize=8)
        else:
            ax.set_xticks([])
        if ax.is_first_col() and ax.is_first_row():
            ax.set_yticks([0, 15, 30, 45, 60])
            ax.set_yticklabels([-15, 0, 15, 30, 45], fontsize=10)
        else:
            ax.set_yticks([])

    plt.show()


def plot_without_past_threshold(nsubs=35, fignum=9, plot=True,
                                data_flat=None):
    """ Remove the trials past-threshold and plot."""

    if data_flat is None:
        _, data_flat = imda.main()

    for datum in data_flat:
        indices = np.zeros(len(datum['obs']), dtype=bool)
        for t, th in enumerate(datum['threshold']):
            if datum['obs'][t] < th:
                indices[t] = True
        for field in ['threshold', 'trial', 'obs', 'reihe', 'choice']:
            datum[field] = datum[field][indices]
    if plot:
        plot_by_thres(fignum=fignum, data_flat=data_flat[:nsubs])
    else:
        return data_flat


def per_mini_block():
    """ Obtains the likelihood for all models on a per-mini-block basis and
    ranks the models based on this.

    For every mini-block, all models aver evaluated and a number of points is
    assigned to each depending on the ranking. At the end, these points are
    added to select the best model.
    """

    data, _ = imda.main()

    for mb in range(48):
        data_short = return_miniblock(data, mb)
        data_flat = imda.flatten_data(data_short)
        imda.add_initial_obs(data_flat)


def return_miniblock(data_dnt, mb):
    """ Returns the data with just the requested mini-block.
    """
    import copy
    data = copy.deepcopy(data_dnt)
    fields = ['choice', 'obs', 'points', 'rechts', 'reihe', 'response',
              'target', 'threshold', 'trial']
    for datum in data:
        for field in fields:
            datum[field] = datum[field][mb]
    return data


def dumb_agent(thres=60):
    """ Play the game with an agent that chooses randomly."""

    import betClass as bc
    PreUpd = False
    printTime = False
    mabe = bc.betMDP(nS=np.round(thres * 1.2).astype(int), thres=thres)

    from time import time

    t1 = time()
    # Check that the class has been fully initiated with a task:
    if hasattr(mabe, 'lnA') is False:
        raise Exception('NotInitiated: The class has not been initiated' +
                        ' with a task')
    T = mabe.V.shape[1]
    wV = mabe.V   # Working set of policies. This will change after choice
    obs = np.zeros(T, dtype=int)    # Observations at time t
    act = np.zeros(T, dtype=int)    # Actions at time t
    sta = np.zeros(T, dtype=int)    # Real states at time t
    bel = np.zeros((T, mabe.Ns))      # Inferred states at time t
    P = np.zeros((T, mabe.Nu))
    W = np.zeros(T)
    # Matrices for diagnostic purposes. If deleted, also deleted the
    # corresponding ones in posteriorOverStates:
    mabe.oQ = []
    mabe.oV = []
    mabe.oH = []

    sta[0] = np.nonzero(mabe.S)[0][0]
    # Some dummy initial values:
#    PosteriorLastState = mabe.D
#    PastAction = 1
#    PriorPrecision = mabe.gamma
    Pupd = []
    for t in range(T - 1):
        # Sample an observation from current state
        obs[t] = mabe.sampleNextObservation(sta[t])
        # Update beliefs over current state and posterior over actions
        # and precision
#        bel[t,:], P[t,:], Gamma = mabe.posteriorOverStates(obs[t], t, wV,
#                                                PosteriorLastState,
#                                                PastAction,
#                                                PriorPrecision,
#                                                PreUpd = PreUpd)
        bel[t, :] = sta[t]
        P[t, :] = [0.5, 0.5]
        Gamma = mabe.gamma

        if PreUpd is True:
            W[t] = Gamma[-1]
            Pupd.append(Gamma)
        else:
            W[t] = Gamma
        # Sample an action and the next state using posteriors
        act[t], sta[t + 1] = mabe.sampleNextState(sta[t], P[t, :])
        # Remove from pool all policies that don't have the selected action
        tempV = []
        for seq in wV:
            if seq[t] == act[t]:
                tempV.append(seq)
        wV = np.array(tempV, dtype=int)
        # Prepare inputs for next iteration
#        PosteriorLastState = bel[t]
#        PastAction = act[t]
#        PriorPrecision = W[t]
    xt = time() - t1
    mabe.Example = {'Obs': obs, 'RealStates': sta, 'InfStates': bel,
                    'Precision': W, 'PostActions': P, 'Actions': act,
                    'xt': xt}
    if PreUpd is True:
        mabe.Example['PrecisionUpdates'] = np.array(Pupd)
    if printTime is True:
        pass
    return mabe


def calculate_best_possible_fit(mu, sd, subjects=[0, ], as_seen=None,
                                data_flat=None):
    """ For a given set of parameters, calculates the absolute best fit that
    the data could possibly provide, by assuming a subject that always chooses
    the option with the highest probability according to the model; the
    likelihood is calculated for this model with this agent.
    """

    if as_seen is None:
        with open('./data/posteriors.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)
    if data_flat is None:
        _, data_flat = imda.main()

    TL = data_flat[0]['TargetLevels']

    mabe = {}
    for t, th in enumerate(TL):
        thres = np.round(th / 10).astype(int)
        mabe[t] = bc.betMDP(nS=np.ceil(thres * 1.2).astype(int), thres=thres)
        mabe[th] = mabe[t]

    posta_inferred = {}
    max_likelihood = {}

    for s in subjects:
        data = data_flat[s]
        deci, trial, state, thres, reihe = (data['choice'], data['trial'],
                                            data['obs'], data['threshold'],
                                            data['reihe'])
        state = np.round(state / 10).astype(int)
        thres = np.round(thres / 10).astype(int)
        deci, trial, state, thres, reihe, _ = invp._remove_above_thres(deci, trial,
                                                                       state, thres, reihe)
        invp._shift_states(state, thres, reihe)
        max_likelihood[s] = 1
        for o in range(len(deci)):
            posta_inferred[(s, o)] = as_seen[(
                mu, sd, state[o], trial[o], thres[o])]
            max_likelihood[s] *= max(posta_inferred[(s, o)][0])

    return posta_inferred, max_likelihood,


def calculate_enhanced_likelihoods(subjects=[0, ], as_seen=None):
    """ Will calculate 'enhanced likelihoods', using the best possible fit
    calculated in calculate_best_possible_fit().

    BAD MATH

    """

    if as_seen is None:
        with open('./data/posteriors.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)
    if isinstance(subjects, int):
        subjects = subjects,

    mu_range = (-15, 45)
    mu_vec = np.arange(mu_range[0], mu_range[1] + 1)
    sd_range = (1, 15)
    sd_vec = np.arange(sd_range[0], sd_range[1] + 1)

    max_likelihood = {}
    for mu in mu_vec:
        for sd in sd_vec:
            _, ml = calculate_best_possible_fit(mu, sd, subjects,
                                                as_seen=as_seen)
            max_likelihood[(mu, sd)] = ml

    likeli, _, _, _, _, _ = invp.main(data_type=['full'],
                                      mu_range=mu_range, sd_range=sd_range, subject=subjects,
                                      as_seen=as_seen, return_results=True,
                                      normalize=False)
    corrected_likelihood = {}
    for s in range(len(likeli)):
        corrected_likelihood[s] = np.zeros((61, 16))
        for m in range(61):
            for d in range(15):
                corrected_likelihood[s][m, d] = likeli[s][m,
                                                          d] * max_likelihood[(mu_vec[m], sd_vec[d])][s]
    return corrected_likelihood


def create_sim_data(shape_pars):
    """ Creates simulated data with four conditions."""

    target_levels = np.array([595, 930, 1035, 1105])
    target_lvls = np.round(target_levels / 10).astype(int)

    tmp_data = {}
    for tg in target_lvls:
        mabes, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
            invp.simulate_data(shape_pars,
                               num_games=12,
                               nS=np.round(1.2 * tg).astype(int),
                               thres=tg))
        tmp_data[tg] = invp._create_data_flat(
            mabes, deci, trial, state, thres, nD, nT)

    data_flat = tmp_data[target_lvls[0]]
    for tg in target_lvls[1:]:
        for name in tmp_data[tg][0].keys():
            data_flat[0][name] = np.hstack(
                [data_flat[0][name], tmp_data[tg][0][name]])
    data_flat[0]['NumberGames'] = 48
    data_flat[0]['NumberTrials'] = 8
    return data_flat


def fit_simulated_data(sd_range=(3, 10), mu_range=(-15, 10), threshold=55,
                       sim_mu=55, sim_sd=5, fignum=10, games=20,
                       retorno=False, data_flat=None, as_seen={}):
    """ Fit the model to simulated data to see if the parameters are recovered"""

    if data_flat is None:
        likeli, _, _, _, _, _ = invp.main(
            data_type=['simulated', 'threshold'], sim_mu=55, sim_sd=5,
            mu_range=mu_range, sd_range=sd_range, trials=[
                0, 1, 2, 3, 4, 5, 6, 7],
            as_seen=as_seen, return_results=True, threshold=threshold,
            games=games, normalize=False)
    else:
        likeli, _, _, _, _, _ = invp.main(data_type=['full'],
                                          mu_range=mu_range, sd_range=sd_range, trials=[
                                              0, 1, 2, 3, 4, 5, 6, 7],
                                          as_seen=as_seen, return_results=True, data_flat=data_flat,
                                          normalize=False)


#    sd_range = (3,10)
    sd_vec = np.arange(sd_range[0], sd_range[1] + 1)
#    mu_range = (-15,10)
    mu_vec = np.arange(mu_range[0], mu_range[1] + 1)

    y0 = (mu_vec == 0).nonzero()[0][0]
    plt.figure(fignum)
    max_logli = likeli[0].max()
    plt.imshow(np.exp(likeli[0] - max_logli), interpolation='none')
    plt.set_cmap('gray_r')
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'threshold $+\mu$')
    plt.xticks([0, len(sd_vec)], [sd_vec[0], sd_vec[-1]])
    plt.yticks([0, y0, len(mu_vec) - 1], [mu_vec[0], mu_vec[y0], mu_vec[-1]])
    plt.title('Real parameters: \n mu = %d, sd = %d' % (sim_mu, sim_sd))
    if retorno:
        return likeli


def fit_many(thres=55, sim_mu=[0, 15, -15], sim_sd=[3, 4, 5], games=48,
             retorno=False, data_flat=None, as_seen=False,
             sd_range=(3, 5), mu_range=(-16, 16)):
    """ Plot many of fit_simulated_data()."""

    if as_seen is False:
        as_seen = {}
    else:
        print('Opening data file')
        try:
            with open('./data/posteriors.pi', 'rb') as mafi:
                as_seen = pickle.load(mafi)
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            as_seen = {}
            print('\nFile not found or contains invalid data')

    k = 100
    likeli = {}
    for mu in sim_mu:
        for sd in sim_sd:
            likeli[(mu, sd)] = fit_simulated_data(sd_range=sd_range,
                                                  mu_range=mu_range, data_flat=data_flat,
                                                  threshold=thres, sim_mu=mu, sim_sd=sd,
                                                  fignum=k, games=games, retorno=retorno,
                                                  as_seen=as_seen)
            k += 1
    invp.concatenate_data(old_files='delete')
    if retorno:
        return likeli


def plot_final_states(data, fignum=11):
    """ Bar plot of how many times states were visited in the data."""

    from matplotlib import gridspec

    if not isinstance(data, list):
        data = [data]
    num_subs = len(data)

    nObs = len(data[0]['trial'])
    target_levels = data[0]['TargetLevels']

    max_S = 0
    for datum in data:
        max_S = max(max_S, datum['obs'].max())

    count = {}
    for tl in target_levels:
        count[tl] = np.zeros(max_S + 1)

    for datum in data:
        for ob in range(nObs):
            cobs = datum['obs'][ob]
            cthr = datum['threshold'][ob]
            count[cthr][cobs] += 1
    fig = plt.figure(fignum)
    plt.clf()
    s2, s1 = utils.calc_subplots(num_subs)
    outer_grid = gridspec.GridSpec(s1, s2)

    for s in range(num_subs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1,
                                                      subplot_spec=outer_grid[s], hspace=0.0,
                                                      wspace=0)
        for th in range(len(target_levels)):
            ax = plt.Subplot(fig, inner_grid[th])
            thres = target_levels[th]

    #        ax.bar(np.arange(max_S+1),count[target_levels[th]])
            ax.hist(data[s]['obs'][data[s]['threshold'] == thres],
                    20, color=(105 / 256, 141 / 256, 198 / 256))
            ax.plot([thres, thres], [0, 20], linewidth=3, color='g')
            ax.plot([thres - 150, thres - 150], [0, 20], color='g')
            ax.plot([thres + 150, thres + 150], [0, 20], color='g')

            xticks = range(0, 1400, 100)
            xticklabels = range(0, 140, 10)

            yticks = range(0, 20, 5)
            yticklabels = yticks

            ax.set_xlim([0, 1400])

            if ax.is_last_row() and ax.is_first_col():
                ax.set_xticks(xticks)
                ax.set_xlabel(r'Points')
                ax.set_xticklabels(xticklabels, fontsize=8)
                ax.set_yticks(yticks)
                ax.set_ylabel(r'Count', fontsize=6)
                ax.set_yticklabels(yticklabels, fontsize=6)
                ax.tick_params(width=0.0)

            else:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_ylabel('Th=%d' % (thres / 10))
            if ax.is_first_row():
                ax.set_title('Histogram of visited states in data')
            else:
                ax.set_title('')

            fig.add_subplot(ax)


def plot_evolution_gamma(mu=None, sd=None, num_games=10,
                         fignum=12, return_dict=False):
    """ Simulates the data set's observations with different parameter values
    to see whether precision evolves significantly differently for different
    priors.

    """

    mabe, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
        invp.simulate_data(nS=72, thres=60, mu=mu, sd=sd))
#    data_flat = invp._create_data_flat(mabe, deci, trial, state, thres, nD, nT)
#
#    for field in ['choice', 'reihe','obs','threshold','trial']:
#        data_flat[0][field] = np.reshape(data_flat[0][field],(-1,nT))
#    data = data_flat # Since it is not flat anymore

    deci = np.reshape(deci, (-1, nT))
    state = np.reshape(state, (-1, nT))

    mu_vec = np.arange(-15, 30 + 1)
    sd_vec = np.arange(1, 15 + 1)
    mu_size = len(mu_vec)
    sd_size = len(sd_vec)
    try:
        nGames = data_flat[0]['choice'].shape[0]
    except:
        nGames = deci.shape[0]

    results = {}
    fig = plt.figure(fignum)
    for m, mu in enumerate(mu_vec):
        for sd, sigma in enumerate(sd_vec):
            for g in range(nGames):
                shape_pars = [mu, sigma]
                mabe.set_prior_goals(select_shape='unimodal_s',
                                     shape_pars=shape_pars)
                results[(mu, sigma, g)] = mabe.full_inference(sta_f=state[g],
                                                              act_f=deci[g], just_return=True)
                plt.subplot(mu_size, sd_size, m * sd_size + sd + 1)
                plt.plot(results[(mu, sigma, g)]['Precision'])

    all_axes = fig.get_axes()
    for x, ax in enumerate(all_axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if ax.is_first_row():
            ax.set_title('%s' % sd_vec[x])
        if ax.is_first_col():
            ax.set_ylabel('%s' % mu_vec[ax.rowNum], fontsize=8)

    if return_dict is True:
        return results


def plot_lnc(mu_vec=None, sd_vec=None, fignum=13):
    """ Plots an array of exp(lnC) for the values of mu and sd provided."""

    from matplotlib import gridspec as gs

    target_levels = [60, 93, 104, 110]
    mabes = {}
    for tl in target_levels:
        mabes[tl] = bc.betMDP(nS=np.round(tl * 1.2).astype(int), thres=tl)
#    mabes = bc.betMDP()
    fig = plt.figure(fignum)
    fig.clf()
    if mu_vec is None:
        mu_vec = np.arange(-15, 45 + 1)
    if sd_vec is None:
        sd_vec = np.arange(1, 15 + 1)

    outer_grid = gs.GridSpec(len(mu_vec), len(sd_vec))
    for m, mu in enumerate(mu_vec):
        for sd, sigma in enumerate(sd_vec):
            for l, tl in enumerate(target_levels):
                inner_grid = gs.GridSpecFromSubplotSpec(4, 1,
                                                        subplot_spec=outer_grid[m, sd],
                                                        hspace=0, wspace=0)
                shape_pars = [mu, sigma]
                clnC = mabes[tl].set_prior_goals(select_shape='unimodal_s',
                                                 shape_pars=shape_pars,
                                                 just_return=True,
                                                 cutoff=False, convolute=False)
                ax = plt.Subplot(fig, inner_grid[l])
                ax.fill_between([0, mabes[tl].nS], [0, 0], [
                                clnC.max(), clnC.max()], color=(0.9, 0.9, 0.9))

                ax.plot(clnC, color=(0.2, 0.5, 0.2))
#                ax.set_ylabel('%s' % tl, rotation=0, ha = 'right')
                ax.set_yticks([])
                ax.set_xlim([0, 140])
                ax.plot([tl, tl], [0, clnC.max()],
                        color=(0.7, 0.3, 0.3), linewidth=2)
                if not ax.is_last_row():
                    ax.set_xticks([])
                if sd == 0 and ax.is_first_row():
                    ax.set_ylabel(r'$\mu = $%s' % mu, rotation=0, ha='right')
                if m == 0 and ax.is_first_row():
                    ax.set_title(r'$\sigma = $%s' % sigma)
                fig.add_subplot(ax)


def calculate_performance(shape_pars, alphas, num_games=10, fignum=14, nS=72, thres=60):
    """ Returns a dictionary containing the performance of the model for all the
    parameter values contained in shape_pars and alpha.
    """
    if not isinstance(shape_pars[1], list):
        shape_pars = invp.switch_shape_pars(shape_pars)
    print(alphas, shape_pars)
    mabe = bc.betMDP(nS=nS, thres=thres)
    big_index = itertools.product(alphas, *shape_pars[1:])
    fig = plt.figure(fignum)
    success = {}
    for i, index in enumerate(big_index):
        success[index] = 0
        for g in range(num_games):
            shape_pars_it = list(index[1:])
            shape_pars_it.insert(0, shape_pars[0])
            mabe.alpha = index[0]
            mabe.set_prior_goals(shape_pars=shape_pars_it, cutoff=False)
            results = mabe.full_inference(just_return=True)
            # tmp = list(index)
            # tmp.append(g)
            # results_index = tuple(tmp)
            # results[results_index] = mabe.full_inference(just_return=True)
            if results['RealStates'][-1] % nS >= mabe.thres:
                success[index] += 1 / num_games

    return success


global MADICT
MADICT = {}


def _get_performance_from_files():
    """Retrieves the performance files saved by _save_result()."""
    results = {}
    for file in os.listdir('./performance/'):
        with open('./performance/' + file, 'rb') as mafi:
            result_tmp = pickle.load(mafi)
            alpha = int(10 * result_tmp[0][0]) / 10
            key = (alpha,) + result_tmp[0][1:]
            results[key] = result_tmp[1]
    return results


def _calculate_performance(le_input):
    """calculate_performance() with single input for Queue()."""
    key = tuple(le_input)
    value = calculate_performance(
        list(le_input[1]), [le_input[0]], num_games=10).popitem()[-1]
    MADICT[key] = value
    print(MADICT)
    return [key, value]


def _save_result(le_input):
    """Saves results of _calculate_performace() to random files."""
    while 1:
        filename = './performance/dict_%s.pi' % np.random.randint(0, 200000)
        if not os.path.isfile(filename):
            break
    with open(filename, 'wb') as mafi:
        pickle.dump(le_input, mafi)


def loop_calculate_performace(num_games, nS=72, thres=60):
    """Loops over calculate_performace for some values of invp.__rds__()."""
    import multiprocessing as mp
    the_generator = _create_queue_list()
    my_robots = mp.Pool(8)
    for input in the_generator:
        my_robots.apply_async(_calculate_performance,
                              args=[input], callback=_save_result)
    my_robots.close()
    my_robots.join()
    print(MADICT)


def _create_queue_list():
    """Creates a list for Queue.map() in loop_calculate_performace()."""
    # step sizes
    step_sizes = {}
    step_sizes['unimodal_s'] = [5, 3]
    step_sizes['exponential'] = [10]
    step_sizes['sigmoid_s'] = [5, 2]
    step_sizes['alpha'] = 10

    shape_pars_all, alphas = invp.__rds__(alpha=True)

    alphas = alphas[alphas < 6]
    alphas = alphas[np.linspace(
        0, len(alphas), step_sizes['alpha'], endpoint=False, dtype=int)]
    # shapes
    for shape_pars in shape_pars_all:
        for ix_sp, par_vecs in enumerate(shape_pars[1:]):
            indices = np.arange(
                0, len(par_vecs), step_sizes[shape_pars[0]][ix_sp])
            shape_pars[ix_sp + 1] = par_vecs[indices]

    # generators
    generators = []
    for shape_pars in shape_pars_all:
        shape_pars[0] = [shape_pars[0]]
        generators.append(itertools.product(*shape_pars))
        generators[-1] = itertools.product(alphas, generators[-1])

    the_generator = itertools.chain(*generators)

    return the_generator


def three_shapes_mc(retorno=True, subjects=None):
    """Calculates the data likelihood for three distinct utility shapes:
    Gaussian, sigmoid and exponential.

    It incorporates rules for the allowed values of the parameters for the three
    shapes:
    Gaussian: At least \mu + \sd points must be visible (not cut off).
    Sigmoid: The maximum must be at least 0.95
    Exponential: No special rules.
    """

    data, data_flat = imda.main()

    if isinstance(subjects, int):
        subjects = subjects,

    if subjects is not None:
        data_flat = [data_flat[t] for t in subjects]

    def prepare_data(thres, data_flat):
        from copy import deepcopy
        dataL = deepcopy(data_flat)

        for datum in dataL:
            indices = datum['threshold'] == thres
            for field in ['choice', 'trial', 'obs', 'reihe', 'threshold']:
                datum[field] = datum[field][indices]
        return dataL

    target_levels = np.array([595, 930, 1035, 1105])
    target_levels_s = [60, 93, 104, 110]
    logli = {}

    # Begin with Gaussian
    with open('./data/posteriors_subj_uni.pi', 'rb') as mafi:
        as_seen = pickle.load(mafi)
    for l, lvl in enumerate(target_levels):
        mu_vec = np.arange(-15, np.round(target_levels_s[l] * 0.2))
        dataL = prepare_data(lvl, data_flat)
        nS = np.round(target_levels_s[l] * 1.2)
        for mu in mu_vec:
            sd_vec = np.arange(1, min(nS - mu, 15))
            for s in range(len(dataL)):
                tmp_logli, _, _, _, _, _ = invp.infer_parameters(data_flat=dataL[s],
                                                                 as_seen=as_seen, normalize=False,
                                                                 shape_pars=['unimodal_s', [mu], sd_vec])
                for sd in range(len(sd_vec)):
                    logli[('unimodal_s', s, lvl, mu, sd_vec[sd])
                          ] = tmp_logli[0, sd]

    # Begin with sigmoid
    with open('./data/posteriors_subj_sigmoid_s.pi', 'rb') as mafi:
        as_seen = pickle.load(mafi)
    cutoff_point = 0.95
    for l, lvl in enumerate(target_levels):
        mu_vec = np.arange(-15, 15)
        dataL = prepare_data(lvl, data_flat)
        nS = np.round(target_levels_s[1] * 1.2)
        for mu in mu_vec:
            possible_slopes = np.arange(1, 30, 2)
#            possible_slopes = np.hstack([possible_slopes, 31])
            slope_min = 10 * np.log(1 / cutoff_point - 1) / \
                (mu + target_levels_s[l] - nS)
            slope_av_min = np.searchsorted(
                possible_slopes, slope_min, side='right')
            if slope_av_min >= len(possible_slopes) - 1:
                continue
            slope_vec = np.arange(possible_slopes[slope_av_min], 30, 2)
            for s in range(len(dataL)):
                tmp_logli, _, _, _, _, _ = invp.infer_parameters(data=dataL[s],
                                                                 as_seen=as_seen, normalize=False,
                                                                 shape_pars=['sigmoid_s', [mu], slope_vec])
                for sl in range(len(slope_vec)):
                    logli[('sigmoid_s', s, lvl, mu, slope_vec[sl])
                          ] = tmp_logli[0, sl]

    with open('./data/posteriors_subj_exp.pi', 'rb') as mafi:
        as_seen = pickle.load(mafi)
    for l, lvl in enumerate(target_levels):
        exp_vec = np.arange(5, 100, 2)
        dataL = prepare_data(lvl, data_flat)
        nS = np.round(target_levels_s[l] * 1.2)
        for s in range(len(dataL)):
            tmp_logli, _, _, _, _, _ = invp.infer_parameters(data=dataL[s],
                                                             as_seen=as_seen, normalize=False,
                                                             shape_pars=['exponential',  exp_vec])
        for ex in range(len(exp_vec)):
            logli[('exponential', s, lvl, exp_vec[ex])] = tmp_logli[ex]

    if retorno:
        return logli
    else:
        with open('./logli_all_shapes.pi', 'wb') as mafi:
            pickle.dump(logli, mafi)


def plot_three_shapes(logli, shapes=None, norm_const=1, fignum=15, normalize_together=False):
    """ Plots the dictionary from the three_shapes() method; creates a plot with the
    shapes of the goals, whose alpha (transparency) is given by their likelihood.

    Parameters
    ----------
    logli: dictionary, keys={shape,subj, thres, par1, par2}
        Contains the log-likelihoods of each key.
    shapes: list of str
        Which shapes are contained in the data. If not provided, it is inferred
        from the data itself.
    norm_const, int
        Controls the maximum alpha that any given lnC can have. The idea is that for
        logli data that spans many subjects, this number is set to the number of
        subjects or something like that.


    """
    import matplotlib.gridspec as gs

    target_levels = [595, 930, 1035, 1105]
    tl_dict = {595: 0, 930: 1, 1035: 2, 1105: 3}

    mabes = {}
    for lvl in target_levels:
        thres = np.round(lvl / 10).astype(int)
        nS = np.round(1.2 * thres).astype(int)
        mabes[lvl] = bc.betMDP(thres=thres, nS=nS)

    # find the maxima to normalize the alpha:
    plots_set = set()
    plots = {}
    inv_plots = {}
    if shapes is None:  # See which shapes are included in the logli data
        for key in logli.keys():
            plots_set.add(key[0])
    else:
        for masha in shapes:
            plots_set.add(masha)
    for t, thing in enumerate(plots_set):
        plots[thing] = t
        inv_plots[t] = thing

#    plots = {'unimodal_s':0, 'sigmoid_s':1, 'exponential':2}

    outer_grid = gs.GridSpec(1, len(plots))

    max_likelihoods = -np.inf * np.ones((len(plots), 4))
    for key in logli.keys():
        if key[0] not in plots:
            continue
        max_likelihoods[plots[key[0]], tl_dict[key[2]]] = max(
            max_likelihoods[plots[key[0]], tl_dict[key[2]]], logli[key])
    max_likelihoods = np.exp(max_likelihoods)
    if normalize_together:
        max_likelihoods = np.tile(max_likelihoods.max(axis=0), (len(plots), 1))
    fig = plt.figure(fignum)
    fig.clf()
    lnC = [0]  # For cases in which not all shapes have data in logli
    for sh in plots.keys():
        for lvl in target_levels:
            inner_grid = gs.GridSpecFromSubplotSpec(
                4, 1, subplot_spec=outer_grid[plots[sh]])
            ax = plt.Subplot(fig, inner_grid[tl_dict[lvl]])
            for key in logli.keys():
                #        plt.subplot(1,2,plots[key[0]])
                if key[2] == lvl and key[0] == sh:
                    lnC = mabes[key[2]].set_prior_goals(select_shape=key[0],
                                                        shape_pars=key[3:], convolute=False,
                                                        cutoff=False, just_return=True)
                    ax.plot(lnC, color='black', alpha=np.exp(
                        logli[key]) / max_likelihoods[plots[key[0]], tl_dict[key[2]]] / norm_const)

            ax.coordinates = [plots[sh], tl_dict[lvl]]
            ticks = ax.get_yticks()
            ymax = ticks[-1]
            ax.set_yticks([])
            ax.set_xlim([0, len(lnC)])
            ax.plot([lvl / 10, lvl / 10], [0, ymax],
                    color='r', linewidth=3, alpha=0.5)
            fig.add_subplot(ax)

    for ax in fig.get_axes():
        if ax.coordinates[0] == 0:
            ax.set_ylabel('threshold:\n %s' %
                          target_levels[ax.rowNum], fontsize=8)
#        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        if ax.rowNum == 3:
            ax.set_xlabel('Points')
        if ax.rowNum == 0:
            ax.set_title(inv_plots[ax.coordinates[0]])
    fig.suptitle(
        'Likelihood for different shapes of priors over final state (goals)')
    plt.savefig('./logli.png', dpi=300)


def plot_inferred_shapes(data_flat, as_seen, shape_pars, shape_pars_r=None,
                         showfig=True, fignum=16, figname=None):
    """ Plots the shapes of the inferred parameters.
    shape_pars_r, if given, is used to plot the shape used for simulating data.
    """
    from itertools import product

    def prepare_data(thres, data_flat_t):
        """Selects those observations with the given threshold."""
        from copy import deepcopy
        dataL = deepcopy(data_flat_t)

        for datum in dataL:
            indices = datum['threshold'] == thres
            for field in ['choice', 'trial', 'obs', 'reihe', 'threshold']:
                datum[field] = datum[field][indices]
        return dataL

    target_levels = data_flat[0]['TargetLevels']

    mabes = {}
    fig = plt.figure(fignum)
    for lvl in target_levels:
        thres = np.round(lvl / 10).astype(int)
        nS = np.round(1.2 * thres).astype(int)
        mabes[lvl] = bc.betMDP(thres=thres, nS=nS)

    for sp, lvl in enumerate(target_levels):
        plt.subplot(4, 1, sp + 1)

        dataL = prepare_data(lvl, data_flat)
        logli, _, _, _, _, _ = invp.infer_parameters(data_flat=dataL[0],
                                                     as_seen=as_seen, normalize=False, shape_pars=shape_pars)

        likelihood = np.exp(logli - logli.max())
        likelihood /= likelihood.sum()

        best_model = np.unravel_index(np.argmax(likelihood), likelihood.shape)
        aux_big_index = [range(len(x)) for x in shape_pars[1:]]
        big_index = product(*aux_big_index)
        for index in big_index:
            c_shape_pars = [shape_pars[i + 1][index[i]]
                            for i in range(len(index))]
            lnC = mabes[lvl].set_prior_goals(select_shape=shape_pars[0],
                                             shape_pars=c_shape_pars, just_return=True,
                                             cutoff=False, convolute=False)
            ax = plt.plot(
                lnC, alpha=likelihood[index], color='black', label='Best fits')
            if index == tuple(best_model):
                best_model_ax, = ax
        if shape_pars_r is not None:
            lnC = mabes[lvl].set_prior_goals(select_shape=shape_pars_r[0],
                                             shape_pars=shape_pars_r[1:], just_return=True,
                                             cutoff=False, convolute=False)
            true_model_ax, = plt.plot(
                lnC, alpha=0.5, color='green', linewidth=1.5, label='True numbers')
            plt.legend(handles=[best_model_ax, true_model_ax],
                       loc=2, fontsize='x-small')
        ylim = plt.gca().get_ylim()
        ymax = ylim[1]
        plt.plot([lvl / 10, lvl / 10], [0, ymax],
                 color='red', linewidth=3, alpha=0.5)
        plt.gca().set_ylim(ylim)
        plt.gca().set_yticks([])
        plt.gca().set_ylabel('Threshold:\n %s' % np.round(lvl / 10).astype(int))
    if showfig:
        plt.show()
    else:
        if figname is None:
            figname = './fig%s.png' % fignum
        plt.savefig(figname, dpi=300)


def prepare_data_for_agent(data_flat):
    """ Changes the format of the data from data_flat to --context--, as
     expected by simulate_with_agent() below.
    """

    context = []

    for datum in data_flat:
        deci, trial, state, thres, reihe = (datum['choice'], datum['trial'],
                                            datum['obs'], datum['threshold'],
                                            datum['reihe'])
        state = np.round(state / 10).astype(int)
        thres = np.round(thres / 10).astype(int)
        deci, trial, state, thres, reihe, _ = invp._remove_above_thres(deci, trial,
                                                                       state, thres, reihe)

        invp._shift_states(state, thres, reihe)

        for t in range(len(deci)):
            context.append([state[t], trial[t], thres[t]])
    return np.array(context, dtype=int)


def adapt_data(datum):
    """ Extracts the data from dict, divides by 10, removes above maxS and
    convolutes with the offers.
    """

    deci, trial, state, thres, reihe = (datum['choice'], datum['trial'],
                                        datum['obs'], datum['threshold'],
                                        datum['reihe'])
    state = np.round(state / 10).astype(int)
    thres = np.round(thres / 10).astype(int)
    deci, trial, state, thres, reihe, _ = invp._remove_above_thres(deci, trial,
                                                                   state, thres, reihe)

    invp._shift_states(state, thres, reihe)

    return deci, trial, state, thres, reihe,


def simulate_with_agent(context, shape_pars, alpha=None):
    """ Calculates the posterior over actions for all the observations in
    --context-- with the agent created with the parameters in --shape_pars--.

    Parameters
    ----------
    context: np.array
        2D array containing an observation in each row, with the columns
        representing: points, trial, threshold.
        Note: it is checked whether Threshold is in 100s or 1000s, and adjusted
              to 100s if necessary.
    shape_pars: list
        List with the following elements:
            Shape name. As found in betClass.set_prior_goals().
            Parameter values. One element of the list for each parameters,
                as ordered in betClass.set_prior_goals().
    """
    from itertools import product as itprod

    target_lvls = (np.round(np.array([595, 930, 1035, 1105]) / 10)).astype(int)

    # Check if threshold is in 100s or 1000s. Adjust to 100s.
    if context[0, 2] > 200:
        context[:, 2] = (np.round(context[:, 2] / 10)).astype(int)

    mabes = {}
    dummy_QS = {}
    for tl in target_lvls:
        mabes[tl] = bc.betMDP(thres=tl, nS=np.round(1.2 * tl))
        if alpha is not None:
            mabes[tl].alpha = alpha
        mabes[tl].set_prior_goals(select_shape=shape_pars[0],
                                  shape_pars=shape_pars[1:], cutoff=False)
        dummy_QS[tl] = mabes[tl].S
        nT = mabes[tl].nT
    wV = {}
    for t in range(nT):
        wV[t] = np.array(list(itprod([0, 1], repeat=nT - t)))

    posta = np.zeros((context.shape[0], 2))
    for n, obs in enumerate(context):
        _, posta[n, :], _ = mabes[obs[2]].posteriorOverStates(obs[0], obs[1],
                                                              wV[obs[1]
                                                                 ], dummy_QS[obs[2]],
                                                              0, 1)

    return posta


def get_context_rp(context):
    """ Gets risk pressure for all rows of --context--."""
    nT = 8
    return (context[:, 2] - context[:, 0] % (np.round(1.2 * context[:, 2]))) / (nT - context[:, 1])


def plot_rp_vs_risky(as_seen=None, fignum=17, subjects=None, savefig=False):
    """ Plots risk pressure vs probability of choosing the risky offer as a scatter plot."""

    from utils import calc_subplots
    import scipy as sp

    if subjects is None:
        subjects = 0,
    num_subs = len(subjects)
    s1, s2 = calc_subplots(num_subs)
    fig = plt.figure(fignum)
    fig.clf()

    data, data_flat = imda.main()
    risk_p = calc_risk_pressure(data_flat)
    if as_seen is None:
        with open('./data/posteriors_subj_uni.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)

    def mafu(x, a, b): return a * x + b
    for s in subjects:

        logli, posta, _, _, _, _ = invp.infer_parameters(
            as_seen=as_seen, normalize=False,
            shape_pars=['unimodal_s', np.arange(-15, 20), np.arange(1, 15)],
            data=data_flat[s])
        max_model = np.unravel_index(logli.argmax(), logli.shape)
        datax = risk_p[s]
        datay = posta[max_model[0], max_model[1], :, 1]
        par_opt, par_covar = sp.optimize.curve_fit(mafu, datax, datay)
        xmin = risk_p[s].min()
        xmax = risk_p[s].max()
        lin_x = np.arange(xmin, xmax, 0.1)
        lin_y = mafu(lin_x, *par_opt)

        xticks = np.arange(0, np.ceil(xmax / 10) * 10 + 1, 10)

        ax = plt.subplot(s1, s2, s + 1)
        ax.scatter(datax, datay, color='r', s=1)
        ax.plot(lin_x, lin_y, color='black', alpha=0.5)
        ax.set_xlim([0, 80])
        ax.set_ylim([0, 1])
        if ax.is_first_col() and ax.is_last_row():
            ax.set_xlabel('Risk pressure')
            ax.set_ylabel('Probability of Risky')

            ax.set_xticklabels(xticks.astype(int))
            ax.set_yticklabels(ax.get_yticks())
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
    if savefig is True:
        plt.savefig('./risky.png', dpi=300)
    else:
        plt.show(block=False)


def likelihood_simulated(nReps=10, shape_pars=None, thres_ix=1,
                         nGames=12, noise=0, inv_temp=0, alpha=64):
    """ Picks random model parameters for active inference, simulates data
    for an entire subject (1 condition, nGames mini-blocks) and finds the
    log-likelihood of the model used to the data.
    """

    target_levels = np.array([595, 930, 1035, 1105])
    target_lvl = np.round(target_levels / 10).astype(int)

    if shape_pars is None:
        shape_pars = ['unimodal_s', 5, 3]
    thres = target_lvl[thres_ix]
    mabe = bc.betMDP(thres=thres, nS=np.round(1.2 * thres))
    mabe.alpha = alpha
    mabe.set_prior_goals(select_shape=shape_pars[0], shape_pars=shape_pars[1:],
                         convolute=True, cutoff=False, just_return=False)

    nT = mabe.nT
    posta = np.zeros((nGames * nT, 2))

    # Generate posteriors
    for n in range(nGames):
        mabe.exampleFull()
        posta[n * nT:(n + 1) * nT, :] = mabe.Example['PostActions']

    # Generate data
#    choice = np.zeros(nReps)
    logli = np.zeros(nReps)
    for rep in range(nReps):
        for t, cPosta in enumerate(posta):
            if cPosta.sum() == 0:
                cPosta = np.array([0.5, 0.5])
            # add noise
            if inv_temp != 0:
                cPosta += inv_temp
                cPosta /= cPosta.sum()
            if noise != 0:
                cPosta += (-noise + 2 * noise * np.random.rand(2)) * cPosta
                if cPosta.min() == 0:
                    cPosta += 0.001
                cPosta /= cPosta.sum()
            logli[rep] += np.random.choice(np.log(cPosta), p=cPosta)
    return logli


def likelihood_data(shape_pars, thres_ix=0, subject=0, data_flat=None,
                    alpha=None):
    """ Gets experimental data and calculates the likelihood for the actinf
    model given by shape_pars.
    """
    import betClass as bc
    import import_data as imda
    import numpy as np

    target_levels = np.array([595, 930, 1035, 1105])
    target_lvl = np.round(target_levels / 10).astype(int)
    thres_sim = target_lvl[thres_ix]

    if data_flat is None:
        _, data_flat = imda.main()
        data_flat = data_flat[subject]
    else:
        try:
            data_flat = data_flat[0]
        except KeyError:
            pass

    deci, trial, state, thres, reihe, = adapt_data(data_flat)
    context = prepare_data_for_agent([data_flat])
    context = context[context[:, 2] == thres_sim, :]

    posta = simulate_with_agent(context, shape_pars, alpha=alpha)

    nD = posta.shape[0]

    logli = 0
    for d in range(nD):
        logli += np.log(posta[d, 0]**(deci[d] == 0)
                        * posta[d, 1]**(deci[d] == 1))

    return logli, posta, context


def plot_likelihood_alpha(subjects, filenames_base=None, fignum=18):
    """ Plots, for each subject in --subjects--, a histogram of the likelihoods
    of the data for each value of alpha in the supplied files.
    """

    if isinstance(subjects, int):
        subjects = subjects,
    if filenames_base is None:
        filenames_base = '/home/dario/Work/Proj_ActiveInference/results/alpha_logli_subj_%s_unimodal_s.pi'

    plt.figure(fignum)
    s1, s2 = utils.calc_subplots(len(subjects))
    for ns, s in enumerate(subjects):
        with open(filenames_base % s, 'rb') as mafi:
            logli = pickle.load(mafi)

        plt.subplot(s1, s2, ns + 1)
        xdata = []
        ydata = []
        for key in logli.keys():
            xdata.append(key)
            ydata.append(logli[key][0].max())

        plt.bar(np.log(xdata), ydata)
    plt.show()


def plot_evolution_likelihood_map(subject, shape='unimodal_s', filenames_base=None, fignum=19):
    """ Plots the likelihood map on the mu-sd plane as alpha changes."""

    if filenames_base is None:
        filenames_base = 'home/dario/Work/Proj_ActiveInference/results/alpha_logli_subj_%s_%s.pi'

    with open(filenames_base % (subject, shape), 'rb') as mafi:
        logli = pickle.load(mafi)

    s1, s2 = calc_subplots(len(logli))
    plt.figure(fignum)
    plt.set_cmap('gray_r')
    keyset = np.sort([key for key in logli.keys()])
    for n, key in enumerate(keyset):
        ax = plt.subplot(s1, s2, n + 1)
        likelihood = np.exp(logli[key][0] - logli[key][0].max())
        ax.imshow(likelihood, aspect=0.25)
        if not ax.is_last_row():
            ax.set_xticks([])
        ax.set_title('%s, %s' % (key, logli[key][0].max()))

    plt.show(block=False)


def plot_best_alpha_per_condition(shape_pars, subjects=None, fignum=20):
    """ Grabs the best alpha valua for each subject from file (which?), creates an
    as_seen for each subject, puts them all together and calls plot_by_thres().

    The files used to get the best alphas are those called
    alpha_logli_subj_#_unimodal_s.pi, created by invp.invert_alpha().
    """

    if isinstance(subjects, int):
        subjects = subjects,

    file_base_alpha = './data/alpha_logli_subj_%s_%s.pi'
    file_base_Qs = './data/qus_subj_%s_%s.pi'
    as_seen = {}
    alpha = {}
    for s in subjects:
        with open(file_base_alpha % (s, shape_pars[0]), 'rb') as mafi:
            likeli_alpha = pickle.load(mafi)
        c_max = -np.inf
        for k, key in enumerate(likeli_alpha.keys()):
            if c_max < likeli_alpha[key].max():
                alpha[s] = key
                c_max = likeli_alpha[key].max()
#            max_li.append([key,likeli_alpha[key].max()])
#        alpha[s] = max_li[np.argmax(max_li[:][1])][0]

        as_seen.update(invp.calculate_posta_from_Q(alpha[s], file_base_Qs % (s, shape_pars[0]),
                                                   guardar=False, regresar=True))
    logli = plot_by_thres(shape_pars=shape_pars,
                          subjects=subjects, as_seen=as_seen, fignum=fignum)
    return logli, alpha


def find_best_pars(subjects, shapes=None):
    """Finds the best model for each subject, separated by condition.

    WARNING: Bad assumption. It finds the best model for all data (i.e. not
    per condition) and takes the alpha from there. Then, with that alpha, it goes
    back and separates by condition. This only works if the best alpha for all
    data is the same as the same alpha per condition, which I wouldn't expect
    to hold true.

    If --shapes-- is provided, only those shapes are considered.
    """

    if shapes is None:
        shapes = [x[0] for x in invp.__rds__()]

    logli = {}  # First entry: shape. Second entry: (subj, thres)
    alpha = {}
    for shape in shapes:
        logli[shape], alpha[shape] = plot_best_alpha_per_condition(
            invp.__rds__(shape), subjects)

    # First entry: subj. Each element a list with [logli, [pars]]
    best_pars = {}
    for subj in subjects:
        for shape in shapes:
            for th in range(4):  # One per threshold
                best_pars[(subj, th, shape)] = [-np.inf, []]
                c_max_logli = logli[shape][(subj, th)].max()
                if c_max_logli > best_pars[(subj, th, shape)][0]:
                    best_indices = np.unravel_index(np.argmax(logli[shape][(subj, th)]),
                                                    logli[shape][(subj, th)].shape)
                    try:
                        tmp = len(best_indices)
                    except TypeError:
                        best_indices = best_indices,

                    c_shape_pars = invp.__rds__(shape)[1:]
                    best_par_values = [c_shape_pars[i][best_indices[i]]
                                       for i in range(len(best_indices))]
                    best_pars[(subj, th, shape)] = [
                        c_max_logli, best_par_values]
    return best_pars


def _search_logli_for_best(logli, shape, number_save, biggest, keyest,
                           force_rule=False):
    """Given a logli, as taken from alpha_logli_subj_#_shape.pi, finds
    the best --number_save-- likelihoods.

    Meant to be used with rank_likelihoods().

    WARNING: biggest and keyest are modified here.
    """

    for key in logli.keys():
        it = np.nditer(logli[key], flags=['multi_index'])
        while not it.finished:
            c_logli = it[0]
            c_index = it.multi_index
            c_biggest = np.where(biggest < c_logli)[0]
            if c_biggest.size > 0:
                new_index = c_biggest[0]
                keyest[new_index + 1:] = keyest[new_index:-1]
                shape_ind = [shape]
                for ix_c in c_index:
                    shape_ind.append(ix_c)
                shape_pars = transform_shape_pars(shape_ind)
                if force_rule is True:
                    if not rules_for_shapes(shape_pars):
                        it.iternext()
                        continue
                keyest[new_index] = [int(key * 10) / 10, shape_pars]
                biggest[new_index + 1:] = biggest[new_index:-1]
                biggest[new_index] = c_logli
            it.iternext()


def performance_subjects(subjects):
    """ Calculates the subjects' performances in the data."""
    data, _ = imda.main()

    if isinstance(subjects, int):
        subjects = subjects,
    performance = {}
    for subject in subjects:
        performance[subject] = (
            (data[subject]['obs'][:, -1] > data[subject]['threshold']) * 1).sum() / 48
    return performance


def transform_shape_pars(shape_indices):
    """Transforms the weird index-based shape_pars into the value-based ones."""
    shape_pars_vec = invp.__rds__(shape_indices[0])
    shape_values = [shape_pars_vec[0], ]
    for index, par_index in enumerate(shape_indices):
        if index == 0:
            continue
        shape_values.append(shape_pars_vec[index][par_index])
    return shape_values


def get_posta_best_model(subject):
    """Retrieves from file the posterior over actions for the subject, for the
    value of the parameters with the highest likelihood (best model).
    """

    _, best_key = rank_likelihoods(subject, 1)

    q_filename = './data/qus_subj_%s_%s.pi' % (subject, best_key[0][1][0])

    return (best_key[0],
            invp.calculate_posta_from_Q(best_key[0][0], Qs=q_filename,
                                        guardar=False, regresar=True))


def rules_for_shapes(shape_pars, number_states=72):
    r"""Checks whether the shape described in --shape_pars-- meets the rules
    established for distinguishing between shapes. These are as follow:
    Gaussian: \mu + \sd < number_states
    Sigmoid: sigmoid[number_states] > 0.95
    Exponential: No especial rules.

    The function returns True if the rules are met.
    """
    threshold = np.ceil(number_states / 1.2)

    if shape_pars[0] == 'unimodal_s':
        rules_met = shape_pars[1] + shape_pars[2] + threshold <= number_states
    elif shape_pars[0] == 'sigmoid_s':
        k = shape_pars[2] / 10  # see the definition of bc.set_prior_goals()
        x0 = shape_pars[1] + threshold
        y = 0.95
        rules_met = number_states > -(1 / k) * np.log((1 / y) - 1) + x0
    elif shape_pars[0] == 'exponential':
        rules_met = True
    else:
        raise ValueError('The shape in shape_pars is not recognized')

    return rules_met


def akaike_compare(subjects=None):
    """Compares two models: one with inter-subject differences in parameters and one
    without.

    --subjects-- defaults to range(20)
    """
    if subjects is None:
        subjects = range(20)
    max_logli_all_data = maximum_likelihood_all_data(subjects)
    max_logli_per_sub = np.zeros(len(subjects))
    for ix_subj, subject in enumerate(subjects):
        max_logli_per_sub[ix_subj], _ = rank_likelihoods(
            subject, number_save=1)
    max_logli_subs = max_logli_per_sub.sum()
    return max_logli_all_data, max_logli_subs
    aic_all = 2 * 3 - 2 * max_logli_all_data
    aic_sub = 2 * 3 * len(subjects) - 2 * max_logli_subs
    return aic_all, aic_sub


def maximum_likelihood_all_data(subjects):
    """Finds the model's maximum loglikelihood for all the data, assuming no
    inter-subject variations.
    """
    shape_pars_all = invp.__rds__()
    shapes = [x[0] for x in shape_pars_all]

    logli_filename_base = './data/alpha_logli_subj_%s_%s.pi'
    logli_full_model = {}
    logli_subject = {}
    for subject in subjects:
        logli_subject[subject] = 0
        for shape in shapes:
            logli_filename = logli_filename_base % (subject, shape)
            with open(logli_filename, 'rb') as mafi:
                logli = pickle.load(mafi)
            # ipdb.set_trace()
            for key in logli:
                if (key, shape) in logli_full_model:
                    logli_full_model[(key, shape)] += logli[key]
                else:
                    logli_full_model[(key, shape)] = logli[key]
    max_logli = -np.inf
    for key in logli_full_model.keys():
        max_logli = max(max_logli, logli_full_model[key].max())
    return max_logli


def rank_likelihoods(subject, shapes=None, number_save=2, force_rule=False):
    """ Goes through the subject's alpha_logli file and finds the X models with
    the highest likelihoods. X is given by number_save.

    Parameters
    ----------
    shapes: list of strings
        Which shapes (from the ones in invp.__rds__()) to look for. If only one shape
        is given, it must still be in the form of a list (e.g. ['exponential',])

    Returns
    -------
    biggest: np.array, size={number_save}
        Likelihood for the --number_save-- most likely models.
    keyest: list, len=number_save
        For each of the models in --biggest--, reports the parameter indices and shape
        in the following format: [alpha, [shape pars], shape].

    """
    if shapes is None:
        shape_pars = invp.__rds__()
        shapes = [sp[0] for sp in shape_pars]
    else:
        try:
            _ = invp.__rds__(shapes[0])
        except ValueError:
            raise ValueError('The given --shapes-- seems to be wrong.' +
                             'Are you sure you gave it as a list?')

    biggest = np.array([-np.inf] * number_save)
    keyest = [[]] * number_save

    for shape in shapes:
        logli_alpha_filename = './data/alpha_logli_subj_%s_%s.pi' % (
            subject, shape)

        with open(logli_alpha_filename, 'rb') as mafi:
            logli = pickle.load(mafi)
        _search_logli_for_best(logli, shape, number_save, biggest, keyest,
                               force_rule=force_rule)
    return biggest, keyest


def rank_likelihoods_cond(subject, number_save=5, force_rule=False):
    """Finds the best parameters for the subject, divided by condition.
    Makes use of the logli_alpha_subj_cond_shape.pi files created by
    invp.calculate_likelihood_by_condition().

    Returns
    -------
    best_pars: dict
        Dictionary whose keys are the subjects in --subjects--, and whose elements are
        a list with one element per th (4 in total), each element being a the list
        [likelihood, alpha, shape_pars].
    """

    shapes = [x[0] for x in invp.__rds__()]
    th_lvls = range(4)
    # (subj, cond, shape)
    loglis_filebase = './data/logli_alpha_subj_%s_cond_%s_%s.pi'
    biggest = {}
    keyest = {}
    for th in th_lvls:
        biggest[th] = np.array([-np.inf] * number_save)
        keyest[th] = [[]] * number_save
        for shape in shapes:
            with open(loglis_filebase % (subject, th, shape), 'rb') as mafi:
                logli_cond = pickle.load(mafi)
                _search_logli_for_best(
                    logli_cond, shape, number_save, biggest[th], keyest[th],
                    force_rule=force_rule)
    return biggest, keyest


def loop_rank_likelihoods(subjects=None, number_save=1, shapes=None,
                          cond=False, force_rule=True):
    """Loops over rank_likelihoods or rank_likelihoods_cond, depending on --cond--."""
    if subjects is None:
        subjects = range(35)
    if cond is True:
        ranking_function = rank_likelihoods_cond
    else:
        ranking_function = rank_likelihoods

    best_model = {}
    for subject in subjects:
        best_model[subject] = ranking_function(
            subject, number_save=number_save, force_rule=force_rule, shapes=shapes)

    return best_model


def fitted_decisions(subjects=(0,), fignum=21, maax=None):
    """For the given subject, the best agent's posteriors are
    plotted alongside the choices made by the subject.

    If axes are provided in maax, the whole thing is plotted there. Notice
    that having multiple subjects with maax=axes will yield weird results.

    If no axes are provided, fignum=21 is used, and multiple subjects are
    plotted in subplots.
    """

    _, data_flat = imda.main()

    if isinstance(subjects, int):
        subjects = subjects,
    if maax is None:
        s1, s2 = calc_subplots(len(subjects))
        fig = plt.figure(fignum, figsize=[6, 4 * s1])
        fig.clf()
        flag_create_max = True

    for num_subj, subject in enumerate(subjects):
        if flag_create_max is True:
            maax = plt.subplot(s1, s2, num_subj + 1)
        best_key, posta = get_posta_best_model(subject)
        shape_pars = best_key[1]
        alpha = best_key[0]
        shape_pars = invp.switch_shape_pars(shape_pars)
        _, posteriors, deci, _, _, _ = invp.infer_parameters(
            data_flat=data_flat[subject], as_seen=posta, no_calc=True, shape_pars=shape_pars)
        _plot_decisions(maax, shape_pars, posteriors, deci)
    plt.tight_layout()
    plt.show(block=False)

    return alpha, shape_pars, posteriors


def _plot_decisions(maax, shape_pars, posteriors, deci):
    """Plotting function for fitted_decisions()."""

    cmap = figure_colors('lines_cmap')
    color_line = cmap(0)
    color_star = [cmap(0.2)]
    color_star.append(cmap(0.5))

    plot_decision = -np.ones(len(deci))
    posteriors = np.squeeze(posteriors)
    for ix_deci, decision in enumerate(deci):
        plot_decision[ix_deci] = (-1 + decision * 2) * \
            (0.5 - posteriors[ix_deci, 1]) + 0.5
    par_labels = {'unimodal_s': [r'$\mu$', r'$\sigma$'], r'sigmoid_s': [
        r'$\kappa$', r'$x_0$'], 'exponential': 'a'}
    plot_label = ''
    for ix_par, par in enumerate(shape_pars[1:]):
        plot_label += '%s = %2.0f, ' % (
            par_labels[shape_pars[0]][ix_par], par[0])
    plot_label = plot_label[:-2]
    maax.plot(posteriors[:, 1],
              color=color_line, label=plot_label)
    maax.plot(np.arange(len(deci))[
        deci == 0], plot_decision[deci == 0], '.', color=color_star[0])
    maax.plot(np.arange(len(deci))[
        deci == 1], plot_decision[deci == 1], '.', color=color_star[1])
    # maax.plot(plot_decision, '*', color=color_star1)
    maax.set_ylim([0, 1])
    maax.set_xlim([0, len(deci) - 1])
    maax.set_xticks([])
    for ytick in maax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(figure_fonts('axes ticks'))
    maax.set_xlabel('Contexts', fontsize=figure_fonts('axes labels'))
    maax.set_ylabel('Probability\n of Risky',
                    fontsize=figure_fonts('axes_labels'))
    maax.set_label('Label via method')
    maax.legend(fontsize=12)


def posta_best(subjects=None):
    """Calculates the posteriors over actions for the best model (including alpha)
    and saves to the file best_subj_#.pi.
    """
    if subjects is None:
        subjects = range(35)
    if isinstance(subjects, int):
        subjects = subjects,

    for subject in tqdm(subjects):
        _, best_key = rank_likelihoods(subject, number_save=1)
        alpha = best_key[0][0]
        shape_pars = best_key[0][1]
        q_filename = './data/qus_subj_%s_%s.pi' % (subject, shape_pars[0])
        save_filename = './best_model_posta_subj_%s.pi' % subject
        invp.calculate_posta_from_Q(alpha, q_filename, filename=save_filename)


def average_risky(subjects=(0,)):
    """Calculates the average probability of choosing risky for the given subjects.

    The data comes from the files ./data/best_model_posta_subj_0.pi.
    """
    if isinstance(subjects, int):
        subjects = subjects,

    _, flata = imda.main()
    filename = './data/best_model_posta_subj_%s.pi'

    statistics = {}
    for subject in subjects:
        with open(filename % subject, 'rb') as mafi:
            as_seen = pickle.load(mafi)
        _, keyest = rank_likelihoods(subject)
        shape_pars = invp.switch_shape_pars(keyest[0][1])
        _, posta, _, _, _, _ = invp.infer_parameters(data_flat=flata[subject], as_seen=as_seen,
                                                     shape_pars=shape_pars)
        posta = np.squeeze(posta)
        statistics[subject] = (posta[:, 1].mean(), posta[:, 1].std())
    return statistics


def _one_agent_one_obs(subjects=(0, 1)):
    """Retrieves the posterior distributions over actions for the subjects, using the best
    available model.

    Returns
    -------
    posta_all: dict, len = len(subjects)
        ['posteriors', 'rp'] per subject. Keys are subject number.
    """

    _, flata = imda.main()
    posta_all = {}
    risk_pressure = calc_risk_pressure(flata)
    for subject in subjects:
        _, best_key = rank_likelihoods(subject, number_save=1)
        shape_pars = invp.switch_shape_pars(best_key[0][1])
        alpha = best_key[0][0]
        try:
            with open('./data/best_model_posta_subj_%s.pi' % subject, 'rb') as mafi:
                as_seen = pickle.load(mafi)
        except FileNotFoundError:
            q_seen = invp.load_or_calculate_q_file(
                subject, shape_pars[0], create=False)
            as_seen = invp.calculate_posta_from_Q(alpha, q_seen, guardar=False,
                                                  regresar=True)
        _, posta, _, _, _, _ = invp.infer_parameters(data_flat=flata[subject],
                                                     shape_pars=shape_pars,
                                                     no_calc=True,
                                                     as_seen=as_seen)
        posta_all[subject] = {'posta': np.squeeze(
            posta), 'rp': risk_pressure[subject]}

    return posta_all


def _one_agent_many_obs(subjects=(0, 1)):
    """Exposes the fitted agents to the data of all of them and plots preferences vs rp.

    Returns
    -------
    posta_all: dict, len() = len(subjects)
        Each element is a dict with keys 'posta' and 'rp' where the posta.shape=(N, 2)
        and rp.shape = (N,).
    """

    _, flata = imda.main()

    # Collect observations
    posta_all = {}
    for subject_test in subjects:
        posta_all[subject_test] = {'posta': np.array([]), 'rp': np.array([])}
        for subject_data in subjects:
            _, keyest = rank_likelihoods(subject_test, number_save=1)
            alpha = keyest[0][0]
            shape_pars = invp.switch_shape_pars(keyest[0][1])
            qs_filename = './data/qus_subj_%s_%s.pi' % (
                subject_data, shape_pars[0])
            as_seen = invp.calculate_posta_from_Q(
                alpha, Qs=qs_filename, guardar=False, regresar=True)
            _, posta, _, _, _, _ = invp.infer_parameters(
                data_flat=flata[subject_data], as_seen=as_seen, shape_pars=shape_pars)
            posta = np.squeeze(posta)
            r_pre = calc_risk_pressure([flata[subject_data]])[0]
            try:
                posta_all[subject_test]['posta'] = np.vstack(
                    [posta_all[subject_test]['posta'], posta])
            except ValueError:
                posta_all[subject_test]['posta'] = posta
            try:
                posta_all[subject_test]['rp'] = np.hstack(
                    [posta_all[subject_test]['rp'], r_pre])
            except ValueError:
                posta_all[subject_test]['rp'] = r_pre
    return posta_all


def _plot_rp_vs_risky_own(subjects, axes, posta_all=None, regresar=False):
    """Plots the probability of risky against risk pressure for the best model
    for each of the subjects in --subjects--.

    The plot of each subject is sent to the corresponding axis in --axes--.

    Parameters:
    subjects: int or list of ints.
    axes: list of pyplot axes.
        One axis must be given for each subject. Plots will be plotted there.
    """

    if isinstance(subjects, int):
        subjects = subjects,

    if posta_all is None:
        posta_all = _one_agent_one_obs(subjects)

    for ix_sub, subject in enumerate(subjects):
        maax = axes[ix_sub]
        cposta = posta_all[subject]['posta']
        c_risk_pr = posta_all[subject]['rp']
        _rp_vs_risky_subplot(c_risk_pr, cposta, maax)

    if regresar:
        return posta_all


def _plot_rp_vs_risky_all(subjects, axes, posta_all=None, regresar=False):
    """Plots the probability of risky against risk pressure for the best model
    for each of the subjects in --subjects--, for all of each other's observations.
    """

    if isinstance(subjects, int):
        subjects = subjects,
    if posta_all is None:
        # posta_all = _one_agent_many_obs(subjects)
        with open('./posta_all.pi', 'rb') as mafi:
            posta_all = pickle.load(mafi)

    for ix_subj, subject in enumerate(subjects):
        cposta = posta_all[subject]['posta']
        c_risk_pressure = posta_all[subject]['rp']
        _rp_vs_risky_subplot(c_risk_pressure, cposta,
                             axes[ix_subj], color='green', alpha=0.3)
    if regresar:
        return posta_all


def _plot_average_risk_dynamics(posta_rp, maaxes, step_size=1, legend='', regresar=False):
    """Plots the average posterior probability of risky."""
    for ix_cpr, key_cpr in enumerate(posta_rp):
        c_posta = posta_rp[key_cpr]['posta']
        c_risk_pressure = posta_rp[key_cpr]['rp']
        average_rp, std_rp = _average_risk_dynamics(c_posta, c_risk_pressure)
        plot_x = np.arange(0, 35, step_size)
        maaxes[ix_cpr].errorbar(
            plot_x, average_rp, std_rp, linewidth=2, label=legend)
        maaxes[ix_cpr].set_ylim([0, 1])
    if regresar:
        return average_rp, std_rp


def _average_risk_dynamics(posta, risk_pressure, step_size=1):
    """Calculates the average posterior probability of risky for all values of rp, with a
    step size of --step_size--.
    """
    vec_size = np.floor(35 / step_size).astype(int)
    data = [[] for x in range(vec_size)]
    for ix_rp, crp in enumerate(risk_pressure):
        if crp > 34:
            continue
        data[np.floor(crp).astype(int)].append(posta[ix_rp][1])

    average = [[] for _ in range(vec_size)]
    std = [[] for _ in range(vec_size)]
    for ix_datum, datum in enumerate(data):
        average[ix_datum] = np.mean(datum)
        std[ix_datum] = np.std(datum)

    return average, std


def _rp_vs_risky_subplot(risk_pr, posta, maax, color=None, alpha=1):
    """Makes the subplots for figure_rp_vs_risky()."""
    color = figure_colors('one line')
    for point, rp in enumerate(risk_pr):
        if rp > 35:
            continue
        maax.plot(rp, posta[point, 1], '.', color=color, alpha=alpha)
    maax.set_ylim([0, 1])


def _rp_vs_risky_pretty(fig):
    """Formats the subplots from figure_rp_vs_risky() nicely."""
    abc = 'ABCDEFGHIJKHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    label_pos = figure_fonts('subplot title position')
    for ix_ax, maax in enumerate(fig.get_axes()):
        xticks = list(range(5, 46, 10))
        xticks.insert(0, 0)
        maax.set_xticks(xticks)
        maax.set_xticklabels(xticks, fontsize=figure_fonts('axes ticks'))
        maax.set_xlabel('Risk pressure',
                        fontsize=figure_fonts('axes labels'))
        maax.set_ylabel('Probability\n of risky',
                        fontsize=figure_fonts('axes labels'))
        maax.set_yticks([0, 0.5])
        maax.set_yticklabels(
            [0, 0.5], fontsize=figure_fonts('axes_ticks'))
        maax.text(label_pos[0], label_pos[1], abc[ix_ax],
                  size=figure_fonts('subplot title'), transform=maax.transAxes)
        maax.label_outer()
        if not maax.is_last_row():
            maax.set_xlabel('')
        if not maax.is_first_col():
            maax.set_ylabel('')
        maax.set_label('Label via method')
        maax.legend(fontsize=10, loc='upper right', ncol=2)


def model_comparison_family(subjects=(0,), number_save=2000):
    """Performs model comparison between families of goal shapes for the given subjects.

    The 'model comparison' is really just comparing the added likelihoods for each shape.
    Not exactly mathematically sound, but gives an idea."""
    best_models = loop_rank_likelihoods(
        subjects=subjects, number_save=number_save)

    shape_pars_all = invp.__rds__()
    all_shapes = [shape_pars[0] for shape_pars in shape_pars_all]
    prob_family = {subject: {shape: 0 for shape in all_shapes}
                   for subject in subjects}

    for subject in best_models.keys():
        for ix_model, logli in enumerate(best_models[subject][0]):
            shape = best_models[subject][1][ix_model][1][0]
            prob_family[subject][shape] += np.exp(logli)
    return prob_family


def akaike_family(subjects=(0,), just_diffs=False):
    """Returns the AIC for each shape, for the given subjects.

    Parameters
    ----------
    just_diffs: bool
        If True, the AIC differences (AIC_i - AIC_min) will be returned
        instead of the AIC values.
"""
    shape_pars_all = invp.__rds__()
    shapes = [shape_pars[0] for shape_pars in shape_pars_all]
    number_k = {shape_pars[0]: len(shape_pars[1:]) + 1
                for shape_pars in shape_pars_all}

    AIC = {}
    for shape in shapes:
        best_models = loop_rank_likelihoods(
            subjects=subjects, shapes=[shape], number_save=1)
        for subject in best_models.keys():
            AIC[(subject, shape)] = 2 * number_k[shape] - \
                best_models[subject][0][0]
    if just_diffs:
        for subject in subjects:
            min_AIC = np.inf
            for shape in shapes:
                min_AIC = min(AIC[(subject, shape)], min_AIC)
            for shape in shapes:
                AIC[(subject, shape)] -= min_AIC
    return AIC


def _get_data_priors(subjects, criterion='AIC'):
    """Prepares the data for table_best_families(), using the criterion given."""
    if criterion == 'AIC':
        ranking = akaike_family(subjects, just_diffs=True)
        value_comparison = 0
    elif criterion == 'ML':
        ranking = ML_family(subjects, just_ratios=True)
        value_comparison = 1

    shapes = set()
    for key in ranking.keys():
        shapes.add(key[1])
    sorted_subjects = []
    sorted_daics = []
    sorted_families = []
    for shape in shapes:
        for subject in subjects:
            if ranking[(subject, shape)] == value_comparison:
                sorted_subjects.append(subject)
                sorted_families.append(shape)
                daics = [float(ranking[(subject, ls)])
                         for ls in ['exponential', 'sigmoid_s', 'unimodal_s']]
                sorted_daics.append('E:%2.2f, S:%2.2f, U:%2.2f' % tuple(daics))
                # break
    return sorted_subjects, sorted_families, sorted_daics


def ML_family(subjects, just_ratios=False):
    """Returns the ML for each shape, for the given subjects.

    Returns
    -------
    ML: dict
        Dictionary with keys (subject, shape), whose elements are the maximum
        likelihood of the given subject/shape combination.
    """

    shapes = [shape_pars[0] for shape_pars in invp.__rds__()]
    ML = {}
    for shape in shapes:
        best_models = loop_rank_likelihoods(
            subjects=subjects, shapes=[shape], number_save=1)
        for subject in subjects:
            ML[(subject, shape)] = np.exp(best_models[subject][0][0])

    if just_ratios:
        for subject in subjects:
            max_ML = -np.inf
            for shape in shapes:
                max_ML = max(ML[(subject, shape)], max_ML)
            for shape in shapes:
                ML[(subject, shape)] /= max_ML
    return ML


def BF_family(subjects, just_ratios=False):
    """Returns the Bayesian factor for each shape, for the given subjects.

    If just_ratios=True, then only the ratios of the factors to the maximum 
    are returned.
    """
    logli_file_base = './data/alpha_logli_subj_%s_%s.pi'
    shapes = [shape_pars[0] for shape_pars in invp.__rds__()]
    BF = {(subject, shape): 0 for subject in subjects for shape in shapes}
    for shape in shapes:
        # if shape == 'unimodal_s':
            # continue
        for subject in subjects:
            filename = logli_file_base % (subject, shape)
            with open(filename, 'rb') as mafi:
                logli = pickle.load(mafi)
            priors = priors_over_parameters(shape)
            for key in logli:
                BF[(subject, shape)] += np.exp(logli[key]) * priors[key]
            for key in logli:
                BF[(subject, shape)] = BF[(subject, shape)].sum()
    return BF


def priors_over_parameters(shape):
    """Calculates the empirical priors over parameter values, by using the likelihoods
    over parameter values for all the data.
    """
    subjects = range(35)
    logli_filename_base = './data/alpha_logli_subj_%s_%s.pi'
    logli_subject = {}
    flag = 0
    for subject in subjects:
        logli_subject[subject] = 0
        logli_filename = logli_filename_base % (subject, shape)
        with open(logli_filename, 'rb') as mafi:
            logli = pickle.load(mafi)
        if flag == 0:
            priors = {key: 0 for key in logli.keys()}
            flag = 1
        for key in logli:
            priors[key] += logli[key]
    # Normalizing:
    sum_priors = 0
    for key in priors.keys():
        sum_priors += priors[key].sum()
    for key in priors.keys():
        priors[key] /= sum_priors

    return priors


def table_best_families(subjects, criterion='AIC'):
    """Creates a table for the paper, in which the best shape is selected for each subject.
    Subjects are grouped by shapes. The criterion used can either be maximum likelihood,
    or the Akaike Information Criterion (AIC).

    Parameters
    ----------
    criterion: ['AIC', 'ML']
        Criterion to use to decide which family is best for each subject. AIC stands for
        Akaike information criterion, ML for maximum likelihood.
    """
    from tabulate import tabulate
    table_headers = ['Subject', 'Shape family']
    if criterion == 'AIC':
        table_headers.append('dAIC')
    elif criterion == 'ML':
        table_headers.append('Likelihood ratios')
    else:
        raise ValueError('Wrong criterion given')

    subj, fam, daic = _get_data_priors(subjects, criterion=criterion)
    table_data = []
    for ix_sub, sub in enumerate(subj):
        table_data.append([sub, fam[ix_sub], daic[ix_sub]])
    my_table = tabulate(table_data, table_headers)
    print(my_table)
    with open('./table_aics.txt', 'w') as mafi:
        mafi.write(my_table)


def figure_effect_alpha(subject=0, alphas=(1, 10), shape_pars=None, fignum=101, maax=None):
    """ Paper figure.
    Plots the effect of different values of alpha on the posteriors over actions for
    all observations in the data for a given subject and model.
    """

    if shape_pars is None:
        shape_pars = ['unimodal_s', 0, 2]

    q_file = './data/qus_subj_%s_%s.pi' % (subject, shape_pars[0])

    if maax is None:
        plt.figure(fignum, figsize=[6, 4])
        plt.clf()
        ax = plt.axes()
    else:
        ax = maax

    posta = {}
    for alpha in alphas:
        posta_dict = invp.calculate_posta_from_Q(
            alpha, q_file, guardar=False, regresar=True)
        posta[alpha] = []
        for key in posta_dict:
            if key[:-3] == tuple(shape_pars[1:]):
                posta[alpha].append(posta_dict[key][0])
        posta[alpha] = np.array(posta[alpha])
        try:
            ax.plot(posta[alpha][:, 1], label=r'$\alpha=%s$' % alpha)
        except IndexError:
            raise IndexError('Seems like the given shape_pars was not found in ' +
                             'the dictionary. Probably bad shape_pars.')

    ax.axis([0, len(posta[alpha][:, 0]), 0, 1])
    ax.set_xlabel('All observations in one block')
    ax.set_ylabel('Probability of choosing risky')

    ax.set_label('Label via method')
    ax.legend()

    if maax is None:
        plt.show()
    return posta


def figure_shapes(number_shapes=5, fignum=102):
    """Plots the goal shapes."""
    from matplotlib import cm
    cmap = figure_colors('lines_cmap')

    def plot_shape(shape_to_plot, maax, label=None, color='black'):
        """Plots the shape and the threshold line on the --ax-- provided."""
        shape_to_plot /= shape_to_plot.max()
        ax.plot(shape_to_plot, label=label, color=color)
        if maax.is_first_col() and ax.is_last_row():
            maax.set_xticks(range(0, mabes.nS, np.floor(
                np.floor(mabes.nS / 5 / 10) * 10).astype(int)))
            maax.set_xlabel('Accumulated points')
            maax.set_yticks([])
            maax.set_ylabel('Valuation (a.u.)')
        else:
            maax.set_xticks([])
            maax.set_yticks([])
            maax.set_ylabel('')
        maax.set_ylim([0, 1.2])
    mabes = bc.betMDP(nS=72, thres=60)

    shape_pars_all = invp.__rds__()

    s1, s2 = calc_subplots(len(shape_pars_all) + 1)
    mafig = plt.figure(fignum, figsize=[6, 4])
    plt.clf()

    labels_pars = [[r'$\mu$', r'$\sigma$'], ['k', r'$x_0$'], ['a']]

    for subp in range(len(shape_pars_all) + 1):
        ax = plt.subplot(s1, s2, subp + 1)
        if subp == 0:
            shape_to_plot = mabes.C[:mabes.nS]
            plot_shape(shape_to_plot, ax, color=cmap(1.0))
        else:
            prod_pars = list(itertools.product(*shape_pars_all[subp - 1][1:]))
            plots_indices = np.linspace(
                0, len(prod_pars), number_shapes, endpoint=False, dtype=int)
            for ix, index in enumerate(plots_indices):
                shape_pars = list(prod_pars[index])
                shape_pars.insert(0, shape_pars_all[subp - 1][0])
                label = ['%s = %2.0f' % (
                    labels_pars[subp - 1][x], shape_pars[x + 1])
                    for x in range(len(shape_pars[1:]))]
                label_final = ''
                for label_part in label:
                    label_final += label_part + ', '
                label_final = label_final[:-2]

                shape_to_plot = mabes.set_prior_goals(
                    shape_pars=shape_pars, convolute=False, cutoff=False, just_return=True)
                plot_shape(shape_to_plot, ax, label=label_final,
                           color=cmap(ix / number_shapes))
    all_axes = mafig.get_axes()
    labels = 'ABCD'
    for nplot, ax in enumerate(all_axes):
        shaded_threshold = np.arange(mabes.thres, mabes.nS)
        bg_color = figure_colors('past_threshold_background')
        ax.fill_between(shaded_threshold, 0, ax.get_ylim()
                        [-1], alpha=0.1, color=bg_color)
        ax.set_title(labels[nplot], loc='left')
        ax.set_label('Label via method')
        if nplot != 0:
            ax.legend(fontsize=6, loc='upper left')
    plt.tight_layout()
    plt.show(block=False)


def figure_likelihood_map(subjects=(0,), number_shapes=5, fignum=103):
    """Paper figure.
    Plots the best --number_shapes-- shapes for each subject, as defined by their likelihood.
    All shapes are plotted in black, while their transparency is determined by their
    likelihood. The highest likelihood will have a transparency of 1, and the rest drop
    quadratically (for effect).

    The \alpha parameter is sadly ignored here.
    """
    if isinstance(subjects, int):
        subjects = subjects,

    s1, s2 = calc_subplots(len(subjects))

    mafig = plt.figure(fignum, figsize=[6, 4])
    plt.clf()

    mabes = bc.betMDP(nS=72, thres=60)
    color_shapes = {'unimodal_s': '#525fb9', 'sigmoid_s': '#3f923a',
                    'exponential': '#6b4f3f'}
    for ix_sub, subject in enumerate(subjects):
        maax = plt.subplot(s1, s2, ix_sub + 1)
        if len(subjects) < 11:
            maax.set_title('ABCDEFGHIJK'[ix_sub], loc='left', fontsize=12)
        else:
            maax.set_title('%s' % subject, loc='left', fontsize=12)
        loglis, keys = rank_likelihoods(
            subject, number_save=number_shapes, force_rule=True)
        likelis = np.exp(loglis - loglis.max())
        for ix_likeli, likeli in enumerate(likelis):
            shape_pars = keys[ix_likeli][1]
            if not rules_for_shapes(shape_pars):
                continue
            le_shape = mabes.set_prior_goals(shape_pars=shape_pars, just_return=True,
                                             convolute=False, cutoff=False)
            le_shape /= le_shape.max()
            maax.plot(le_shape, alpha=likeli**2,
                      color=color_shapes[shape_pars[0]])

    background_color = figure_colors('past_threshold_background')
    for maax in mafig.get_axes():
        if not maax.is_last_row():
            maax.set_xticks([])
        else:
            maax.set_xlabel('Accumulated points', fontsize=12)
            maax.set_ylabel('Valuation (a.u.)', fontsize=12)
        maax.set_yticks([])
        maax.set_xlim([0, 72])
        maax.fill_between(np.arange(60, 72.01, 0.1), 0,
                          1, alpha=0.2, color=background_color)
    plt.tight_layout()
    plt.show(block=False)


def figure_decisions(subject, shape_pars_sim=None, fignum=104):
    """Plots, for the given subject, the best agent's posteriors
    alongside the choices made by the subject. Below it, the same thing
    for a simulated agent, given by shape_pars_sim.
    """
    fig = plt.figure(fignum, figsize=[6, 4])
    fig.clf()

    maax = plt.subplot(2, 1, 1)
    alpha, shape_pars, posteriors = fitted_decisions(subject, maax=maax)
    # Sample new decisions from posteriors
    posteriors = np.squeeze(posteriors)
    deci = np.zeros(posteriors.shape[0])
    for ix_posta, posta in enumerate(posteriors):
        deci[ix_posta] = np.random.choice([0, 1], p=posta, size=1)[0]
    maax = plt.subplot(2, 1, 2)
    _plot_decisions(maax, shape_pars, posteriors, deci)
    label_pos = figure_fonts('subplot title position')
    for ix_ax, maax in enumerate(fig.get_axes()):
        # maax.set_title('AB'[ix_ax], loc='left')
        maax.text(label_pos[0], label_pos[1], 'AB'[ix_ax],
                  size=figure_fonts('subplot title'), transform=maax.transAxes)
    plt.tight_layout()
    plt.show(block=False)


def figure_rp_vs_risky(subjects=(0,), fignum=105):
    """ asdf"""
    if isinstance(subjects, int):
        subjects = subjects,
    nsubs = len(subjects)

    fig = plt.figure(fignum, figsize=[6, 6])
    fig.clf()

    outer_grid = gs.GridSpec(1, nsubs)

    fontdict = {'family': 'times new roman',
                'size': 12}
    AB = 'AB'
    for ix_subj, subject in enumerate(subjects):
        inner_grid = gs.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_grid[ix_subj], hspace=0, height_ratios=[1, 1, 0.5])
        maax = plt.Subplot(fig, inner_grid[0])
        posta_one = _plot_rp_vs_risky_own(subject, [maax], regresar=True)
        posta_one_s = {subject: posta_one[subject]}
        avg_one, std_one = _plot_average_risk_dynamics(
            posta_one_s, [maax], legend='Own contexts', regresar=True)
        maax.set_xticks([])
        maax.set_ylim([0, 1])
        if ix_subj == 0:
            maax.set_yticks([0.25, 0.5])
            maax.set_ylabel('P(risky)')
        else:
            maax.set_yticks([])
        maax.set_title('Subject %s' % AB[ix_subj])
        maax.text(0.5, 0.9, 'Own observations')
        fig.add_subplot(maax)

        maax = plt.Subplot(fig, inner_grid[1])
        posta_all = _plot_rp_vs_risky_all(subject, [maax], regresar=True)
        posta_all_s = {subject: posta_all[subject]}
        avg_all, std_all = _plot_average_risk_dynamics(
            posta_all_s, [maax], legend='All contexts', regresar=True)
        maax.set_xticks([])
        maax.set_ylim([0, 1])
        if ix_subj == 0:
            maax.set_yticks([0.25, 0.5])
            maax.set_ylabel('P(risky)')
        else:
            maax.set_yticks([])
        maax.text(0.5, 0.9, 'All observations')
        fig.add_subplot(maax)

        maax = plt.Subplot(fig, inner_grid[2])
        avg_diff = np.abs(np.array(avg_one) - np.array(avg_all))
        maax.bar(np.arange(35), avg_diff)
        maax.set_xlabel('Risk pressure', fontdict=fontdict)
        maax.set_ylim([0, 0.3])
        if ix_subj == 0:
            maax.set_ylabel('Difference\nin mean', fontdict=fontdict)
            maax.set_yticks([0, 0.1, 0.2])
        else:
            maax.set_yticks([])
        fig.add_subplot(maax)

    #_rp_vs_risky_pretty(fig)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    plt.show(block=False)


def figure_one_agent_many_obs(subjects=(0, 1), fignum=106):
    """Each agent is exposed to the observations of all others."""
    posta_all = _one_agent_many_obs(subjects=subjects)
    # with open('./posta_all.pi', 'rb') as mafi:
    #     posta_all = pickle.load(mafi)

    s1, s2 = calc_subplots(len(subjects))
    fig = plt.figure(fignum)

    for ix_sub, subject in enumerate(subjects):
        maax = plt.subplot(s1, s2, ix_sub + 1)
        _rp_vs_risky_subplot(
            posta_all[subject]['rp'], posta_all[subject]['posta'], maax)
    _rp_vs_risky_pretty(fig)
    plt.show(block=False)


def figure_colors(element):
    """Returns the standard colors for the requested element. If the requested
    element is not found, returns 'black'.
    """
    from matplotlib import cm
    cmap = cm.get_cmap('Paired')

    if element == 'past_threshold_background':
        return '#c8af42'
    elif element == 'lines_cmap':
        return cmap
    elif element == 'one line':
        return cmap(0.5)
    else:
        return 'black'


def figure_fonts(element):
    """Returns the standard font size for the requested element. If the requested
    element is not found, returns 12. If - -element - - is 'font', the name of the
    font is returned as a string.
    """
    if element == 'subplot title':
        return 20
    elif element == 'axes labels':
        return 12
    elif element == 'axes ticks':
        return 10
    elif element == 'font':
        return 'Times New Roman'
    elif element == 'subplot title position':
        return [-0.1, 1]
    else:
        return 12
