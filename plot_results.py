#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:09:19 2017

@author: dario
"""

#%%
from matplotlib import pyplot as plt
import numpy as np
from utils import calc_subplots

def plot_likelihoods(likeli, fignum = 1, ax = None):
    """ Plots likelihoods over models."""
    raise NotImplementedError('This function was messed up during the '
                              + 'hurricane. Needs fixing.')
    pltx, plty = calc_subplots(len(likeli))
    if fignum is not None:
        plt.figure(fignum)
        plt.clf
    if ax is None:
        ax = plt.gca()
    subs = list(likeli.keys())
    for i in range(1,pltx*plty+1):
        plt.subplot(pltx, plty, i)
        ax.imshow(likeli[subs[i-1]], aspect = 0.2)
        imshow.tick_params(width=0.01)
        plt.set_cmap('gray_r')
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\mu$')
        plt.xticks([0,5,10], [0,5,10])
        plt.yticks(np.arange(0,60,10), np.arange(-15,45,10))

    #    plt.colorbar()
    plt.suptitle('Likelihood for all models')

def risk_pressure(data, rmv_after_thres = True):
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
    for d,datum in enumerate(data):
        state = datum['obs']
        thres = datum['threshold']
        reihe = datum['reihe']
        state = np.round(state/10).astype(int)
        thres = np.round(thres/10).astype(int)
        if rmv_after_thres:
            _, trial, obs, thres, reihe = rat(datum['choice'], datum['trial'],
                                       state, thres, reihe)
        else:
            obs = state
            trial = datum['trial']
        risk_p[d] = (thres - obs)/(8 - trial)
        risk_p[d][risk_p[d]<0] = 0

    return risk_p



def plot_rp_pr(posta, mu, sd, fignum = 2):
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
    import import_data as imda

    _, data = imda.main()

    rp = risk_pressure(data)

    nSubjects = len(posta)
    a1, a2 = calc_subplots(nSubjects)

    if fignum is not None:
        plt.figure(fignum)
        plt.clf
    for s in range(nSubjects):
#        plt.subplot(a1, a2, s+1)
        assert posta[s][mu,sd,:][:,1].shape == rp[s].shape
        plt.plot(rp[s], posta[s][mu,sd,:][:,1], '.')
    if fignum is not None:
        plt.title(r'Actinf''s p(risky) for $\mu = %d, \sigma = %d$' %
                  (np.arange(-15,45)[mu], np.arange(1,15)[sd]))
        plt.xlabel('Risk pressure (as per Kolling_2014)')
        plt.ylabel('p(risky) for ActInf')
    else:
        plt.title(r'$\mu = %d, \sigma = %d$' %  (np.arange(-15,45)[mu],
                                                 np.arange(1,15)[sd]))
        plt.xticks([],[])

def plot_all_pr(posta, mu_values, sd_values, fignum = 3):
    """ Creates an array of plots for all mu_values and sd_values (one-to-one,
    not all combinations), using plot_rp_pr().
    """
    assert len(mu_values) == len(sd_values), ("mu_values and sd_values do "+
                                               "not have the same number of "+
                                               "elements")
    a1, a2 = calc_subplots(len(mu_values))
    plt.figure(fignum)
    plt.clf
    for i in range(len(mu_values)):
        plt.subplot(a1, a2, i+1)
        plot_rp_pr(posta, mu_values[i], sd_values[i], None)

    plt.suptitle(r'Actinf''s posterior probability of risky for different'+
                 ' values of $\mu$ and $\sigma$')

def plot_by_thres(shape_pars = None, subjects = None, trials = None, fignum = 4,
                  data_flat = None, priors = None, as_seen = None):
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
    """
    import matplotlib.gridspec as gridspec
    import invert_parameters as invp
    import pickle
    import numpy as np
    import import_data as imda

    if shape_pars is None:
        shape_pars = ['unimodal_s', np.arange(-15,45+1), np.arange(1, 15+1)]

    if data_flat is None:
        _, data_flat = imda.main()
        if subjects is None:
            subjects = range(len(data_flat))
        elif isinstance(subjects, int):
            subjects = subjects,
    else:
        subjects = range(len(data_flat))
    if trials is None:
        trials = [0,1,2,3,4,5,6,7,]
#    target_levels = [595, 930, 1035, 1105]
    nSubs = len(subjects)
    nTL = 4 # Number of thresholds in the data
    if priors is None:
        priors = np.ones(4)
    tl1, tl2 = calc_subplots(nTL)
    s2, s1 = calc_subplots(nSubs)
    fig = plt.figure(fignum)
    fig.clf()
    outer_grid = gridspec.GridSpec(s1,s2)
    if as_seen is None:
        data_file = './data/posteriors.pi'
        with open(data_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)
            print('File opened; data loaded.')
    plt.set_cmap('gray_r')

    xlen = shape_pars[2].size
    xvec = shape_pars[2]
#    xlen = sd_range[1] - sd_range[0]
#    xvec = np.arange(*sd_range)
    xticks = [0, int(xlen/2), xlen]
    xticklabels = [xvec[0], xvec[xticks[1]], xvec[-1]]

    ylen = shape_pars[1].size
    yvec = shape_pars[1]
#    ylen = mu_range[1] - mu_range[0]
#    yvec = np.arange(*mu_range)
    yticks = [0, int(ylen/2), ylen]
    yticklabels = [yvec[0], yvec[yticks[1]], yvec[-1]]

    for s in range(nSubs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4,1,
                              subplot_spec = outer_grid[s], hspace=0.0,
                              wspace=0)

        vTL = np.arange(4)[data_flat[s]['TargetLevels'] != 0]

        for th in vTL:
            ax = plt.Subplot(fig, inner_grid[th])
            likeli, _, _, _, _, _ = invp.main(data_type=['threshold', 'pruned'],
                                  threshold=th, shape_pars = shape_pars,
                                  subject=subjects[s],
                                  data_flat = data_flat,
                                  trials = trials, as_seen = as_seen,
                                  return_results=True, normalize = False)

            ax.imshow(np.exp(likeli[subjects[s]])*priors[th], aspect=0.25,
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
#            ax2 = ax.twinx()
#            ax2.yaxis.set_view_interval(0,60)
#            ax.yaxis.set_view_interval(0,60)
#            ax2.imshow(likeli[subjects[s]]*priors[th], aspect=0.13, interpolation='none')
#            ax2.set_ylabel('%d' % target_levels[th])
#            ax2.set_yticks([0,60])
#            ax2.set_yticklabels([])
            fig.add_subplot(ax)
#            fig.add_subplot(ax2)

#    all_axes = fig.get_axes()

    #show only the outside spines

#    for a,ax in enumerate(all_axes):
#        if ax.is_last_row() and ax.is_first_col():
#            ax.set_xticks([1,5,10])
#            ax.set_xlabel(r'$\sigma$')
#            ax.set_xticklabels([1,5,10], fontsize=8)
#            ax.set_yticks([0,30,60])
#            ax.set_ylabel(r'$\mu$', fontsize=6)
#            ax.set_yticklabels([-15,15,45], fontsize=6)
#        else:
#            ax.set_yticks([])
#            ax.set_xticks([])
#        if ax.is_first_row():
#            ax.set_title('Subject: %d' % (a/4), fontsize=6)
#        else:
#            ax.set_title('')
    plt.tight_layout()
    plt.show()

def select_by_rp(rp_lims, trials = None):
    """ Will select the obs that meet the requirement of having a so-and-so
    risk pressure.
    """
    import import_data as imda
    import numpy as np

    rp_min, rp_max = rp_lims
    _, data_flat = imda.main()

    rp = risk_pressure(data_flat, False)
    indices = {}
    for i in rp:
        tmp1 = rp[i] >= rp_min
        tmp2 = rp[i] <= rp_max
        indices[i] = np.logical_and(tmp1, tmp2)
#    raise Exception
    for d, datum in enumerate(data_flat):
        for field in ['threshold','trial','obs','reihe','choice']:
            datum[field] = datum[field][indices[d]]

    plot_by_thres(trials = trials, data_flat = data_flat)



if __name__ == '__main__':
    import invert_parameters as invp

    subjects = [0,1,2,3,4,5,6,7,8,9]
    trials = [0,1,2,3,4,5,6,7]

    likeli, posta, deci, trial, state, mu_sigma = invp.main(data_type =
                ['full','pruned'], mu_range = (-15, 45), sd_range = (1,15),
                subject = subjects, trials = trials, return_results = True)

    plot_likelihoods(likeli)
    plot_rp_pr(posta, 5, 10)

def select_by_doability(rp_lims, trials, data_flat = None, fignum = 5,
                        as_seen = None):
    """ Will select the obs that meet the requirement of having a so-and-so
    risk pressure. These trials will be sent to plot_by_thres for plotting.

    Parameters
    rp_lims: list of duples of floats
        Each element of the list, with its corresponding element of trials,
        determines the range of rp to be used.
    trials: list of ints
        See above.
    """
    import import_data as imda
    import numpy as np
    if data_flat is None:
        _, data_flat = imda.main()
    if isinstance(rp_lims, tuple):
        rp_lims = [rp_lims,]
    if isinstance(trials, int):
        trials = trials,

    rp_min = {}
    rp_max = {}
    for r,rl in enumerate(rp_lims):
        rp_min[r] = rl[0]
        rp_max[r] = rl[1]

    nC = len(rp_lims)
    if len(trials) != nC:
        raise TypeError('The two inputs must have the same length')


    # Keep only those observations where the RP is within the given limits
    rp = risk_pressure(data_flat, False)

    indices = {}
    for s in range(len(data_flat)):
        indices[s] = [0]*len(data_flat[s]['trial'])
    for c in range(nC):
        for i in rp: # one i for every subject in data_flat
            tmp1 = rp[i] >= rp_min[c]
            tmp2 = rp[i] <= rp_max[c]
            indices_rp = np.logical_and(tmp1, tmp2)
            indices_tr = data_flat[i]['trial'] == trials[c]
            indices_mx = np.logical_and(indices_rp,indices_tr)
            indices[i] = np.logical_or(indices[i], indices_mx)
#    raise Exception
    for d, datum in enumerate(data_flat):
        for field in ['threshold','trial','obs','reihe','choice']:
            datum[field] = datum[field][indices[d]]

    # For every subject, check whether there are no observations left for a
    # given threshold. This is needed for plot_by_thresholds():
    for datum in data_flat:
        for tl in range(4): # target levels
            if datum['TargetLevels'][tl] not in datum['threshold']:
                datum['TargetLevels'][tl] = 0

#    return rp
    plot_by_thres(trials = trials, data_flat = data_flat, fignum = fignum,
                  as_seen = as_seen)

def prepare_inputs_doability(trials_left = 1, as_seen = None):
    """ Little 'script' to prepare inputs for select_by_doability."""
    for c in range(1,trials_left + 1):
        rp_lims = []
        trials = range(7,7-c, -1)
        for t in trials:
            rp_lims.append((0.8*c*14, 1.2*c*20))
        select_by_doability(rp_lims, trials, fignum=c, as_seen = as_seen)

def concatenate_subjects():
    """ Concatenates the data from one subject into a single-subject-like
    structure to do empirical priors.

    TODO:
    NOTE: Doing this makes the likelihoods so small that the precision is not
    and everything goes to zero. To use this, I would need to fix that problem.
    """
    import import_data as imda

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

    for field in ['TargetLevels','NumberGames','NumberTrials']:
        data_out[field] = data_flat[0][field]
    return [data_out,]

def concatenate_smart(plot = True, retorno = False, fignum = 6):
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

    import invert_parameters as invp
    import pickle
    from matplotlib import pyplot as plt


    with open('./data/posteriors.pi','rb') as mafi:
        as_seen = pickle.load(mafi)


    logli = {}
    for th in range(4): #Target values
        likeli, _, _, _, _, _ = invp.main(data_type=['threshold'],
                              threshold=th, mu_range=(-15,45),
                              subject = range(35),
                              sd_range=(1,15), as_seen = as_seen,
                              return_results=True)
        logli[th] = 0
        for s in range(len(likeli)):
            logli[th] += np.log(likeli[s])

    if plot:
        plt.figure(fignum)
        for th in range(4):
            plt.subplot(2,2,th+1)
            plt.imshow(logli[th])
            plt.set_cmap('gray_r')
    if retorno:
        return logli
def plot_concatenated_linear(fignum = 7):
    """ Gets loglikelihood from concatenate_smart, applies exponential and
    plots.
    """
    import numpy as np

    logli = concatenate_smart(plot = False, retorno = True)
    likeli = {}
    for key in logli.keys():
        likeli[key] = np.exp(logli[key] - logli[key].max())

    for i in range(4):
        ax = plt.subplot(2,2,i+1)
        ax.imshow(likeli[i], aspect = 0.2, interpolation='none')
        ax.tick_params(width = 0.01)
        ax.set_title('Threshold: %d' % [595,930,1035,1105][i])
    plt.suptitle(r'Likelihood map over all subjects')

def plot_animation_trials(subject = 0, nDT = 2, fignum = 8, data_flat = None,
                          nA = 3, as_seen = None):
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
    import pickle
    import matplotlib.gridspec as gridspec
    import invert_parameters as invp

    if as_seen is None:
        data_file = './data/posteriors.pi'
        with open(data_file, 'rb') as mafi:
            as_seen = pickle.load(mafi)
            print('File opened; data loaded.')

    target_levels = [595, 930, 1035, 1105]
    fig = plt.figure(fignum)
    fig.clf()
    nTL = 4 #number of Target Levels
    all_trials = range(8)
    t2, t1 = calc_subplots(nDT)
    outer_grid = gridspec.GridSpec(t1,t2)
    for i in range(nDT):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4,1,
                              subplot_spec = outer_grid[i],
                              wspace=0.0, hspace=0.0)
        trials = all_trials[i:(i+nA)]
        for th in range(nTL):
            shape_pars = ['unimodal_s', np.arange(-15,45+1), np.arange(1,15+1)]
            loglikeli, _, _, _, _, _ = invp.main(data_type=['threshold', 'pruned'],
                                  threshold=th, shape_pars = shape_pars,
                                  subject=subject,
                                  trials = trials, as_seen = as_seen,
                                  data_flat = data_flat,
                                  return_results=True)
            ax = plt.Subplot(fig, inner_grid[th])
            ax.imshow(np.exp(loglikeli[subject]), aspect=0.2, interpolation='none')
            plt.set_cmap('gray_r')
            ax.tick_params(width=0.01)
            if ax.is_first_row():
                ax.set_title('Trials = %s, Thres = %d' % (list(trials),target_levels[th]),
                                                 fontsize=6)
#            ax.set_yticks([0,15,30,45,60])
#            ax.set_yticklabels([-15,0,15,30,45], fontsize=10)
#            ax.set_xticks([1,5,10])
#            ax.set_xticklabels([1,5,10], fontsize=8)

            fig.add_subplot(ax)
#            plt.adjust_subplots(hspace = 0, vspace = 0)
    all_axes = fig.get_axes()

    for ax in all_axes:
        if ax.is_last_row():
            ax.set_xticks([1,5,10])
            ax.set_xticklabels([1,5,10], fontsize=8)
        else:
            ax.set_xticks([])
        if ax.is_first_col() and ax.is_first_row():
            ax.set_yticks([0,15,30,45,60])
            ax.set_yticklabels([-15,0,15,30,45], fontsize=10)
        else:
            ax.set_yticks([])

    plt.show()


def plot_without_past_threshold(nsubs = 35, fignum = 9, plot = True,
                                data_flat = None):
    """ Remove the trials past-threshold and plot."""
    import import_data as imda
    import numpy as np

    if data_flat is None:
        _, data_flat = imda.main()

    for datum in data_flat:
        indices = np.zeros(len(datum['obs']), dtype = bool)
        for t, th in enumerate(datum['threshold']):
            if datum['obs'][t] < th:
                indices[t] = True
        for field in ['threshold','trial','obs','reihe','choice']:
            datum[field] = datum[field][indices]
    if plot:
        plot_by_thres(fignum = fignum, data_flat = data_flat[:nsubs])
    else:
        return data_flat

def per_mini_block():
    """ Obtains the likelihood for all models on a per-mini-block basis and
    ranks the models based on this.

    For every mini-block, all models aver evaluated and a number of points is
    assigned to each depending on the ranking. At the end, these points are
    added to select the best model.
    """

    import import_data as imda

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
    fields = ['choice','obs','points','rechts','reihe','response',
        'target','threshold','trial']
    for datum in data:
        for field in fields:
            datum[field] = datum[field][mb]
    return data

def dumb_agent():
    """ Play the game with an agent that chooses randomly."""

    import betClass as bc
    PreUpd = False
    printTime = False
    mabe = bc.betMDP(nS = 70, thres = 50)

    from time import time

    t1 = time()
    # Check that the class has been fully initiated with a task:
    if hasattr(mabe, 'lnA') is False:
        raise Exception('NotInitiated: The class has not been initiated'+
                        ' with a task')
    T = mabe.V.shape[1]
    wV = mabe.V   # Working set of policies. This will change after choice
    obs = np.zeros(T, dtype=int)    # Observations at time t
    act = np.zeros(T, dtype=int)    # Actions at time t
    sta = np.zeros(T, dtype=int)    # Real states at time t
    bel = np.zeros((T,mabe.Ns))      # Inferred states at time t
    P   = np.zeros((T, mabe.Nu))
    W   = np.zeros(T)
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
    for t in range(T-1):
        # Sample an observation from current state
        obs[t] = mabe.sampleNextObservation(sta[t])
        # Update beliefs over current state and posterior over actions
        # and precision
#        bel[t,:], P[t,:], Gamma = mabe.posteriorOverStates(obs[t], t, wV,
#                                                PosteriorLastState,
#                                                PastAction,
#                                                PriorPrecision,
#                                                PreUpd = PreUpd)
        bel[t,:] = sta[t]
        P[t,:] = [0.5, 0.5]
        Gamma = mabe.gamma

        if PreUpd is True:
            W[t] = Gamma[-1]
            Pupd.append(Gamma)
        else:
            W[t] = Gamma
        # Sample an action and the next state using posteriors
        act[t], sta[t+1] = mabe.sampleNextState( sta[t], P[t,:])
        # Remove from pool all policies that don't have the selected action
        tempV = []
        for seq in wV:
            if seq[t] == act[t]:
                tempV.append(seq)
        wV = np.array(tempV, dtype = int)
        # Prepare inputs for next iteration
#        PosteriorLastState = bel[t]
#        PastAction = act[t]
#        PriorPrecision = W[t]
    xt = time() - t1
    mabe.Example = {'Obs':obs, 'RealStates':sta, 'InfStates':bel,
                    'Precision':W, 'PostActions':P, 'Actions':act,
                    'xt':xt}
    if PreUpd is True:
        mabe.Example['PrecisionUpdates'] = np.array(Pupd)
    if printTime is True:
        pass
    return mabe

def calculate_best_possible_fit(mu, sd, subjects = [0,], as_seen = None,
                                data_flat = None):
    """ For a given set of parameters, calculates the absolute best fit that
    the data could possibly provide, by assuming a subject that always chooses
    the option with the highest probability according to the model; the
    likelihood is calculated for this model with this agent.
    """

    import betClass as bc
    import import_data as imda
    import pickle
    import invert_parameters as invp
    if as_seen is None:
        with open('./data/posteriors.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)
    if data_flat is None:
        _, data_flat = imda.main()

    TL = data_flat[0]['TargetLevels']

    mabe = {}
    for t, th in enumerate(TL):
        thres = np.round(th/10).astype(int)
        mabe[t] = bc.betMDP(nS = np.ceil(thres*1.2).astype(int), thres = thres)
        mabe[th] = mabe[t]

    posta_inferred = {}
    max_likelihood = {}

    for s in subjects:
        data = data_flat[s]
        deci, trial, state, thres, reihe = (data['choice'], data['trial'],
                                            data['obs'], data['threshold'],
                                            data['reihe'])
        state = np.round(state/10).astype(int)
        thres = np.round(thres/10).astype(int)
        deci, trial, state, thres, reihe = invp._remove_above_thres(deci, trial,
                                                        state, thres, reihe)
        invp._shift_states(state, thres, reihe)
        max_likelihood[s] = 1
        for o in range(len(deci)):
            posta_inferred[(s,o)] = as_seen[(mu, sd, state[o], trial[o], thres[o])]
            max_likelihood[s] *= max(posta_inferred[(s,o)][0])

    return posta_inferred, max_likelihood,

def calculate_enhanced_likelihoods(subjects = [0,], as_seen = None):
    """ Will calculate 'enhanced likelihoods', using the best possible fit
    calculated in calculate_best_possible_fit().

    BAD MATH

    """
    import pickle
    import numpy as np
    import invert_parameters as invp

    if as_seen is None:
        with open('./data/posteriors.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)
    if isinstance(subjects, int):
        subjects = subjects,

    mu_range = (-15,45)
    mu_vec = np.arange(mu_range[0], mu_range[1]+1)
    sd_range = (1,15)
    sd_vec = np.arange(sd_range[0], sd_range[1]+1)

    max_likelihood = {}
    for mu in mu_vec:
        for sd in sd_vec:
            _, ml = calculate_best_possible_fit(mu, sd, subjects,
                                                as_seen = as_seen)
            max_likelihood[(mu,sd)] = ml

    likeli, _, _, _, _, _ = invp.main(data_type=['full'],
                          mu_range=mu_range, sd_range=sd_range, subject=subjects,
                          as_seen = as_seen, return_results=True,
                          normalize = False)
    corrected_likelihood = {}
    for s in range(len(likeli)):
        corrected_likelihood[s] = np.zeros((61,16))
        for m in range(61):
            for d in range(15):
                corrected_likelihood[s][m,d] = likeli[s][m,d]*max_likelihood[(mu_vec[m], sd_vec[d])][s]
    return corrected_likelihood

def create_sim_data(shape_pars):
    """ Creates simulated data with four conditions."""
    import invert_parameters as invp
    import numpy as np


    target_levels = np.array([595, 930, 1035, 1105])
    target_lvls = np.round(target_levels/10).astype(int)

    tmp_data = {}
    for tg in target_lvls:
        mabes, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
                                          invp.simulate_data(shape_pars,
                                          num_games = 12,
                                          nS = np.round(1.2*tg).astype(int),
                                          thres = tg))
        tmp_data[tg] = invp._create_data_flat(mabes, deci, trial, state, thres, nD, nT)

    data_flat = tmp_data[target_lvls[0]]
    for tg in target_lvls[1:]:
        for name in tmp_data[tg][0].keys():
            data_flat[0][name] = np.hstack([data_flat[0][name],tmp_data[tg][0][name]])
    data_flat[0]['NumberGames'] = 48
    data_flat[0]['NumberTrials'] = 8
    return data_flat

def fit_simulated_data(sd_range = (3,10), mu_range = (-15,10), threshold = 55,
                       sim_mu = 55, sim_sd = 5, fignum = 10, games = 20,
                       retorno = False, data_flat = None, as_seen = {}):
    """ Fit the model to simulated data to see if the parameters are recovered"""
    from matplotlib import pyplot as plt
    import invert_parameters as invp

    if data_flat is None:
        likeli, _, _, _, _, _ = invp.main(
            data_type = ['simulated', 'threshold'], sim_mu = 55, sim_sd = 5,
            mu_range = mu_range, sd_range = sd_range, trials = [0,1,2,3,4,5,6,7],
            as_seen = as_seen, return_results = True, threshold = threshold,
            games = games, normalize = False)
    else:
        likeli, _, _, _, _, _ = invp.main(data_type = ['full'],
            mu_range = mu_range, sd_range = sd_range, trials = [0,1,2,3,4,5,6,7],
            as_seen = as_seen, return_results = True, data_flat = data_flat,
            normalize = False)


#    sd_range = (3,10)
    sd_vec = np.arange(sd_range[0],sd_range[1]+1)
#    mu_range = (-15,10)
    mu_vec = np.arange(mu_range[0], mu_range[1]+1)

    y0 = (mu_vec==0).nonzero()[0][0]
    plt.figure(fignum)
    max_logli = likeli[0].max()
    plt.imshow(np.exp(likeli[0]-max_logli), interpolation = 'none')
    plt.set_cmap('gray_r')
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'threshold $+\mu$')
    plt.xticks([0, len(sd_vec)], [sd_vec[0], sd_vec[-1]])
    plt.yticks([0,y0,len(mu_vec)-1], [mu_vec[0],mu_vec[y0], mu_vec[-1]])
    plt.title('Real parameters: \n mu = %d, sd = %d' % (sim_mu, sim_sd))
    if retorno:
        return likeli

def fit_many(thres = 55, sim_mu = [0,15,-15], sim_sd = [3,4,5], games = 48,
             retorno = False, data_flat = None, as_seen = False,
             sd_range = (3,5), mu_range = (-16,16)):
    """ Plot many of fit_simulated_data()."""
    import pickle
    import invert_parameters as invp

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
            likeli[(mu,sd)] = fit_simulated_data(sd_range = sd_range,
                               mu_range = mu_range, data_flat = data_flat,
                               threshold = thres, sim_mu = mu, sim_sd = sd,
                               fignum = k, games = games, retorno = retorno,
                               as_seen = as_seen)
            k += 1
    invp.concatenate_data(old_files = 'delete')
    if retorno:
        return likeli

def plot_final_states(data, fignum = 11):
    """ Bar plot of how many times states were visited in the data."""
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    import utils

    if not isinstance(data, list):
        data = [data]
    nSubs = len(data)

    nObs = len(data[0]['trial'])
    target_levels = data[0]['TargetLevels']

    max_S = 0
    for datum in data:
        max_S = max(max_S,datum['obs'].max())

    count = {}
    for tl in target_levels:
        count[tl] = np.zeros(max_S+1)


    for datum in data:
        for ob in range(nObs):
            cobs = datum['obs'][ob]
            cthr = datum['threshold'][ob]
            count[cthr][cobs] += 1
    fig = plt.figure(fignum)
    plt.clf()
    s2, s1 = utils.calc_subplots(nSubs)
    outer_grid = gridspec.GridSpec(s1,s2)

    for s in range(nSubs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(4,1,
                              subplot_spec = outer_grid[s], hspace=0.0,
                              wspace=0)
        for th in range(len(target_levels)):
            ax = plt.Subplot(fig, inner_grid[th])
            thres = target_levels[th]

    #        ax.bar(np.arange(max_S+1),count[target_levels[th]])
            ax.hist(data[s]['obs'][data[s]['threshold']==thres],20, color = (105/256,141/256,198/256))
            ax.plot([thres, thres], [0, 20], linewidth = 3, color = 'g')
            ax.plot([thres-150, thres-150], [0,20], color = 'g')
            ax.plot([thres+150, thres+150], [0,20], color = 'g')

            xticks = range(0,1400,100)
            xticklabels = range(0,140,10)

            yticks = range(0,20,5)
            yticklabels = yticks

            ax.set_xlim([0,1400])

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
                ax.set_ylabel('Th=%d' % (thres/10))
            if ax.is_first_row():
                ax.set_title('Histogram of visited states in data')
            else:
                ax.set_title('')

            fig.add_subplot(ax)

def plot_evolution_gamma(mu = None, sd = None, num_games = 10,
                         fignum = 12, return_dict = False):
    """ Simulates the data set's observations with different parameter values
    to see whether precision evolves significantly differently for different
    priors.

    """

    import invert_parameters as invp
    from matplotlib import pyplot as plt


    mabe, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
                  invp.simulate_data(nS = 72, thres = 60, mu = mu, sd = sd))
#    data_flat = invp._create_data_flat(mabe, deci, trial, state, thres, nD, nT)
#
#    for field in ['choice', 'reihe','obs','threshold','trial']:
#        data_flat[0][field] = np.reshape(data_flat[0][field],(-1,nT))
#    data = data_flat # Since it is not flat anymore

    deci = np.reshape(deci, (-1,nT))
    state = np.reshape(state, (-1,nT))

    mu_vec = np.arange(-15,30+1)
    sd_vec = np.arange(1,15+1)
    mu_size = len(mu_vec)
    sd_size = len(sd_vec)
    try:
        nGames = data_flat[0]['choice'].shape[0]
    except:
        nGames = deci.shape[0]

    results = {}
    fig = plt.figure(fignum)
    for m,mu in enumerate(mu_vec):
        for sd,sigma in enumerate(sd_vec):
            for g in range(nGames):
                shape_pars = [mu, sigma]
                mabe.set_prior_goals(selectShape = 'unimodal_s',
                                     shape_pars=shape_pars)
                results[(mu,sigma,g)] = mabe.full_inference(sta_f = state[g],
                        act_f = deci[g], just_return = True)
                plt.subplot(mu_size,sd_size,m*sd_size + sd +1)
                plt.plot(results[(mu,sigma,g)]['Precision'])

    all_axes = fig.get_axes()
    for x,ax in enumerate(all_axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if ax.is_first_row():
            ax.set_title('%s' % sd_vec[x])
        if ax.is_first_col():
            ax.set_ylabel('%s' % mu_vec[ax.rowNum], fontsize=8)

    if return_dict is True:
        return results

def plot_lnc(mu_vec = None, sd_vec = None, fignum = 13):
    """ Plots an array of exp(lnC) for the values of mu and sd provided."""
    from matplotlib import pyplot as plt
    import betClass as bc
    import numpy as np
    from matplotlib import gridspec as gs


    target_levels = [60,93,104,110]
    mabes = {}
    for tl in target_levels:
        mabes[tl] = bc.betMDP(nS = np.round(tl*1.2).astype(int), thres = tl)
#    mabes = bc.betMDP()
    fig = plt.figure(fignum)
    fig.clf()
    if mu_vec is None:
        mu_vec = np.arange(-15,45+1)
    if sd_vec is None:
        sd_vec = np.arange(1,15+1)
    outer_grid = gs.GridSpec(len(mu_vec), len(sd_vec))
    for m, mu in enumerate(mu_vec):
        for sd, sigma in enumerate(sd_vec):
            for l,tl in enumerate(target_levels):
                inner_grid = gs.GridSpecFromSubplotSpec(4,1,
                                          subplot_spec = outer_grid[m,sd],
                                          hspace = 0, wspace = 0)
                shape_pars = [mu, sigma]
                clnC = mabes[tl].set_prior_goals(selectShape = 'unimodal_s',
                                            shape_pars = shape_pars,
                                            just_return=True,
                                            cutoff = False, convolute = False)
                ax = plt.Subplot(fig, inner_grid[l])
                ax.fill_between([0,mabes[tl].nS],[0,0],[clnC.max(), clnC.max()], color=(0.9, 0.9, 0.9))

                ax.plot(clnC, color=(0.2,0.5,0.2))
#                ax.set_ylabel('%s' % tl, rotation=0, ha = 'right')
                ax.set_yticks([])
                ax.set_xlim([0,140])
                ax.plot([tl, tl],[0, clnC.max()], color=(0.7,0.3,0.3), linewidth=2)
                if not ax.is_last_row():
                    ax.set_xticks([])
                if sd == 0 and ax.is_first_row():
                    ax.set_ylabel(r'$\mu = $%s' % mu, rotation=0, ha='right')
                if m == 0 and ax.is_first_row():
                    ax.set_title(r'$\sigma = $%s' % sigma)
                fig.add_subplot(ax)

def plot_performance(shape_pars, nGames = 10, fignum = 14, nS = 72, thres = 60):
    """ Plots a matrix containing the performance of the model for all the
    parameter values contained in shape_pars
    """

    from matplotlib import pyplot as plt
    import itertools as it
    import betClass as bc

    mabe = bc.betMDP(nS = nS, thres = thres)
#
#    mabe, deci, trial, state, thres, posta, preci, stateV, nD, nT = (
#                  invp.simulate_data(num_games = nGames, nS = 72, thres = 60))
#
#    deci = np.reshape(deci, (-1,nT))
#    state = np.reshape(state, (-1,nT))

#    nGames = deci.shape[0]
    sizes_pars = tuple([len(x) for x in shape_pars[1:]])
    big_index = it.product(*shape_pars[1:])
    results = {}
    fig = plt.figure(fignum)
    success = np.zeros(sizes_pars)
    for i,index in enumerate(big_index):
        for g in range(nGames):
            shape_pars_it = index
            mabe.set_prior_goals(selectShape = 'unimodal_s',
                                 shape_pars=shape_pars_it, cutoff = False)
            tmp = list(index)
            tmp.append(g)
            results_index = tuple(tmp)
            results[results_index] = mabe.full_inference(just_return = True)
#            print('Final state', results[results_index]['RealStates'][-1]%nS)
            if results[results_index]['RealStates'][-1]%nS >= mabe.thres:
#                print('Index:',i, index)
                success[np.unravel_index(i,sizes_pars)] += 1
#                raise Exception
    plt.imshow(success, interpolation='none')
    plt.set_cmap('gray_r')

#    return results, success


def plot_inferred_shapes(data, as_seen, shape_pars, shape_pars_r = None, fignum=16):
    """ Plots the shapes of the inferred parameters.
    shape_pars_r, if given, is used to plot the shape used for simulating data.
    """
    import invert_parameters as invp
    import betClass as bc
    from matplotlib import pyplot as plt
    
    target_levels = [595,930,1035,1105]
    
    mabes = {}
    for lvl in target_levels:
        thres = np.round(lvl/10).astype(int)
        nS = np.round(1.2*thres).astype(int)
        mabes[lvl] = bc.betMDP(thres = thres, nS = nS)

    
