#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:09:19 2017

@author: dario
"""
#%%
from matplotlib import pyplot as plt
import numpy as np

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

def risk_pressure(data):
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
        _, trial, obs, thres, reihe = rat(datum['choice'], datum['trial'],
                                   state, thres, reihe)
        risk_p[d] = (abs(thres - obs))/(8 - trial)
    return risk_p

def calc_subplots(nSubjects):
    """ Calculates a good arrangement for the subplots given the number of
    subjects.
    # TODO: Make it smarter for prime numbers
    """
    sqrtns = nSubjects**(1/2)
    if abs(sqrtns - np.ceil(sqrtns))< 0.001:
        a1 = a2 = np.ceil(sqrtns)
    else:
        divs = nSubjects%np.arange(2,nSubjects)
        divs = np.arange(2,nSubjects)[divs==0]
        if divs.size==0:
            a1 = np.ceil(nSubjects/2)
            a2 = 2
        else:
            a1 = divs[np.ceil(len(divs)/2).astype(int)]
            a2 = nSubjects/a1
    return int(a1), int(a2)

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

def plot_by_thres(subjects = None, trials = None, fignum = 4, data_flat = None):
    """ Calculate and plot the likeli for the given subjects, separating the
    different thresholds into sub-subplots.

    The function can be called by specifying subjects and trial numbers to use
    or it can be given data as input. Giving data as input overrides the other
    parameters.
    """
    import matplotlib.gridspec as gridspec
    import invert_parameters as invp
    import pickle

    if subjects is None:
        subjects = [0,1,2,3,4,5,6,7,8,9]
    elif isinstance(subjects, int):
        subjects = subjects,

    if trials is None:
        trials = [0,1,2,3,4,5,6,7,]
    target_levels = [595, 930, 1035, 1105]
    nSubs = len(subjects)
    nTL = 4 # Number of thresholds in the data

    tl1, tl2 = calc_subplots(nTL)
    s2, s1 = calc_subplots(nSubs)
    fig = plt.figure(fignum)
    fig.clf()
    outer_grid = gridspec.GridSpec(s1,s2)
    data_file = './data/posteriors.pi'
    with open(data_file, 'rb') as mafi:
        as_seen = pickle.load(mafi)
        print('File opened; data loaded.')
    plt.set_cmap('gray_r')

    for s in range(nSubs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(tl1,tl2,
                              subplot_spec = outer_grid[s])
        for th in range(nTL):
            ax = plt.Subplot(fig, inner_grid[th])
            likeli, _, _, _, _, _ = invp.main(data_flat,
                                  data_type=['threshold', 'pruned'],
                                  threshold=th, mu_range=(-15,45),
                                  sd_range=(1,15), subject=subjects[s],
                                  trials = trials, as_seen = as_seen,
                                  return_results=True)
#            plot_likelihoods(likeli, None, ax)
            ax.imshow(likeli[subjects[s]], aspect=0.2, interpolation=None)
            ax.tick_params(width=0.01)
            ax.set_title('S = %d, Thres = %d' % (s,target_levels[th]),
                                                 fontsize=10)
            ax.set_yticks([0,15,30,45,60])
            ax.set_yticklabels([-15,0,15,30,45], fontsize=10)
            ax.set_xticks([1,5,10])
            ax.set_xticklabels([1,5,10], fontsize=8)

            fig.add_subplot(ax)

    all_axes = fig.get_axes()

    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
            ax.set_xlabel(r'$\sigma$')
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.set_ylabel(r'threshold + $\mu$')
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    plt.show()


def select_by_rp(rp_min, rp_max):
    """ Will select the obs that meet the requirement of having a so-and-so
    risk pressure.
    """
    import import_data as imda

    _, data_flat = imda.main()

    rp = risk_pressure(data_flat)
    indices = {}
    for i,r in enumerate(rp):
        indices[i] = r >= rp_min and r <= rp_max

    for d, datum in enumerate(data_flat):
        for field in ['threshold','trial','obs','reihe','choice']:
            datum[field] = datum[field][indices[d]]

    plot_by_thres(trials = [6,7], data_flat = data_flat)
    
    

def select_by_random(p_value = 0.05):
    """ Will select the obs that are not significantly different from random
    choice.
    """
    raise NotImplementedError('Not yeeet')

def select_by_precision(precision_threshold):
    """ Will select the obs for which the actinf agent has very high precision.
    """
    raise NotImplementedError('Needs more thinking')
    import import_data as imda

    _, data_flat = imda.main()

    deci, trial, state, thres, reihe = (data['choice'], data['trial'],
                                            data['obs'], data['threshold'],
                                            data['reihe'])
    state = np.round(state/10).astype(int)
    thres = np.round(thres/10).astype(int)
    deci, trial, state, thres, reihe = _remove_above_thres(deci, trial,
                                                        state, thres, reihe)
    target_levels = np.round(data['TargetLevels']/10)










    
if __name__ == '__main__':
    import invert_parameters as invp

    subjects = [0,1,2,3,4,5,6,7,8,9]
    trials = [0,1,2,3,4,5,6,7]

    likeli, posta, deci, trial, state, mu_sigma = invp.main(data_type =
                ['full','pruned'], mu_range = (-15, 45), sd_range = (1,15),
                subject = subjects, trials = trials, return_results = True)

    plot_likelihoods(likeli)
    plot_rp_pr(posta, 5, 10)
