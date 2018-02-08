# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import import_data as imda
import plot_results as pr
from utils import calc_subplots

plt.rc('text', usetex=True)


def data_dv_vs_prisky_exp_all(subjects=None, bins=3, dv_range=None, use_dv=True,
                              rmv_after_thres=False):
    """Bins the data for all subjects to calculate the average p(risky)
    for each of the bins.
    """
    if subjects is None:
        subjects = range(35)

    _, flata = imda.main()
    if use_dv:
        delta_v = pr.calc_delta_v(flata, rmv_after_thres=rmv_after_thres)
    else:
        delta_v = pr.calc_risk_pressure(flata, rmv_after_thres=rmv_after_thres)
    if dv_range is None:
        random_s = np.random.choice(subjects)
        dv_range = (delta_v[random_s].min(), delta_v[random_s].max())
    if isinstance(bins, int):
        dv_bins = np.linspace(dv_range[0], dv_range[1], bins)
    else:
        dv_bins = bins
        bins = len(dv_bins)

    # concatenate data
    delta_v_all = np.array([])
    choices_all = np.array([])
    for subject in subjects:
        delta_v_all = np.hstack(
            [delta_v_all, delta_v[subject]])
        choices_all = np.hstack([choices_all, flata[subject]['choice']])

    binned_rp = np.digitize(delta_v_all, dv_bins)
    choices = [choices_all[np.where(binned_rp == x)[0]]
               for x in range(1, bins + 1)]
    conf_intervals = [pr.confidence_intervals_prisky(
        choice) for choice in choices]
    prisky = np.array([np.mean(choice) for choice in choices])
    return prisky, dv_bins, conf_intervals


def plot_rp(subjects=None, bins=4, rp_range=None, rmv_after_thres=False, color=None,
            maax=None, fignum=3):
    """Plots data_dv_vs_prisky_exp_all(use_dv=False)."""
    if color is None:
        color = 'black'
    if maax is None:
        fig = plt.figure(fignum)
        fig.clear()
        maax = fig.add_subplot(111)
    prisky, dv_bins, conf_intervals = data_dv_vs_prisky_exp_all(
        subjects, bins, rp_range, False, rmv_after_thres)
    conf_intervals = np.array(conf_intervals)[:, 1] - prisky
    maax.errorbar(dv_bins, prisky, yerr=conf_intervals, color=color)
    xlim = maax.get_xlim()
    maax.plot(xlim, [0.5, 0.5], color='black', linewidth=2)
    maax.set_ylim(np.array(maax.get_ylim()) + np.array([-0.1, 0.1]))
    maax.set_xlabel(r'$\Delta V = V_{risky} - V_{safe}$')
    maax.set_ylabel('P(risky)')
    if bins == 4:
        labels = ['Low', 'High']
        maax.set_xticks(dv_bins[[0, -1]])
        maax.set_xticklabels(labels)
    plt.show(block=False)


def plot_dv(subjects=None, bins=4, dv_range=None, rmv_after_thres=False, color=None,
            maax=None, fignum=2):
    """Plots data_dv_vs_prisky_exp_all()."""
    if color is None:
        color = 'black'
    if maax is None:
        fig = plt.figure(fignum)
        fig.clear()
        maax = fig.add_subplot(111)
    prisky, dv_bins, conf_intervals = data_dv_vs_prisky_exp_all(
        subjects, bins, dv_range, rmv_after_thres)
    conf_intervals = np.array(conf_intervals)[:, 1] - prisky
    maax.errorbar(dv_bins, prisky, yerr=conf_intervals, color=color)
    xlim = maax.get_xlim()
    maax.plot(xlim, [0.5, 0.5], color='black', linewidth=2)
    maax.set_ylim(np.array(maax.get_ylim()) + np.array([-0.1, 0.1]))
    maax.set_xlabel(r'$\Delta V = V_{risky} - V_{safe}$')
    maax.set_ylabel('P(risky)')
    if bins == 4:
        labels = ['Low', 'High']
        maax.set_xticks(dv_bins[[0, -1]])
        maax.set_xticklabels(labels)


def linear_regression(subjects=None, regress_trial=True):
    """Does linear regression on the experimental data from imda, using the
    regressors specified in --regressors--.

    Parameters
    ----------
    regress_trial : bool
        If True, trial number is taken into account too.
    """
    from invert_parameters import _remove_above_thres as rat
    import pandas as pd
    import statsmodels.formula.api as sm

    if subjects is None:
        subjects = range(35)

    _, flata = imda.main()

    risk_pressure = pr.calc_risk_pressure(flata)
    delta_v = pr.calc_delta_v(flata)

    risk_pressure_all = np.array([])
    delta_v_all = np.array([])
    choices_all = np.array([])
    trials_all = np.array([])
    for subject in subjects:
        risk_pressure_all = np.hstack(
            [risk_pressure_all, risk_pressure[subject]])
        delta_v_all = np.hstack([delta_v_all, delta_v[subject]])
        indices = rat(flata[subject]['choice'], flata[subject]['trial'],
                      flata[subject]['obs'], flata[subject]['threshold'],
                      flata[subject]['reihe'])[-1]

        choices_all = np.hstack(
            [choices_all, flata[subject]['choice'][indices]])
        trials_all = np.hstack([trials_all, flata[subject]['trial'][indices]])

    choices_all[choices_all == 0] = -1
    risk_pressure_all = (risk_pressure_all -
                         risk_pressure_all.mean()) / risk_pressure_all.std()
    delta_v_all = (delta_v_all - delta_v_all.mean()) / delta_v_all.std()
    trials_all = (trials_all - trials_all.mean()) / trials_all.std()

    data = pd.DataFrame(
        {'cte': -np.ones(len(risk_pressure_all)), 'rp': risk_pressure_all,
         'dv': delta_v_all, 'ch': choices_all, 'tr': trials_all})
    formula = 'ch ~ dv%s + rp' % (" + tr" * regress_trial)
    results = sm.glm(formula=formula, data=data).fit()
    return results


def plot_linear_regression(subjects=None, regress_trial=True, color=None,
                           maax=None, fignum=1):
    """Plots the results from linear_regression()."""
    if maax is None:
        fig = plt.figure(fignum)
        fig.clear()
        maax = fig.add_subplot(111)

    if color is None:
        color = 'black'

    results = linear_regression(subjects, regress_trial)

    bar_heights = results.params.values
    if regress_trial:
        bar_labels = ['Choice\n bias', r'\Delta V', 'Trial', 'Risk\n pressure']
    else:
        bar_labels = ['Choice\n bias', r'\Delta V', 'Risk\n pressure']

    maax.bar(range(len(bar_heights)), bar_heights,
             tick_label=bar_labels, align='center', color=color,
             ecolor='gray', yerr=results.bse.values, width=0.95)
    plt.show(block=False)
    return results


def plot_decisions_vs_rp(subjects=None, fignum=3):
    """Plots decisions (1 for risky, 0 for safe) as a scatter plot, as a function
    of risk pressure.
    """
    if subjects is None:
        subjects = range(35)

    _, flata = imda.main()

    rps = pr.calc_risk_pressure(flata, rmv_after_thres=False)

    fig = plt.figure(fignum)
    fig.clear()

    subplots = calc_subplots(len(subjects))

    for ix_sub, subject in enumerate(subjects):
        maax = fig.add_subplot(subplots[0], subplots[1], ix_sub + 1)
        maax.scatter(rps[subject], flata[subject]['choice'], alpha=0.1)
    plt.show(block=False)


def context_bias(trials, points, thres):
    """ Calculates the context bias, defined here in terms of the probabilities
    of success if choosing the safe vs. the risky choice at all remaining trials.
    This assumes a fixed reward and probability, defined as the average of the
    ones used in the experiments.
    """
    nT = 8

    p = np.array([0.73, 0.35])
    r = np.array([140, 250])

    try:
        _ = trials[0]
    except:
        trials = [trials, ]
    try:
        _ = points[0]
    except:
        points = [points, ]
    try:
        _ = thres[0]
    except:
        thres = [thres, ]

    prob = -np.ones((len(trials), 2))
    for ct, ctrial in enumerate(trials):
        min_t = np.ceil((thres[ct] - points[ct]) / r)
        prob[ct, :] = np.array([1 - sp.stats.binom.cdf(k=min_t[x], p=p[x], n=nT - ctrial)
                                for x in range(2)])

    return prob
