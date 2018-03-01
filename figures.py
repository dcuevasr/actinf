"""Figures and tables for the paper."""

# import pickle

from itertools import product as prd
import pickle
from os.path import isfile
import copy

import ipdb  # noqa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from tabulate import tabulate


import invert_parameters as invp
import plot_results as pr
import import_data as imda
import betClass as bc
from utils import calc_subplots
import clustering as cl
import kolling_stats as ks
import bias_analysis as ba

SUBJECTS = (1, 31, 5)


def new_data():
    """Create a data set for figure 7, with one threshold, and exactly one
    observation for each trial-RP-AP combination.
    """
    threshold = 60
    max_rp = 35
    max_t = 8

    points = _new_points(threshold, max_rp).reshape(-1, order='f')

    all_combs = np.array(list(prd(np.arange(8), points)))
    points = all_combs[:, 1]
    offers = all_combs[:, 0]
    trials = np.tile(np.arange(max_t), max_rp * 8)

    data_flat = {'NumberGames': 1,
                 'NumberTrials': max_t,
                 'TargetLevels': np.array([threshold * 10]),
                 'obs': points * 10,
                 'reihe': offers + 1,
                 'threshold': np.tile(threshold * 10, len(trials)),
                 'trial': trials,
                 'choice': np.tile(0, len(trials))}

    return data_flat


def _new_points(threshold, max_rp):
    """Gets the points for new_data()."""
    points = np.zeros((8, max_rp))
    for trial in range(8):
        all_rps = (threshold - np.arange(threshold)) / (8 - trial)
        for c_rp in range(max_rp):
            rp_diff = all_rps - c_rp
            rp_diff[rp_diff < 0] = np.inf
            points[trial, c_rp] = np.argmin(rp_diff)
    return points


def alpha_hist(subjects=None, maax=None, fignum=1, color=None, divisors=None,
               shapes=None):
    """Plots a histogram of the values for the alpha parameter for the all
    the given subjects.
    """
    if subjects is None:
        subjects = range(35)

    if maax is None:
        fig = plt.figure(fignum)
        plt.clf()
        maax = plt.gca()

    if color is None:
        color = 'black'

    if divisors is None:
        divisors = 'auto'  # (0, 0.5, 1, 1.5, 2, 3, 50)

    best_pars = ba.best_model(subjects, shapes=shapes)
    alphas = []
    for subject in subjects:
        alphas.append(best_pars[subject][1][0][0])

    hist, bins = np.histogram(alphas, bins=divisors)

    x_tick_ranges = []
    for ix_div, _ in enumerate(bins[:-1]):
        x_tick_ranges.append([bins[ix_div], bins[ix_div + 1]])
    x_labels = []
    for x_tick in x_tick_ranges[:-1]:
        x_labels.append('[%2.1f, %2.1f)' % (x_tick[0], x_tick[1]))
    x_labels.append(r'$>$ %2.1f' % bins[-2])

    maax.bar(range(len(hist)), hist, width=1, color=color)
    # maax.set_title(r'Histogram of values of MDM')
    maax.set_title('A', loc='left')
    maax.set_xlabel('Maximum decision multiplier (MDM)')
    maax.set_ylabel(r'Number of subjects')
    maax.set_xticks(np.arange(len(hist)) + 0.5)
    maax.set_xticklabels(x_labels, fontsize=8)
    plt.show(block=False)
    return alphas, hist, bins


def bias_hist(subjects=None, maax=None, fignum=7, color=None, divisors=None,
              shapes=None):
    """Plots a histogram of the values for the alpha parameter for the all
    the given subjects.
    """
    if subjects is None:
        subjects = range(35)

    if maax is None:
        fig = plt.figure(fignum)
        plt.clf()
        maax = plt.gca()

    if color is None:
        color = 'black'

    if divisors is None:
        divisors = 'auto'  # (0, 0.5, 1, 1.5, 2, 3, 50)

    best_pars = ba.best_model(subjects, shapes=shapes)
    bias = []
    for subject in subjects:
        bias.append(best_pars[subject][1][0][1])

    hist, bins = np.histogram(bias, bins=divisors)

    x_tick_ranges = []
    for ix_div, _ in enumerate(bins[:-1]):
        x_tick_ranges.append([bins[ix_div], bins[ix_div + 1]])
    x_labels = []
    for x_tick in x_tick_ranges:
        x_labels.append('[%2.1f, %2.1f)' % (x_tick[0], x_tick[1]))
    # x_labels.append(r'$>$ %2.1f' % bins[-2])

    maax.bar(range(len(hist)), hist, width=1, color=color)
    # maax.set_title(r'Histogram of values of MDM')
    maax.set_title('B', loc='left')
    maax.set_xlabel('Choice bias')
    maax.set_ylabel(r'Number of subjects')
    maax.set_xticks(np.arange(len(hist)) + 0.5)
    maax.set_xticklabels(x_labels, fontsize=8)
    plt.show(block=False)
    return bias, hist, bins


def likelihood_map(subjects=(4, 1), shapes=None,
                   number_shapes=5, force_rule=True,
                   alphas=None, fignum=2):
    """Plots the best --number_shapes-- shapes for each subject,
    as defined by their likelihood.  All shapes are plotted in black, while
    their transparency is determined by their likelihood.  The highest
    likelihood will have a transparency of 1, and the rest drop quadratically
    (for effect).

    The \alpha parameter is sadly ignored here.

    The defaults for --subjects - - are those used in the paper.
    """
    if isinstance(subjects, int):
        subjects = subjects,
    if alphas is None:
        alphas = [None] * len(subjects)

    s1, s2 = calc_subplots(len(subjects))

    mafig = plt.figure(fignum, figsize=[6, 4])
    plt.clf()

    mabes = bc.betMDP(nS=72, thres=60)
    color_shapes = {'unimodal_s': '#4286f4', 'sigmoid_s': '#f48f41',
                    'exponential': '#6b4f3f'}
    for ix_sub, subject in enumerate(subjects):
        alpha = alphas[ix_sub]
        maax = plt.subplot(s1, s2, ix_sub + 1)
        if len(subjects) < 11:
            maax.set_title('ABCDEFGHIJK'[ix_sub], loc='left', fontsize=12)
        else:
            maax.set_title('%s' % subject, loc='left', fontsize=12)
        loglis, keys = pr.rank_likelihoods(
            subject, shapes=shapes, number_save=number_shapes,
            alpha=alpha, force_rule=force_rule)
        likelis = np.exp(loglis - loglis.max())

        for ix_likeli, likeli in enumerate(likelis):
            shape_pars = keys[ix_likeli][1]
            if not invp.rules_for_shapes(shape_pars):
                continue
            le_shape = mabes.set_prior_goals(shape_pars=shape_pars, just_return=True,
                                             convolute=False, cutoff=False)
            le_shape /= le_shape.max()
            maax.plot(le_shape, alpha=likeli ** 2,
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


def plot_best_shape(subjects=None, shapes=None, maax=None, fignum=3):
    """Plots the best shape (within --shapes--, the highest-likelihood
    parameter values) for each subject, on top of each other.
    """
    if subjects is None:
        subjects = range(35)

    if shapes is None:
        shapes = ['unimodal_s']

    if maax is None:
        fig = plt.figure(fignum)
        maax = fig.add_subplot(111)

    best_pars = pr.loop_rank_likelihoods(subjects, shapes=shapes)
    mabe = bc.betMDP(nS=72, thres=50)
    for subject in subjects:
        shape_pars = [best_pars[subject][1][0][-1][0]] + \
            best_pars[subject][1][0][-1][1:]
        lnc = mabe.set_prior_goals(
            shape_pars=shape_pars, convolute=False, cutoff=False, just_return=True)
        lnc /= lnc.max()
        maax.plot(lnc, color='black', alpha=0.5, linewidth=3)
    plt.show(block=False)


def plot_cluster_shapes(subjects=None, maax=None, shape=None, colors=None,
                        centroids=None, labels=None, legends=None, fignum=4):
    """Finds the centers of the clusters in the data and plots the shapes that
    these centers create.
    """
    if subjects is None:
        subjects = range(35)
    if shape is None:
        shape = 'unimodal_s'
    flag_noshow = True
    if maax is None:
        fig = plt.figure(fignum)
        maax = fig.add_subplot(111)
        flag_noshow = False
    if colors is None:
        colors = ['blue', 'red', 'green']
    if labels is None:
        labels = colors
    if legends is None:
        legends = colors

    if centroids is None:
        centroids, _, _ = cl.clustering(subjects, k=3,
                                        clustering_type='kmeans',
                                        shape=shape)

    mabes = bc.betMDP(nS=72, thres=60)
    for ix_cen, centroid in enumerate(centroids):
        shape_pars = [shape] + [par for par in centroid]
        lnc = mabes.set_prior_goals(shape_pars=shape_pars, convolute=False,
                                    cutoff=False, just_return=True)
        lnc /= lnc.max()
        maax.plot(lnc, color=colors[ix_cen],
                  linewidth=3, label=legends[ix_cen])
        maax.set_xlim([0, 71])
        maax.set_xticklabels(np.array(maax.get_xticks(), dtype=int) * 10)

        shaded_threshold = np.arange(mabes.thres, mabes.nS)
        bg_color = figure_colors('past_threshold_background')
        maax.fill_between(shaded_threshold, 0, maax.get_ylim()
                          [-1], alpha=0.1, color=bg_color)
    # maax.set_title('Cluster centers'' shapes')
    maax.set_title('C', loc='left')
    maax.set_xlabel('points')
    maax.set_ylabel('valuation (a.u.)')
    if not flag_noshow:
        plt.show(block=False)


def fake_sim_data(parameters, shape=None):
    """Creates a list --sim_data-- to feed to figure_posta_per_trial(). This
    is to use figure_posta_per_trial() with any shape_pars desired, instead
    of on relying on using --subjects--.

    WILL NOT WORK WITH ANYTHING OTHER THAN EXPONENTIAL.

    Parameters
    ----------
    pars : array-like
        List of parameter values to use. It should be provided in the format
        [[par1_1, par1_2, ...], [par2_1, par2_2, ...], ..., [parn_1, ...]].

    Returns
    -------
    sim_data : list of chars
        List of filenames of the form './data/simulated_posta_per_trial_%02d'
        which is fed to figure_posta_per_trial(). Each filename contains the
        simulated posteriors (in the form of --as_seen--) for the corresponding
        parameter values.
    """
    if shape is None:
        shape = 'exponential'

    if shape != 'exponential':
        raise NotImplementedError('This function only works now with' +
                                  ' the exponential shape.')

    filename_skel = './data/simulated_posta_per_trial_%02d.pi'
    num_pars = len(parameters)
    num_par_vals = len(parameters[0])
    sim_data = []
    shape_pars = []
    for ix_num_par in range(num_pars):
        for ix_num_par_val in range(num_par_vals):
            sim_data.append(filename_skel %
                            parameters[ix_num_par][ix_num_par_val])
            shape_pars.append(
                [shape, [parameters[ix_num_par][ix_num_par_val]]])
    sim_data.append('./data/data_flat_per_trial.pi')

    return sim_data, shape_pars


def _get_posta_all(subjects, data_flat, as_seen_all, shape_pars_all, best_pars,
                   no_calc):
    """For figure_posta_per_trial().

    Gets --posta_all-- for all the subjects, which contains the risk
    pressure for all observations in --data_flat-- and the posteriors."""
    posta_all = {}
    rp = pr.calc_risk_pressure(data_flat)
    for ix_subject, subject in enumerate(subjects):
        as_seen = as_seen_all[subject]
        if shape_pars_all is None:
            shape_pars = invp.switch_shape_pars(
                best_pars[subject][1][0][-1], force='list')
        else:
            shape_pars = shape_pars_all[ix_subject]
        # if ix_subject == 0:
        #     ipdb.set_trace()
        _, posta, _, trial, _, _, le_q = invp.infer_parameters(
            data_flat=data_flat[subject], shape_pars=shape_pars,
            no_calc=no_calc, return_Qs=True, as_seen=as_seen)
        if le_q is {}:
            raise ValueError('qs empty')
            posta = invp.calculate_posta_from_Q(20, le_q,
                                                regresar=True, guardar=False)
        posta_all[subject] = {
            'rp': rp[subject], 'posta': np.squeeze(posta), 'trial': trial}
    return posta_all


def _pretty_plots_per_trial(maaxes, subjects, colors_avgs, colors_dots):
    """Makes the plots in figure_posta_per_trial() prettier by moving around
    the ticks and labels and stuff.
    """
    for ix_maax, maax in enumerate(maaxes):
        maax.set_xlim([0, 35])
        maax.set_ylim([0, 1])
        maax.set_xticklabels(np.array(maax.get_xticks(), dtype=int) * 10)
        maax.set_title('trial: %d' % (ix_maax + 1),
                       loc='left', fontdict={'fontsize': 10})
        if maax.is_last_row():
            maax.set_xlabel('Risk pressure')
        else:
            maax.set_xticks([])
        if maax.is_first_col():
            maax.set_ylabel('P(risky)')
        else:
            maax.set_yticks([])

    # Awesome legends!
    if len(subjects) == 2:
        labels = ['highSTP', 'lowSTP']
        maax = maaxes[0]
        for ix_sub, _ in enumerate(subjects):
            maax.plot([], color=colors_avgs[ix_sub], label=labels[ix_sub])
        maax.legend(fontsize=8)


def trim_data(value, field, flata=None, create_data=False, filename=None):
    """Takes simulated observations and keeps only those that match --offer--."""
    if flata is None:
        if create_data:
            flata = new_data()
        else:
            if filename is None:
                filename = './data/data_flat_per_trial.pi'
            with open(filename, 'rb') as mafi:
                flata = pickle.load(mafi)
    indices = flata[field] == value
    for field in ['choice', 'obs', 'reihe', 'threshold', 'trial']:
        flata[field] = flata[field][indices]
    return flata


def histogram_clusters(subjects=None, shape=None, membership=None,
                       colors=None, labels=None, maax=None, fignum=5):
    """Makes a bar plot for membership of subjects in clusters."""
    if subjects is None:
        subjects = range(35)
    if colors is None:
        colors = ['blue', 'red', 'green']
    if labels is None:
        labels = colors
    if shape is None:
        shape = 'unimodal_s'

    if membership is None:
        _, membership, _ = cl.clustering(subjects, k=3,
                                         clustering_type='kmeans',
                                         shape=shape)
    membership = np.array([membership[key] for key in membership.keys()])
    ix_centroid = np.unique(membership)
    ix_centroid.sort()

    count_members = np.array([len(np.where(membership == x)[0])
                              for x in ix_centroid])

    if maax is None:
        fig = plt.figure(fignum)
        fig.clear()
        maax = fig.add_subplot(111)

    maax.bar(ix_centroid, count_members, width=1,
             color=colors)
    maax.set_xticks(ix_centroid + 0.5)
    maax.set_xticklabels(labels)
    maax.set_ylim(1.2 * np.array(maax.get_ylim()))
    # maax.set_title('Cluster members')
    maax.set_title('C', loc='left')
    maax.set_ylabel('Number of subjects')
    plt.show(block=False)


def scatter_kappas(subjects=None, shape=None, membership=None,
                   colors=None, centroids=None, labels=None, maax=None,
                   fignum=6):
    """Makes a scatter plot of kappa values, coloring them according to
    their membership.
    """
    if subjects is None:
        subjects = range(35)
    if colors is None:
        colors = ['blue', 'red', 'green']
    if labels is None:
        labels = colors
    if shape is None:
        shape = 'exponential'

    if membership is None:
        _, membership, _ = cl.clustering(subjects, k=2,
                                         clustering_type='kmeans',
                                         shape=shape)
    membership = np.array([membership[key] for key in membership.keys()])
    num_clusters = np.unique(membership).size

    if maax is None:
        fig = plt.figure(fignum)
        fig.clear()
        maax = fig.add_subplot(111)

    best_pars = ba.best_model(subjects, shapes=[shape])
    kappa = np.zeros(len(subjects))
    for subject in best_pars.keys():
        kappa[subject] = best_pars[subject][1][0][-1][1]

    counter = 1
    for c_cluster in range(num_clusters):
        c_subs = np.where(membership == c_cluster)[0]
        maax.scatter(range(counter, counter + len(c_subs)), kappa[c_subs],
                     color=colors[c_cluster])
        if centroids is not None:
            maax.plot(range(counter, counter + len(c_subs)),
                      np.tile(centroids[c_cluster], len(c_subs)),
                      color=colors[c_cluster])
        counter += len(c_subs)

    maax.set_xlim(0, len(subjects) + 1)
    maax.set_ylim(0, 110)
    maax.set_yticklabels(np.array(maax.get_yticks() / 10, dtype=int))
    maax.set_xlabel('Subjects')
    maax.set_ylabel('Sensitivity to points (STP)')
    maax.set_title('D', loc='left')
    plt.show(block=False)


def calc_posta_sim(shape=None, shape_pars=None, data_file=None,
                   overwrite=False):
    """Calculate posteriors for the set of simulated data, for all the
    values of the shape parameter for --shape--. The resulting --_as_seen--
    is saved to files with the format './data/simulated_posta_per_trial_%2d.pi'

    Parameters
    ----------
    shape : string \in {'exponential', 'unimodal_s', 'sigmoid_s'}
        Determines which shape family to use. Overwritten by the use of
        --shape_pars--. Defaults to 'exponential'.
    shape_pars : list
        Instead of --shape--, the function can take as input directly
        the shape_pars in the form [shape, par1, par2, ...].
    data_file : string
        File name where the simulated contexts are found. If none is
        provided, it will try to load "./data/data_flat_per_trial.pi". If it
        does't exist, new data will be simulated.
    """
    if shape is None:
        shape = 'exponential'
    if shape_pars is None:
        shape_pars = invp.__rds__(shape)

    alpha = 20

    if data_file is None:
        data_file = './data/data_flat_per_trial.pi'
    try:
        with open(data_file, 'rb') as mafi:
            flata = pickle.load(mafi)
    except FileNotFoundError:
        flata = new_data()

    for par_num in range(len(shape_pars[1])):
        if shape == 'exponential':
            filename = './data/simulated_posta_per_trial_%02d.pi' \
                       % shape_pars[1][par_num]
        else:
            filename = './data/simulated_posta_per_trial_%02d.pi' \
                       % par_num
        if isfile(filename) and not overwrite:
            continue
        c_shape_pars = [shape_pars[0]] + [[shape_pars[x][par_num]]
                                          for x in range(1, len(shape_pars))]
        output = invp.infer_parameters(data_flat=flata, shape_pars=c_shape_pars,
                                       return_Qs=True, as_seen={},
                                       no_calc=False)

        invp.calculate_posta_from_Q(alpha, Qs=output[-1], filename=filename)


def table_best_families(subjects, criterion='AIC'):
    """Creates a table for the paper, in which the best shape is selected for
    each subject.  Subjects are grouped by shapes.  The criterion used can
    either be maximum likelihood, or the Akaike Information Criterion(AIC).

    Parameters
    ----------

    criterion: ['AIC', 'ML', 'BF']
    Criterion to use to decide which family is best for each subject.  AIC
    stands for Akaike information criterion, ML for maximum likelihood.
    """
    table_headers = ['Subject', 'Shape family']
    if criterion == 'AIC' or criterion == 'AICf':
        table_headers.append('dAIC')
    elif criterion == 'ML':
        table_headers.append('Likelihood ratios')
    elif criterion == 'BF':
        table_headers.append('Bayesian factors')
    elif criterion == 'BIC':
        table_headers.append('BIC')
    else:
        raise ValueError('Wrong criterion given')

    subj, fam, daic = pr._get_data_priors(subjects, criterion=criterion)
    table_data = []
    for ix_sub, sub in enumerate(subj):
        table_data.append([sub, fam[ix_sub], daic[ix_sub]])
    my_table = tabulate(table_data, table_headers)
    print(my_table)
    with open('./table_%s.txt' % criterion, 'w') as mafi:
        mafi.write(my_table)


def figure_2_kolling(fignum=102):
    """Reproduces Kolling's figures 1c and 2."""
    color = figure_colors('histograms')
    fig = plt.figure(fignum, figsize=(8, 4))
    fig.clear()
    grid = gs.GridSpec(1, 3, width_ratios=[4, 4, 3], hspace=0.01)
    ks.plot_dv(bins=4, maax=fig.add_subplot(grid[0]))

    maax = fig.add_subplot(grid[1])
    ks.plot_linear_regression(color=color, maax=maax)
    maax.set_ylabel('beta weights (a.u.)')

    maax = fig.add_subplot(grid[2])
    ks.plot_linear_regression(color=color, regress_trial=False, maax=maax)
    maax.set_ylabel('beta weights (a.u.)')

    for ix, masub in enumerate(fig.get_axes()):
        masub.set_title('ABC'[ix], fontsize=16)
    grid.tight_layout(fig)
    plt.show(block=False)


def figure_3_behavioral_rp(subjects=SUBJECTS, bins=5, rp_range=(0, 35),
                           fignum=103):
    """Figure for behavioral data.
    (A) Histogram of the number of times each risk-pressure is encountered by
    all subjects.
    (B) Average prisky for some subjects.
    """
    fig = plt.figure(fignum, figsize=(6, 3))
    fig.clear()
    _, flata = imda.main()
    outer_grid = gs.GridSpec(1, 2, width_ratios=(2, 1))

    friendly_color = figure_colors('lines_cmap')(0.2)

    # A
    maax = plt.subplot(outer_grid[0])

    risk_pressure_sub = pr.calc_risk_pressure(flata, rmv_after_thres=False)
    risk_pressure = np.array([])
    for subject in risk_pressure_sub:
        risk_pressure = np.hstack([risk_pressure, risk_pressure_sub[subject]])
    maax.hist(risk_pressure, color=figure_colors('histograms'))
    maax.set_ylabel('Number of times encountered')
    maax.set_xlabel('Risk pressure')
    maax.set_xticklabels(10 * np.array(maax.get_xticks().astype(int)))
    maax.set_title('A')
    fig.add_subplot(maax)

    # B
    inner_grid = gs.GridSpecFromSubplotSpec(
        len(subjects), 1, subplot_spec=outer_grid[1], wspace=0.1)
    for ix_subject, subject in enumerate(subjects):
        maax = plt.Subplot(fig, inner_grid[ix_subject])
        pr.plot_rp_bins([subject], bins=bins, rp_range=rp_range,
                        maaxes=[maax], color=friendly_color)
        maax.set_ylim((0, 1))
        if ix_subject == 0:
            maax.set_title('B')
        if ix_subject != len(subjects) - 1:
            maax.set_xticks([])
        else:
            maax.set_xticks((0, 10, 20, 35))
            maax.set_xticklabels((0, 100, 200, 350))
            maax.set_xlabel('Risk pressure')
        if ix_subject == 1:
            maax.set_ylabel('P(risky)')

        maax.set_yticks((0, 0.5))
        fig.add_subplot(maax)
    fig.tight_layout()
    plt.show(block=False)


def figure_4_shapes(fignum=104):
    """Plots the goal shapes."""
    from matplotlib import cm
    cmap = figure_colors('lines_cmap')

    def plot_shape(shape_to_plot, maax, label=None, color='black'):
        """Plots the shape and the threshold line on the - -ax - - provided."""
        shape_to_plot /= shape_to_plot.max()
        maax.plot(shape_to_plot, label=label, color=color)
        if maax.is_first_col() and maax.is_last_row():
            xticks = np.arange(0, mabes.nS, np.floor(
                np.floor(mabes.nS / 5 / 10) * 10).astype(int))
            maax.set_xticks(xticks)
            maax.set_xticklabels(xticks * 10)
            maax.set_xlabel('Accumulated points')
            maax.set_yticks([])
            maax.set_ylabel('Valuation (a.u.)')
        else:
            maax.set_xticks([])
            maax.set_yticks([])
            maax.set_ylabel('')
        maax.set_ylim([0, 1.2])

    def plot_shades_and_labels(mabes, mafig):
        """Actual plotting is done here."""
        all_axes = mafig.get_axes()
        labels = 'ABCD'
        for nplot, maax in enumerate(all_axes):
            shaded_threshold = np.arange(mabes.thres, mabes.nS)
            bg_color = figure_colors('past_threshold_background')
            maax.fill_between(shaded_threshold, 0, maax.get_ylim()
                              [-1], alpha=0.1, color=bg_color)
            maax.set_title(labels[nplot], loc='left')
            maax.set_label('Label via method')
            if nplot != 0:
                maax.legend(fontsize=6, loc='best')

    mabes = bc.betMDP(nS=72, thres=60)

    shape_pars_all = invp.__rds__()

    subplot_size = calc_subplots(len(shape_pars_all) + 1)
    mafig = plt.figure(fignum, figsize=[6, 4])
    plt.clf()

    # labels_pars = [[r'$\mu$', r'$\sigma$'], ['k', r'$x_0$'], ['a']]

    shape_pars_all = [['unimodal_s', [-15, 0, 5], [15, 1, 5]],
                      ['sigmoid_s', [-15, 0, 5], [2, 5, 20]],
                      ['exponential', [5, 20, 100]]]
    multiplier_shape = [[1, 1], [1, 0.1], [0.1, ]]
    labels_all = [[r'$\mu$', r'$\sigma$'], ['k', r'$x_0$'], ['a', ]]

    for subp in range(len(shape_pars_all) + 1):
        maax = plt.subplot(subplot_size[0], subplot_size[1], subp + 1)
        if subp == 0:
            shape_to_plot = mabes.C[:mabes.nS]
            plot_shape(shape_to_plot, maax, color=cmap(1.0))
        else:
            shape_pars = shape_pars_all[subp - 1]
            for ix_val in range(3):
                c_shape_pars = [shape_pars[ix_el + 1][ix_val]
                                for ix_el in range(len(shape_pars) - 1)]
                c_shape_pars.insert(0, shape_pars[0])

                shape_to_plot = mabes.set_prior_goals(
                    shape_pars=c_shape_pars, convolute=False,
                    cutoff=False, just_return=True)
                label = ['%s = %2.1f, ' % (labels_all[subp - 1][c_par],
                                           shape_pars[c_par + 1][ix_val] *
                                           multiplier_shape[subp - 1][c_par])
                         for c_par in range(len(shape_pars) - 1)]
                label_final = "".join(label)[:-2]
                plot_shape(shape_to_plot, maax, label=label_final,
                           color=cmap(ix_val / len(shape_pars_all[0][1])))

    plot_shades_and_labels(mabes, mafig)

    plt.tight_layout()
    plt.show(block=False)


def figure_5_parameter_values(subjects=None, shape=None, fignum=105):
    """Paper figure.
    Plots, on the left, the histogram of alphas. On the right, the inferred
    shapes for all selected subjects.
    """
    if subjects is None:
        subjects = range(35)

    if shape is None:
        # shapes = [shape_pars[0] for shape_pars in invp.__rds__()]
        shape = 'exponential'

    fig = plt.figure(fignum, figsize=(10, 6))
    fig.clear()

    num_colors = 3
    cmap = figure_colors('lines_cmap')
    colors = [cmap(x) for x in np.linspace(0, 1, num_colors)]

    maax_alpha = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
    maax_bias = plt.subplot2grid((2, 2), (0, 1), rowspan=1)
    maax_lnc = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
    maax_clu = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
    alpha_hist(subjects, maax=maax_alpha, shapes=[
        shape], color=figure_colors('histograms'), divisors='auto')
    bias_hist(subjects, maax=maax_bias, shapes=[shape],
              color=figure_colors('histograms'),
              divisors=np.linspace(0.6, 1.2, 7))
    if shape == 'unimodal_s':
        centroids, membership, _ = cl.clustering(
            subjects, k=3, clustering_type='kmeans', shape=shape)
        cluster_names = [[]] * 3
        cluster_names[centroids[:, 1].argmax()] = r'$\uparrow \sigma$'
        cluster_names[centroids[:, 0].argmax()] = r'$\uparrow \mu$'
        cluster_names[centroids.sum(axis=1).argmin()] = r'$\downarrow \mu$'
    elif shape == 'exponential':
        centroids, membership, distorsion = cl.clustering(
            subjects, k=2, clustering_type='kmeans', shape=shape)
        cluster_names = [[]] * 2
        cluster_names[centroids.argmax()] = '$highSTP$'
        cluster_names[centroids.argmin()] = '$lowSTP$'

    plot_cluster_shapes(subjects=subjects, shape=shape, maax=maax_lnc,
                        colors=colors, centroids=centroids,
                        legends=cluster_names)
    # Assign cluster names properly, due to variations in my clustering:
    scatter_kappas(subjects=subjects, shape=shape, maax=maax_clu,
                   membership=membership, centroids=centroids, colors=colors)
    # histogram_clusters(subjects=subjects, shape=shape,
    #                    maax=maax_clu, membership=membership,
    #                    labels=cluster_names, colors=colors)
    maax_lnc.set_label('Label via method')
    maax_lnc.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)


def figure_6_rp_vs_risky(subjects=None, subjects_data=None, all_others=False,
                         shapes=None, do_bias=True, fignum=106):
    """Plots risk pressure vs probability of risky choice for --subjects - -.
    Additionaly, it plots risk pressure vs probability of the same subjects,
    but taking in all the observations of all subjects together.

    The defaults are those used for the paper.

    Parameters
    ----------
    all_others: bool
        Whether to expose all subjects in --subjects - - to each other's data
        (or to those in --all_others - - (see below)). If set to True, a new
        column of subplots will be created.
    subjects_data: array_like, defaults to - -subjects - -
        Subjects from which the data will be taken. Every subject in
        --subjects - - will be exposed to the data of every subject in
        --subjects_data - -.
    shapes: list of strings
        Which shapes to consider for the best_keys. Even if only one is given,
        it must be in a list. E.g. ['exponential']. This is just tunneled to
        other functions.
    """

    # plt.rc('text', usetex=True)

    if subjects is None:
        subjects = SUBJECTS

    if subjects_data is None:
        subjects_data = subjects

    if shapes is None:
        shapes = ['exponential']

    if isinstance(subjects, int):
        subjects = subjects,
    nsubs = len(subjects)

    color_dots = figure_colors('histograms')
    color_avgs = figure_colors('lines_cmap')(0.1)
    color_data = figure_colors('lines_cmap')(0.2)

    fig = plt.figure(fignum, figsize=[9, 6])
    fig.clear()

    outer_grid = gs.GridSpec(nsubs, 2 + all_others * 1, hspace=0.3, wspace=0.1)

    AB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for ix_subj, subject in enumerate(subjects):
        maax = plt.subplot(outer_grid[ix_subj, 0])
        posta_one = pr.plot_rp_vs_risky_own(
            [subject], [maax], regresar=True, color=color_dots, do_bias=do_bias,
            shapes=shapes)
        posta_one_s = {subject: posta_one[subject]}
        _ = pr.plot_average_risk_dynamics(subjects=[subject],
                                          posta_rp=posta_one_s,
                                          maaxes=[maax],
                                          legend='Own contexts',
                                          regresar=True, color=color_avgs,)
        if maax.is_last_row():
            maax.set_xlabel('Risk pressure')
        else:
            maax.set_xticks([])
        maax.set_ylim([0, 1])
        maax.set_yticks([0.25, 0.5])
        maax.set_xlim([0, 35])
        subject_label = '{\\fontsize{16pt}{3em}\\selectfont{}Subject %s}' \
                        % AB[ix_subj]
        mala = maax.set_ylabel(
            subject_label + '\n{\\fontsize{12pt}{3em}\\selectfont{}P(risky)}')
        mala.set_multialignment('center')
        maax.set_title(all_others * 'Own observations' +
                       (not all_others) * 'Inferred preferences')
        fig.add_subplot(maax)

        if all_others:
            maax = plt.subplot(outer_grid[ix_subj, 1])
            # maax = plt.Subplot(fig, inner_grid[1])
            posta_all = pr._plot_rp_vs_risky_all(subjects=[subject],
                                                 axes=[maax],
                                                 subjects_data=subjects_data,
                                                 regresar=True,
                                                 color=color_dots,
                                                 shapes=shapes)
            posta_all_s = {subject: posta_all[subject]}
            _ = pr.plot_average_risk_dynamics(subjects=[subject],
                                              posta_rp=posta_all_s,
                                              maaxes=[maax],
                                              legend='All contexts',
                                              regresar=True, color=color_avgs)
            # maax.set_xticks([])
            if maax.is_last_row():
                maax.set_xlabel('Risk pressure')
            else:
                maax.set_xticks([])
            maax.set_ylim([0, 1])
            maax.set_yticks([])
            maax.set_title('All observations')
            fig.add_subplot(maax)

        # Third column
        maax = plt.subplot(outer_grid[ix_subj, -1])
        pr.plot_pr_vs_prisky_vs_data(subjects=[subject], maaxes=[maax],
                                     bins=np.array([0, 5, 10, 15]),
                                     do_bias=do_bias,
                                     posta_all=posta_one_s,
                                     colors=[color_data, color_avgs])
        maax.set_ylim([0, 1])
        maax.set_title('Method comparison')

    for maax in fig.get_axes():
        xlim = maax.get_xlim()
        maax.plot(xlim, [0.5, 0.5], color='black', alpha=0.5)
        if not maax.is_first_row():
            maax.set_title('')
        if maax.is_last_row():
            maax.set_xticklabels(
                np.array(maax.get_xticks(), dtype=np.int) * 10)
        if maax.is_last_row():
            maax.set_xlabel('Risk pressure')

    # outer_grid.tight_layout(fig)
    plt.show(block=False)


def _process_sim_data(sim_data, subjects, best_pars, do_bias, trim=None, field=None):
    """Opens the files in --sim_data-- and gets posteriors with bias and stuff.

    This helper function is used by figure_7_posta_per_trial() and figure_8_single_offers()
    to process their input --sim_data--, even if it was not provided at runtime.
    """

    if sim_data is None:
        filename_base = './data/simulated_posta_per_trial_%02d_1p5.pi'
        sim_data = []
        for subject in subjects:
            sim_data.append(filename_base % best_pars[subject][1][0][-1][1])
        sim_data.append('./data/data_flat_per_trial.pi')
    # Grab simulated observations:
    with open(sim_data[-1], 'rb') as mafi:
        data_flat = {}
        temp_data = pickle.load(mafi)
        if not trim is None:
            temp_data = trim_data(trim, field=field, flata=temp_data)
        for subject in subjects:
            data_flat[subject] = temp_data

    # Attempt to load as_seen.
    as_seen_all = {}
    for ix_sub, subject in enumerate(subjects):
        try:
            with open(sim_data[ix_sub], 'rb') as mafi:
                as_seen_all[subject] = pickle.load(mafi)
                if do_bias is True:
                    invp.apply_bias(as_seen_all[subject],
                                    best_pars[subject][1][0][1])
                # as_seen.update(new_seen)
        except FileNotFoundError:
            raise
    return as_seen_all, data_flat


def figure_7_posta_per_trial(subjects=None, shapes=None, sim_data=None,
                             shape_pars_all=None, do_bias=True, trim=None,
                             field=None, offer=1, fignum=107):
    """Similar to figure_rp_vs_risky, but with data divided into trials, with
    one subplot per trial.

    All inputs default to those from the paper.
    """
    if subjects is None:
        subjects = SUBJECTS[:-1]

    if shapes is None:
        shapes = ['exponential']

    if do_bias is False:
        rank_fun = pr.loop_rank_likelihoods
    else:
        rank_fun = ba.best_model
    best_pars = rank_fun(subjects=subjects, number_save=1, shapes=shapes)
    if sim_data is None:
        no_calc = False
    else:
        no_calc = True

    as_seen_all, data_flat = _process_sim_data(sim_data, subjects, best_pars, do_bias,
                                               trim, field)

    fig = plt.figure(fignum)
    fig.clear()
    for trial in range(8):
        fig.add_subplot(4, 2, trial + 1)
    maaxes = fig.get_axes()

    cmap = figure_colors('lines_cmap')
    colors_avgs = np.array([cmap(x) for x in np.linspace(0, 1, 3)])
    colors_dots = colors_avgs
    colors_dots /= colors_dots.max(axis=1, keepdims=True)

    posta_all = _get_posta_all(subjects, data_flat, as_seen_all, shape_pars_all,
                               best_pars, no_calc)

    for ix_sub, subject in enumerate(subjects):
        for trial in range(8):
            offset = ix_sub * 0.4
            maax = maaxes[trial]
            these_trials = posta_all[subject]['trial'] == trial
            this_posta = {subject: {
                'rp': posta_all[subject]['rp'][these_trials],
                'posta': posta_all[subject]['posta'][these_trials, :]}}
            pr._plot_rp_vs_risky_own([subject], axes=[maax],
                                     posta_all=this_posta,
                                     color=colors_dots[ix_sub], offset=offset,
                                     alpha=0.2)
    for subject in subjects:
        trim_data(offer, field='reihe', flata=data_flat[subject])
    posta_all = _get_posta_all(subjects, data_flat, as_seen_all, shape_pars_all,
                               best_pars, no_calc)

    for ix_sub, subject in enumerate(subjects):
        for trial in range(8):
            maax = maaxes[trial]
            these_trials = posta_all[subject]['trial'] == trial
            this_posta = {subject: {'rp': posta_all[subject]['rp'][these_trials],
                                    'posta': posta_all[subject]['posta'][these_trials, :]}}
            pr._plot_average_risk_dynamics([subject], maaxes=[maax],
                                           posta_rp=this_posta, color=colors_avgs[ix_sub])
    # Make my plots pretty.
    _pretty_plots_per_trial(maaxes, subjects, colors_avgs, colors_dots)

    fig.tight_layout()
    plt.show(block=False)
    # return posta_all


def figure_8_single_offers(subjects=None, shapes=None, offers=None, trial=4, sim_data=None, do_bias=True,
                           fignum=8):
    """Paper figure.

    Follows the action pairs in --offers-- for the given trial for all values of risk
    pressure.

    Parameters
    ----------
    offers : array_like of ints \in range(8)
    Which offers to follow.
    """
    if subjects is None:
        subjects = SUBJECTS[:-1]
    if shapes is None:
        shapes = ['exponential']
    if offers is None:
        offers = np.array([1])
    if do_bias is False:
        rank_fun = pr.loop_rank_likelihoods
    else:
        rank_fun = ba.best_model

    if sim_data is None:
        no_calc = False
    else:
        no_calc = True

    no_calc = True

    best_pars = rank_fun(subjects=subjects, number_save=1, shapes=shapes)
    cmap = figure_colors('lines_cmap')
    colors_avgs = np.array([cmap(x) for x in np.linspace(0, 1, 3)])
    colors_dots = colors_avgs
    colors_dots /= colors_dots.max(axis=1, keepdims=True)

    fig = plt.figure(fignum)
    maax = fig.add_subplot(111)

    as_seen_all, data_flat = _process_sim_data(
        sim_data, subjects, best_pars, do_bias)

    for offer in offers:
        copy_flata = copy.deepcopy(data_flat)
        for subject in subjects:
            trim_data(offer, field='reihe', flata=copy_flata[subject])
        posta_all = _get_posta_all(subjects, copy_flata, as_seen_all, None,
                                   best_pars, no_calc)
        for ix_sub, subject in enumerate(subjects):
            this_posta = {subject: {'rp': posta_all[subject]['rp'],
                                    'posta': posta_all[subject]['posta']}}
            pr.plot_average_risk_dynamics([subject], maaxes=[maax],
                                          posta_rp=this_posta, color=colors_avgs[ix_sub])


def sup_figure_behavioral_rp(subjects=range(35), bins=4, rp_range=None, fignum=201):
    """RP vs prisky, binning and averaging approach. Like figure 2, but for all
    subjects.
    """

    fig = plt.figure(fignum, figsize=(4, 7))
    fig.clear()

    num_plots = 7
    grid = gs.GridSpec(num_plots, 1, wspace=0.1, hspace=0.1)

    colors = figure_colors('lines_cmap')(np.linspace(0, 1, 35 / num_plots))

    subjects = np.array(subjects).reshape((num_plots, -1))
    for ix_some, some_subs in enumerate(subjects):
        maax = plt.Subplot(fig, grid[ix_some])
        for ix_sub, subject in enumerate(some_subs):
            pr.plot_rp_bins(subjects=[subject], maaxes=[
                            maax], offset=ix_sub * 0.4, color=colors[ix_sub, :])
        fig.add_subplot(maax)
    for maax in fig.get_axes():
        maax.set_ylim([0, 1])
        if not maax.is_last_row():
            maax.set_yticks([0.5])
            maax.set_xticks([])
        else:
            maax.set_yticks([0, 0.5, 1])
            maax.set_xlabel('Risk pressure')
            maax.set_ylabel('p(risky)')
    grid.tight_layout(fig)
    plt.draw()
    plt.show(block=False)


def sup_figure_rp_vs_risky(fignum=202):
    """Like figure_rp_vs_risky() but in a compact form with no
    --all_others--.
    """
    num_plots = 12

    fig = plt.figure(fignum)
    fig.clear()

    outer_grid = gs.GridSpec(4, 3, hspace=0.01, wspace=0.01)
    subjects = np.array(list(range(35)) + [-1]).reshape((num_plots, -1))
    for ix_plot in range(num_plots):
        maax = plt.Subplot(fig, outer_grid[ix_plot])
        for subject in subjects[ix_plot, :]:
            if subject == -1:
                continue
            posta_one = pr._one_agent_one_obs(
                [subject], shapes=['exponential'])
            pr._plot_average_risk_dynamics(subjects=[subject], maaxes=[maax],
                                           posta_rp=posta_one)
        if not maax.is_last_row():
            maax.set_xticks([])
        else:
            maax.set_xlabel('Risk pressure')
        if not maax.is_first_col():
            maax.set_yticks([])
        else:
            maax.set_yticks([0.5])
            maax.set_ylabel('p(risky)')
        if maax.is_first_col() and maax.is_last_row():
            maax.set_yticks([0, 0.5, 1])

        fig.add_subplot(maax)
    outer_grid.tight_layout(fig)
    plt.show(block=False)


def sup_table_parameters():
    """Table with inferred parameters for all subjects."""
    best_pars = pr.loop_rank_likelihoods(shapes=['exponential'])
    table_headers = ['Subject', r'$\alpha$', r'$\kappa$']
    table_data = []
    with open('./table_pars.cvs', 'w') as mafi:
        mafi.write('Subject, Alpha, Kappa\n')
        for subject in range(35):
            data = [subject, best_pars[subject][1][0]
                    [0], best_pars[subject][1][0][1][-1]]
            mafi.write(''.join(['%s, ' % datum for datum in data]) + '\n')


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
    elif element == 'histograms':
        return '#a6cee3'
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


def plot_posta_not_per_trial(subjects=None, subjects_data=None, posta_all=None,
                             maaxes=None, shapes=None, fignum=1072):
    """Similar to figure_rp_vs_risky, but with data divided into trials, with
    one subplot per trial.

    All inputs default to those from the paper. Note that, as of now, this only
    works for subjects = (1, 2). #TODO: expand to all subjects.
    """
    data_flat = None
    if subjects is None:
        subjects = SUBJECTS[:-1]
        with open('./data/simulated_posta_per_trial_99.pi', 'rb') as mafi:
            as_seen = pickle.load(mafi)
        with open('./data/simulated_posta_per_trial_29.pi', 'rb') as mafi:
            as_seen.update(pickle.load(mafi))
        with open('./data/data_flat_per_trial.pi', 'rb') as mafi:
            data_flat = {}
            temp_data = pickle.load(mafi)
            for subject in subjects:
                data_flat[subject] = temp_data

    if maaxes is None:
        fig = plt.figure(fignum)
        fig.clear()
        fig.add_subplot(1, 2, 1)
        fig.add_subplot(1, 2, 2)
        maaxes = fig.get_axes()
    else:
        fig = None

    if shapes is None:
        shapes = ['exponential']

    cmap = figure_colors('lines_cmap')
    colors_avgs = np.array([cmap(x) for x in np.linspace(0, 1, len(subjects))])
    colors_dots = colors_avgs
    colors_dots /= colors_dots.max(axis=1, keepdims=True)

    posta_all = pr.one_agent_many_obs(
        subjects, shapes=shapes, return_t=True, as_seen=as_seen,
        subjects_data=subjects_data, flata=data_flat)

    for ix_sub, subject in enumerate(subjects):
        maax = maaxes[ix_sub]
        this_posta = {subject: {'rp': posta_all[subject]['rp'],
                                'posta': posta_all[subject]['posta']}}
        pr._plot_average_risk_dynamics([subject], maaxes=[maax],
                                       posta_rp=this_posta, color=colors_avgs[ix_sub])

    plt.show(block=False)
