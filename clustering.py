#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clustering crap routines."""
from copy import deepcopy
import pickle

import ipdb

import numpy as np
from scipy.stats import entropy
from scipy.stats import norm

import invert_parameters as invp
import bias_analysis as ba


def clustering(subjects=None, k=2, clustering_type=None, shape=None,
               force_rule=True):
    """Main function of this module. It will do the following:
    1. Find the best alpha for each subject. Fix alphas.
    2. Find the --k-- best clusters, according to --shape--,
    and --clustering_type--.
    3. Find which cluster each subject belongs to.
    4. Plot everything nicely.

    Parameters
    ----------
    subjects : array-like of ints
        Subjects to consider. Defaults to range(35).
    k : int
        Number of clusters to consider.
    clustering_type: ['kmeans', 'kmedoids']
        Whether to use k-means to create the best clustering
        centers, or k-medoids to choose them from the likelihood
        maps of each subject. Note that k-means uses the maximum
        likelihood parameters, while k-medoids uses the entire
        likelihood map, taking uncertainty into account.
    shape : string
        Which shape to use. See invp.__rds__() for the available
        shapes. If no shape is specified, 'unimodal_s' will be used.

    Returns
    -------
    centroids: list (with --k-- elements)
        Centers of the clusters. If --clustering_type-- is 'kmeans',
        the values of the parameters are returned. If k-medoids is
        used, the number of the subjects that were chosen as the
        centroids are returned.
    cluster_membership: list (len(subjects))
        Returns an array with an element per subject, stating to
        which cluster (as indexed in --centroids--) this subject
        belongs.
    """
    if subjects is None:
        subjects = range(35)
    if shape is None:
        shape = 'unimodal_s'
    if clustering_type is None:
        clustering_type = 'kmedoids'

    shape_pars, all_alphas = invp.__rds__(shape, alpha=True)
    size_space = [len(all_alphas)] + [len(parx) for parx in shape_pars[1:]]

    likelihoods = likelihood_map_shape(subjects, shape=shape)
    if force_rule:
        for subject in subjects:
            likelihoods[subjects] = force_rule_likelihood(
                shape, likelihood=likelihoods[subject])
    alphas = {}
    best_shapes = np.zeros((len(subjects), len(shape_pars[1:])))
    for ix_subject, subject in enumerate(subjects):
        ix_best = np.unravel_index(
            likelihoods[subject].argmax(), size_space)
        ix_alpha = ix_best[0]
        alphas[subject] = all_alphas[ix_alpha]

    if clustering_type == 'kmeans':
        best_bias = ba.best_model(subjects, shapes=[shape])
        best_shapes = np.zeros((len(subjects), len(shape_pars[1:])))
        for ix_sub, subject in enumerate(subjects):
            best_shapes[ix_sub, :] = best_bias[subject][1][0][-1][1:]

    if clustering_type == 'kmedoids':
        centroids, data_dists = find_clusters_medoids(subjects, k=k, stretch=True,
                                                      shape=shape, alphas=alphas,
                                                      likelihoods=likelihoods)
        membership = loop_assign_cluster(
            subjects, data_dists, data_dists[centroids],
            clustering_type=clustering_type)
        return centroids, membership

    elif clustering_type == 'kmeans':
        centroids, distorsion = find_clusters_kmeans(k, best_shapes)
        membership = loop_assign_cluster(
            subjects, best_shapes, centroids, clustering_type=clustering_type)
        return centroids, membership, distorsion


def loop_assign_cluster(subjects, data_dists, centroids, clustering_type):
    """Finds the cluster that all subjects belong to.

    Parameters
    ----------
    subjects : array-like.
    data_dists :
    centroids :

    Returns
    -------
    membership : array-like (len(subjects))
        For each subject, the index of --centroids-- to which it belongs.
    """
    membership = {}
    for subject in subjects:
        membership[subject], _ = assign_cluster(clustering_type,
                                                data_dists[subject],
                                                centroids)

    return membership


def find_clusters_kmeans(k, subject_pars):
    """Finds the best means for the clustering, using the k-means algorithm."""
    from scipy.cluster.vq import whiten, kmeans

    pars_std = np.std(subject_pars, axis=0)
    whitened = whiten(subject_pars)
    codebook, distortion = kmeans(whitened, k)

    for col in codebook:
        col *= pars_std

    return codebook, distortion


def find_clusters_medoids(subjects=None, k=2, stretch=False, shape=None,
                          likelihoods=None, alphas=None):
    """Finds the best set of 'means' for the clustering, using the k-medoids
    algorithm.

    Parameters
    ----------
    k: int
        Number of clusters
    stretch: bool
        If True, the stretched version of the 2D likelihood map will be used.
        If False, the projected version. Stretched means that the N-dimensional
        array of parameter values (for N parameters) is flattened into a single
        vector. The projected version only works with Exponential (untested: I
        think the others would work if you flatten (stretch) them first) and
        will add all values of alpha to project into the shape parameters-space
    """
    if shape is None:
        shape = 'exponential'
    if subjects is None:
        subjects = range(35)

    if not alphas is None:
        _, all_alphas = invp.__rds__(shape, alpha=True)
        ix_alphas = [np.where(all_alphas == alphas[ix])[0]
                     for ix in range(len(subjects))]
    if likelihoods is None:
        likelihoods = likelihood_map_shape(subjects, shape=shape)

    data_dists = []
    for ix_subject, subject in enumerate(subjects):
        ix_alpha = ix_alphas[ix_subject]
        if stretch:
            new_likeli = np.reshape(likelihoods[subject][ix_alpha], -1)
        else:
            new_likeli = likelihoods[subject].sum(axis=0)
        data_dists.append(new_likeli)
    data_dists = np.array(data_dists)

    return k_medoids_kl(data_dists, k, 20), data_dists


def kl_clusters_to_shape_pars(clusters, data_dists=None):
    """Takes the output of model_selection_clustering() and turns it into the
    closest shape_pars available in invp.__rds__().

    Parameters
    ----------
    clusters: 1D - or 2D - array
        If 1D, it is assumed to be indices corresponding to - -data_dists - - rows.
        This scheme can be used with the direct outputs of
        model_selection_clustering(). --data_dists - - must be provided too. If
        2D, it is assumed to be directly the subjects' posteriors who were
        selected as the best cluster centroids.
    data_dists: 2D array
        To be used in combination with a 1D - -clusters - -. See above.

    Returns
    -------
    shape_pars_clusters: list
        As many elements as clusters there are. Each one being the closest
        --shape_pars - - that exists in invp.__rds__().
    """
    if clusters.ndim == 1:
        best_shapes = data_dists[clusters, :]
    else:
        best_shapes = clusters

    shape_pars_all = invp.__rds__('exponential')
    best_shape_pars = []
    for best_shape in best_shapes:
        max_kappa = shape_pars_all[1][best_shape.argmax()]
        kappa_dist = np.inf
        for c_kappa in shape_pars_all[1]:
            new_kappa_dist = abs(c_kappa - max_kappa)
            if new_kappa_dist < kappa_dist:
                current_best_kappa = c_kappa
                kappa_dist = new_kappa_dist
        best_shape_pars.append(['exponential', current_best_kappa])

    return best_shape_pars


def k_medoids_kl(data_dists, k, timeout=100):
    """Performs clustering based on the probability distributions of data points
    given by --data_dists-- using the algorithm called k-medoids with a
    Kullback-Leibler divergence, as presented in [1].

    [1]Jiang, B., Pei, J., Tao, Y., and Lin, X. (2013). Clustering Uncertain
    Data Based on Probability Distribution Similarity. IEEE Transactions on
    Knowledge and Data Engineering 25, 751–763.

    Parameters
    ----------
    data_dists: 2D - array
        (d, s) - array of d data points and a discrete probability distribution
        in a space whose size is s.
    k: int
        Number of centroids to calculate.

    Returns
    -------
    centroids: 1D - array
        (k)-array containing the indices of the elements of --data_dists-- that
        were chosen as the k centroids of the clusters.
    """
    num_data, _ = data_dists.shape
    # Building phase:
    c_centroids = _building_phase(k, num_data, data_dists)

    # =========================================================================
    # Swapping phase
    # =========================================================================
    not_converge = True
    c_centroids = np.array(c_centroids)
    best_kl = calculate_total_kl(c_centroids, data_dists=data_dists)
    current_time = 0
    while not_converge:
        new_centroid, old_centroid, new_kl = _one_swap(
            num_data, data_dists, c_centroids)
        if new_kl.min() < best_kl and current_time <= timeout:
            c_centroids[old_centroid] = new_centroid
            best_kl = new_kl.min()
            current_time += 1
            # print(c_centroids, new_centroid)
        else:
            not_converge = False

    return c_centroids


def _building_phase(k, num_data, data_dists):
    """ Building phase of the k-medoids function above."""
    c_centroids = []
    total_kl = np.inf * np.ones(num_data)
    for ix_datum in range(num_data):
        total_kl[ix_datum] = 0
        for iy_datum in range(num_data):
            if ix_datum == iy_datum:
                continue
            current_p = data_dists[ix_datum]
            current_q = data_dists[iy_datum]
            total_kl[ix_datum] += entropy(current_q, current_p)
    c_centroids.append(total_kl.argmin())

    # Choosing the i-th C_i, 1<i<=k
    for _ in range(1, k):
        total_kl = np.zeros(num_data)
        for ix_datum in range(num_data):
            if ix_datum in c_centroids:
                continue
            for iy_datum in range(num_data):
                if iy_datum == ix_datum or iy_datum in c_centroids:
                    continue
                _, iy_div = assign_cluster('kmedoids',
                                           0, data_dists[(iy_datum,) +
                                                         tuple(c_centroids) +
                                                         (ix_datum,), :])

                total_kl[ix_datum] += iy_div
        total_kl[c_centroids] = np.inf
        c_centroids.append(total_kl.argmin())
    return c_centroids


def _one_swap(num_data, data_dists, c_centroids):
    """Does one swap of cluster centroids for k-medoids."""
    new_kl = np.zeros(num_data)
    for ix_datum in range(num_data):
        if ix_datum in c_centroids:
            new_kl[ix_datum] = np.inf
            continue
        this_cluster, _ = assign_cluster('kmedoids', data_dists[ix_datum, :],
                                         data_dists[tuple(c_centroids), :])
        new_centroids = deepcopy(c_centroids[:])
        new_centroids[this_cluster] = ix_datum
        new_kl[ix_datum] = calculate_total_kl(new_centroids, data_dists)
    new_centroid = new_kl.argmin()
    old_centroid, _ = assign_cluster('kmedoids',
                                     data_dists[new_centroid, :],
                                     data_dists[tuple(c_centroids), :])
    return new_centroid, old_centroid, new_kl


def assign_cluster(clustering_type, *args, **kwargs):
    """Finds the column in --data_dists-- which is the closest to that
    specified by --index--.

    Parameters
    ----------
    clustering_type: {'kmedoids', 'kmeans'}
        Whether to use k-medoids or k-means.

    For k-medoids:
    data_dists: 2D - array
        Matrix with one row per data point; each row is the probability
        mass distribution of that data point.
    dist_to_compare: int or 1D - array
        Indicates the row of - -data_dists - - which is the one for which
        the function will find the closest row(other than itself). If
        it's an 1D - array, it is taken as the one to find a cluster for.

    For k-means:
    member: numpy array
        Member for whom the cluster is to be found, in terms of a tuple
        (par1, par2, ...).
    clusters : 2D numpy array
        All available clusters, in terms of the tuple (par1, par2, ...).
        One row per cluster.

    Returns
    -------
    best_ix: int
        Row of --data_dists-- which is closest to that given by
        --index--.
    best_div: float
        KL divergence or Euclidian distance with the corresponding cluster.
    """
    if clustering_type is None:
        clustering_type = 'kmedoids'

    if clustering_type == 'kmedoids':
        return _assign_kmedoids(*args, **kwargs)
    elif clustering_type == 'kmeans':
        return _assign_kmeans(*args, **kwargs)
    else:
        raise ValueError('Unrecognized --clustering_type--')


def _assign_kmeans(member, clusters):
    """Finds the corresponding centroid in --clusters-- for that in
    --member--.
    """
    le_diff = abs(clusters - member)
    best_ix = le_diff.sum(axis=1).argmin()
    best_div = le_diff.min()

    return best_ix, best_div


def _assign_kmedoids(dist_to_compare, data_dists):
    """Finds the corresponding centroid in --data_dists-- for that in
    dist_to_compare.
    """
    dist_index = np.inf
    if isinstance(dist_to_compare, int):
        dist_index = dist_to_compare
        c_element = data_dists[dist_to_compare]
    else:
        c_element = dist_to_compare

    best_div = np.inf
    for ix_datum, datum in enumerate(data_dists):
        if ix_datum == dist_index:
            continue
        new_div = entropy(c_element, datum)
        if best_div > new_div:
            best_div = new_div
            best_ix = ix_datum
    return best_ix, best_div


def calculate_total_kl(clusters, data_dists):
    """Calculates the total KL divergence in --data_dists - - by clustering the
    rows in the clusters indicated by - -clusters - -.
    """
    if not isinstance(clusters, tuple):
        clusters = tuple(clusters)

    total_kl = 0
    for datum in data_dists:
        _, best_div = assign_cluster('kmedoids',
                                     datum, data_dists[clusters, :])
        total_kl += best_div
    return total_kl


def generate_simulated_data(d, s, k):
    """ Generates - -d - - Gaussian probability distributions in an - -s - - space.

    This data is to be used as test for the clustering, and as such is made
    with some built - in clusters(k of them), so results can be assessed.
    """
    sigma = 2
    clusters = np.linspace(0, s, k + 1, endpoint=False)
    clusters = clusters[1:]

    space = np.arange(s)

    data_dists = np.zeros((d, s))
    for ix_d in range(d):
        c_mean = np.random.choice(clusters) + 5 * np.random.randn()
        data_dists[ix_d, :] = norm.pdf(space, c_mean, sigma)
        data_dists[ix_d, :] /= data_dists[ix_d, :].sum()

    return data_dists


def likelihood_map_shape(subjects, loglikeli=False, shape=None):
    r"""Creates an array per subject with the data likelihoods for all values of the
    parameters in the given shape and the values of alpha.

    Returns
    -------
    likelihoods: list of matrices
        Every matrix is the likelihood map for one subject, with \alpha on the first
        coordinate and the shape parameters next (in the order given by invp.__rds__).
    loglikeli: bool
        If True, the log-likelihood will be returned. If False, the normalized likelihood.

    """
    if shape is None:
        shape = 'exponential'
    all_pars, alphas = invp.__rds__(shape, alpha=True)
    num_vals = np.prod([len(this_par) for this_par in all_pars[1:]])
#    num_vals = len(all_pars[1])
    num_alphas = len(alphas)

    likelihoods = dict()
    for subject in subjects:
        likelihoods[subject] = -np.inf * np.ones((num_alphas, num_vals))
        with open('./data/alpha_logli_subj_%s_%s.pi' % (subject, shape), 'rb') as mafi:
            logli = pickle.load(mafi)
        logli_keys = np.array(list(logli.keys()))
        logli_keys.sort()
        for ix_alpha, alpha in enumerate(logli_keys):
            # tmp_logli = np.exp(logli[alpha] - logli[alpha].max())
            likelihoods[subject][ix_alpha, :] = np.reshape(logli[alpha], -1)
        if loglikeli is False:
            likelihoods[subject] = np.exp(
                likelihoods[subject] - likelihoods[subject].max())
            likelihoods[subject] /= likelihoods[subject].sum()

    return likelihoods


def force_rule_likelihood(shape, likelihood=None, loglikelihood=None,
                          has_alpha=True):
    """Makes the likelihood 0 or the log-likelihood 1 for all parameter
    values that do not meet the rules set for shape separation (to be
    found in invp.force_rule()).

    Parameters
    ----------
    likelihood/loglikelihood: numpy array
        Depending on which one is given, its entries that do not meet
        the rule are set to 0 or 1, respectively.
    shape: string
        Shape to consider.
    has_alpha: book
        Whether the alpha parameter is included in the --likelihood--
        (or log) array. If True, the first dimension of the array is
        assumed to be alpha, and is thus ignored.

    Returns
    -------
    forced: numpy array (same size as input)
        Likelihood or log-likelihood with its entries modified.
    """

    shape_pars = invp.__rds__(shape)

    size_shape = [len(parx) for parx in shape_pars[1:]]

    if not likelihood is None:
        logli = likelihood  # Despite its name, it could be not-log
        bad_value = 0.
    elif not loglikelihood is None:
        logli = loglikelihood
        bad_value = - np.inf
    else:
        raise ValueError('Neither --likelihood-- nor --loglikelihood--' +
                         ' were provided.')
    logli_shape = logli.shape

    if has_alpha is True:
        alpha_shape = logli_shape[0]
        logli_shape = logli_shape[1:]

    reshaped = False
    # If the shape dimensions are flattened, unravel them:
    if len(size_shape) > len(logli_shape):
        reshaped = True
        # Try to reshape
        if has_alpha is True:
            new_shape = [alpha_shape, ] + size_shape
        else:
            new_shape = size_shape
        logli = np.reshape(logli, new_shape)

    rule_array = np.logical_not(invp.rule_all(shapes=[shape])[0])
    if has_alpha is True:
        logli[:, rule_array] = bad_value
    else:
        logli[rule_array] = bad_value

    if reshaped is True:
        # TODO: reshape
        pass

    return logli
