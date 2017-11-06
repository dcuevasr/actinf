#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clustering crap routines."""
import ipdb
from copy import deepcopy

import numpy as np
from scipy.stats import entropy
from scipy.stats import norm


def k_medoids_KL(data_dists, k, timeout=100):
    """Performs clustering based on the probability distributions of data points
    given by --data_dists-- using the algorithm called k-medoids with a
    Kullback-Leibler divergence, as presented in [1].

    [1]Jiang, B., Pei, J., Tao, Y., and Lin, X. (2013). Clustering Uncertain
    Data Based on Probability Distribution Similarity. IEEE Transactions on
    Knowledge and Data Engineering 25, 751–763.


    Parameters
    ----------
    data_dists: 2D-array
        (d, s)-array of d data points and a discrete probability distribution
        in a space whose size is s.
    k: int
        Number of centroids to calculate.

    Returns
    -------
    centroids: 1D-array
        (k)-array containing the indices of the elements of --data_dists-- that
        were chosen as the k centroids of the clusters.
    """
    num_data, size_space = data_dists.shape
    # Building phase
    # =========================================================================
    # Choosing the first C_i
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
    for c_k in range(1, k):
        total_kl = np.zeros(num_data)
        for ix_datum in range(num_data):
            if ix_datum in c_centroids:
                continue
            for iy_datum in range(num_data):
                if iy_datum == ix_datum or iy_datum in c_centroids:
                    continue
                iy_cluster, iy_div = find_cluster(
                    0, data_dists[(iy_datum,) + tuple(c_centroids) +
                                  (ix_datum,), :])
                total_kl[ix_datum] += iy_div
        total_kl[c_centroids] = np.inf
        c_centroids.append(total_kl.argmin())
    # =========================================================================
    # Swapping phase
    # =========================================================================
    not_converge = True
    c_centroids = np.array(c_centroids)
    best_kl = calculate_total_kl(c_centroids, data_dists=data_dists)
    current_time = 0
    while not_converge:
        new_kl = np.zeros(num_data)
        for ix_datum in range(num_data):
            #            if ix_datum == 0:
            #                ipdb.set_trace()
            if ix_datum in c_centroids:
                new_kl[ix_datum] = np.inf
                continue
            this_cluster, _ = find_cluster(data_dists[ix_datum, :],
                                           data_dists[tuple(c_centroids), :])
            new_centroids = deepcopy(c_centroids[:])
            new_centroids[this_cluster] = ix_datum
            new_kl[ix_datum] = calculate_total_kl(new_centroids, data_dists)
        print(best_kl, new_kl.min())
        new_centroid = new_kl.argmin()
        old_centroid, _ = find_cluster(data_dists[new_centroid, :],
                                       data_dists[tuple(c_centroids), :])
        if new_kl.min() < best_kl and current_time <= timeout:
            c_centroids[old_centroid] = new_centroid
            best_kl = new_kl.min()
            current_time += 1
            # print(c_centroids, new_centroid)
        else:
            not_converge = False
    print(current_time)
    return c_centroids


def find_cluster(dist_to_compare, data_dists):
    """Finds the column in --data_dists-- which is the closest to that
    specified by --index--.

    Parameters
    ----------
    data_dists: 2D-array
        Matrix with one row per data point; each row is the probability
        mass distribution of that data point.
    dist_to_compare: int or 1D-array
        Indicates the row of --data_dists-- which is the one for which
        the function will find the closest row (other than itself). If
        it's an 1D-array, it is taken as the one to find a cluster for.

    Returns
    -------
    best_ix: int
        Row of --data_dists-- which is closest to that given by
        --index--.
    best_div: float
        KL divergence with the corresponding cluster.
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
    """Calculates the total KL divergence in --data_dists-- by clustering the
    rows in the clusters indicated by --clusters--.
    """
    if not isinstance(clusters, tuple):
        clusters = tuple(clusters)

    total_kl = 0
    for ix_datum, datum in enumerate(data_dists):
        _, best_div = find_cluster(datum, data_dists[clusters, :])
        total_kl += best_div
    return total_kl


def generate_simulated_data(d, s, k):
    """ Generates --d-- Gaussian probability distributions in an --s-- space.

    This data is to be used as test for the clustering, and as such is made
    with some built-in clusters (k of them), so results can be assessed.
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
