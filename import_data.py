#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:00:32 2017

@author: dario

Series of functions to import data from the Kolling_2014 experiment obtained by
Sven Breitmeyer in Matlab format.

Data is cleaned and formatted in the way necessary for performing inference
over hyperparameters.

"""
import os
import numpy as np
from scipy import io

class cd:
    """ Context thingy for temporarily changing working folder.
    """
    def __init__(self, newPath, force=False):
        self.newPath = newPath
        self.force = force

    def __enter__(self):
        self.oldPath = os.getcwd()
        try:
            os.chdir(self.newPath)
        except FileNotFoundError:
            if self.force:
                os.mkdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.oldPath)


def import_kolling_data(path_to_data = None):
    """ Reads the data for the Kolling experiment as saved by Sven Breitmeyer.
    """
    if path_to_data is None:
        path_to_data = './exp_data/'
    with cd(path_to_data):
        files = [file for file in os.listdir('.') if file[-3:] == 'mat']
        data = []
        for file in files:
            data.append(io.loadmat(file))
            data[-1]['filename'] = file
    if not data:
        raise RuntimeError('No .mat files in the specified folder.')
    return data

def clean_data(data):
    """ Cleans the mess that importing the data from matlab causes and prunes
    unnecessary fields from the data.
    """
    nZ = len(data) #number of participants
    data_clean = [data[z]['block'][0] for z in range(nZ)]
    data_out = []
    for datin in data_clean:
        data_out.append({})
        for mana in ['points', 'reihe', 'response', 'rechts', 'target']:
            data_out[-1][mana] = np.array(datin[mana])
            for do in range(data_out[-1][mana].shape[0]):
                data_out[-1][mana][do] = data_out[-1][mana][do].squeeze()
            data_out[-1][mana] = np.array(list(data_out[-1][mana]), dtype=int)

    return data_out

def enhance_data(data):
    """ Transforms the fields of the data (already cleaned in clean_data) into
    something usable for the python algorithms and stuff.

    Works in-place.
    """
    nZ = len(data) #number of participants
    nD, nT = data[0]['response'].shape
    target=np.array([595,930,1035,1105])  # From Sven's code

    for z in range(nZ):
        #calculate actual response
        data[z]['choice'] = np.logical_xor(data[z]['response'],
                                           data[z]['rechts'])*1
        #add trial number
        data[z]['trial'] = np.tile(np.arange(nT), (nD,1)).astype(int)
        data[z]['threshold'] = target[data[z]['target']-1].astype(int)
        data[z]['obs'] = np.cumsum(data[z]['points'], axis=1).astype(int)
        data[z]['TargetLevels'] = target
        data[z]['NumberGames'] = nD
        data[z]['NumberTrials'] = nT

def flatten_data(data):
    R""" Flatten all relevant fields into one long string of 'independent'
    trials.
    """
    nZ = len(data)
    nD, nT = data[0]['response'].shape
    data_flat = []
    for z in range(nZ):
        data_flat.append({})
        data_flat[-1]['TargetLevels'] = data[z]['TargetLevels'].astype(int)
        data_flat[-1]['NumberGames'] = data[z]['NumberGames']
        data_flat[-1]['NumberTrials'] = data[z]['NumberTrials']
        for name in ['choice', 'obs', 'trial', 'reihe']:
            data_flat[-1][name] = np.ndarray.flatten(data[z][name])
        data_flat[-1]['threshold'] = np.ndarray.flatten(np.tile(data[z]['threshold'], (nT,1)), order='F')
    return data_flat

def add_initial_obs(data_flat):
    R""" Adds the initial observation to the data. To do this, it removes the
    last observation from each game (from the data) and adds a zero at the
    beginning of each game, changing the concept from "observations after
    action was taken" to "observations before action was taken".

    Works in-place.
    """
    nD = data_flat[0]['NumberGames']
    nT = data_flat[0]['NumberTrials']
    for n in range(len(data_flat)):
        data_flat[n]['obs'] = np.reshape(np.hstack([np.zeros((nD,1)),
                              np.reshape(data_flat[n]['obs'], (-1, nT))[:,:-1]]), (-1))
#        for name in ['reihe', 'threshold', 'choice', 'trial']:
#            data_flat[n][name] = np.reshape(np.hstack([np.reshape(data_flat[n][name],
#                                  (-1,nT)), np.zeros((nD,1), dtype=int)]), (-1))

def small_data(data, nGames = 2):
    R"""Remove most observations from the data for test runs.

    For every subject in data, only the first nGames games are kept (that is,
    nT*nGames observations in total).

    """
    import copy
    nT = data[0]['NumberTrials']
    nObs = nT*nGames
    data_out = copy.deepcopy(data)
    for z in range(len(data_out)):
        for name in ['obs','threshold','reihe','choice','trial']:
            data_out[z][name] = data_out[z][name][:nObs]
            data_out[z]['NumberGames'] = nGames
#    data_out['threshold'] = data_out['threshold'][nObs]
#    data_out['reihe'] = data_out['reihe'][nObs]
#    data_out['choice'] = data_out['choice'][nObs]
#    data_out['trial'] = data_out['trial'][nObs]
    return data_out

def test_past_threshold(data):
    r""" Obtains distributions over points obtained by all subjects, separated
    by threshold.

    The purpose is to see what cutoff should be used (i.e. how many points
    past-threshold should be included in the actinf simulations).
    """
#    from matplotlib import pyplot as plt
    nZ = len(data)

    threses = np.unique(data[0]['threshold'])
    nThres = threses.size
    dist = {}
    for nt in range(nThres):
        dist[threses[nt]] = []

    for z in range(nZ):
        flag = 0
        k = 0
        while flag==0:
            try:
                dist[data[z]['threshold'][k]].append(data[z]['obs'][k,-1])
                k += 1
            except:
                flag = 1

    for nt in range(nThres):
        dist[threses[nt]] = np.reshape(np.array(dist[threses[nt]]),(-1))


    return dist

def plot_past_threshold(dist, fignum = None, ax = None, threshold = None):
    r""" Plots histograms of the output of test_past_threshold().

    Parameters
    ----------
    dist: dict of 1-array of float
        Output of the function test_past_threshold() above.
    fignum: 1-array, int (or int)
        Figure number to plot to.
    ax: int
        Axes to plot to. Overrides fignum.
    threshold: 1-array, int (or int). Default = all in data
        Set of thresholds (indices) to plot. If an array is provided, they will
        be plotted in a subplot array.
    """
    from matplotlib import pyplot as plt
    import utils

    if ax is None:
        if fignum is None:
            fig = plt.figure()
            fig.clf()
        else:
            fig = plt.figure(fignum)
            fig.clf()
        ax = fig.gca()


    if threshold is None:
        threshold = range(len(dist))
        s1, s2 = utils.calc_subplots(len(dist))
    elif isinstance(threshold, int):
        s1, s2 = 1, 1
    else:
        s1, s2 = utils.calc_subplots(len(threshold))
        assert len(threshold) <= len(dist), ('More thresholds provided than'+
                                              'the data has.')
    keys = list(dist.keys())
    keys = [keys[t] for t in threshold]

    for k,key in enumerate(keys):
        ax = plt.subplot(s1,s2,k+1)
        ax.hist(dist[key])
        ax.set_title('Threshold: %d' % key)
        ax.axvline(key, linewidth=3, color='r')

def cap_states(data_flat):
    r""" Caps the number of states as 1.2 times the threshold.

    Works in-place
    """
    target_levels = data_flat[0]['TargetLevels']
    max_state = {}
    for tlvl in target_levels:
        max_state[tlvl] = int(1.2*tlvl)
    nts = data_flat[0]['obs'].size
    for da in range(len(data_flat)):
        for s in range(nts):
            if data_flat[da]['threshold'][s]==0:
                continue
            data_flat[da]['obs'][s] = min(data_flat[da]['obs'][s],
                                      max_state[data_flat[da]['threshold'][s]])
def prune_trials(data_flat, trials):
    r""" Will remove from the data the trials not in 'trials'. The idea is
    to remove the earlier trials, which contain (presumably) less information
    than later trials, but take much longer to simulate with active inference.
    """
    import copy
    data_out = copy.deepcopy(data_flat)
    tTot = data_out[0]['trial'].size
    for d, datum in enumerate(data_flat):
        indices = np.zeros(tTot, dtype=bool)
        for t in trials:
            indices = indices + np.array(datum['trial']==t)
        for name in ['obs', 'threshold', 'reihe','choice','trial']:
            data_out[d][name] = datum[name][indices]

    return data_out
def main(path_to_data = None):
    data = import_kolling_data(path_to_data)
    data = clean_data(data)
    enhance_data(data)
    data_flat = flatten_data(data)
    add_initial_obs(data_flat)
    return data, data_flat

if __name__ == "__main__":
    data, data_flat = main()
    small = small_data(data_flat, 10)
    small_prune = prune_trials(small, [0,1,2])
    prune = prune_trials(data_flat, [0,1,2])
