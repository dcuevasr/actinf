#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:46:59 2017

@author: dario

Script for sampling the entire parameter space for the Kolling task, in which
the goals are the one free parameter, which itself is parametrized in terms of
the sufficient statistics of a normal distribution.

Data is created for a number of games, with a 'true' parametrization for the
goals, which is not normal (maybe change this?). Then, runs are simulated with
the active inference agent for all the parameter values.

"""
from time import time
#%% Generate data
import betClass as bc
import numpy as np

mabes = bc.betMDP()
lnc = mabes.lnC
gammi = mabes.gamma
lambd = mabes.lambd
alpha = mabes.alpha
beta = mabes.beta
v = mabes.V.astype('int8')
s = mabes.S.astype('float64')
h = mabes.H
b = mabes.B


nD = 10  # Number of games played by subjects
nS = mabes.nS
nT = mabes.nT
data = np.zeros(nD*nT, dtype=np.float64)
trial = np.zeros(nD*nT, dtype=np.int64)
state = np.zeros(nD*nT, dtype=np.int64)
posta = np.zeros((nD*nT, 2), dtype=np.float64)
for d in range(nD):
    mabes.exampleFull()
    data[d*nT:(d+1)*nT] = mabes.Example['Actions']
    trial[d*nT:(d+1)*nT] = range(nT)
    state[d*nT:(d+1)*nT] = mabes.Example['RealStates']
    posta[d*nT:(d+1)*nT,:] = mabes.Example['PostActions']

stateV = np.zeros((nD*nT, mabes.Ns))

for n, st in enumerate(state):
    stateV[n, st] = 1

#%% Create mesh of parameter values
min_mean = 55 # Minimum value for the mean
max_mean = nS
min_sigma = 10 # >0
max_sigma = 20 # Maximum value for the variance
mu_sigma = np.zeros((max_mean-min_mean, max_sigma-min_sigma, 2))
arr_lnc = np.zeros((max_mean-min_mean, max_sigma-min_sigma, mabes.Ns))
for m, mu in enumerate(range(min_mean, max_mean)):
    for s, sigma in enumerate(range(min_sigma,max_sigma)):
        mu_sigma[m, s,:] = [mu, sigma]
        arr_lnc[m, s, :] = np.log(mabes.set_prior_goals(selectShape='unimodal',
                                                 Gmean = mu, Gscale= sigma,
                                                 just_return = True,
                                                 convolute = True)+np.exp(-16))


#%% Now with pure python
mabed = {}
for m, mu in enumerate(range(min_mean, max_mean)):
    for sd, sigma in enumerate(range(min_sigma,max_sigma)):
        mabed[(mu, sd)] = bc.betMDP()
        mabed[(mu, sd)].lnc = arr_lnc[m, sd, :]
        for s, sta in state:
            posta_inferred = mabed[(mu, s)].posteriorOverStates(state[s],
                                            trial[s], v, 0, 0, gammi,
#%% Compile the Actinf theano function
#import actinfThClass as afOp
#maaf2 = afOp.actinfOpElewise(lnc, lambd, alpha, beta, v, stateV, h, b, trial)
#
#
##%% Get posteriors for all observations in stateV
#print("Starting")
#t_ini = time()
#from theano import tensor as T
#test2 = maaf2(T.as_tensor(np.hstack([gammi, lnc]))).eval()
#post_act = np.zeros((max_mean-min_mean, max_sigma-min_sigma, 2*nT*nD))
#for m, mu in enumerate(range(min_mean, max_mean)):
#    for s, sigma in enumerate(range(min_sigma,max_sigma)):
#        clnc = arr_lnc[m,s,:]
#        post_act[m, s, :] = maaf2(T.as_tensor(np.hstack([gammi, clnc]))).eval()
#print(time() - t_ini)
#
##%% Calculate marginal p(y) = sum_\theta {p(y|\theta)}
#
## Reorder to flatten all trials from all games
#new_post = np.reshape(post_act,
#                      (max_mean - min_mean, max_sigma - min_sigma,-1, 2))
#new_post = np.swapaxes(new_post, -1, -2) + np.exp(-16)
#
## Marginalize over all values of the parameters
#p_of_y_marg = np.sum(new_post, axis = (0,1))
#def likelihood(data, dist):
#    return dist[0]**data[0]*dist[1]**data[1]
#
## p(y|\theta)/p(y)
#likelihood_model = np.zeros((max_mean - min_mean, max_sigma - min_sigma))
#for mu in range(max_mean - min_mean):
#    for sd in range(max_sigma - min_sigma):
#        likelihood_model[mu, sd] = np.prod(likelihood(data, new_post[mu, sd,:,:]))






