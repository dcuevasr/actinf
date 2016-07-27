# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:30:19 2016

@author: dario
"""

import actinfClass as af
import cbet
import numpy as np
import time
#import scipy as sp

oldtime = time.time() # Measuring performance
T = 8  #Number of trials per mini-block

# Initiate the MDP as an instance of the clas actinfClass
MDP_temp = cbet.betMDP()
MDP_temp.setMDP()
mdp = af.MDPmodel(MDP_temp.MDP)

obs = np.zeros(T, dtype=int)    # Observations at time t
act = np.zeros(T, dtype=int)    # Actions at time t
sta = np.zeros(T, dtype=int)    # Real states at time t
bel = np.zeros((T,mdp.Ns))      # Inferred states at time t

sta[0] = np.nonzero(mdp.S)[0][0]
# Some dummy initial values:
PosteriorLastState = 1
PastAction = 1
PriorPrecision = 1
for t in xrange(T-1):
    # Sample an observation from current state
    obs[t] = mdp.sampleNextObservation(sta[t])
    # Update beliefs over current state and posterior over actions and precision
    bel[t,:], P, W = mdp.posteriorOverStates(obs[t], t, mdp.V,
                                            PosteriorLastState, PastAction,
                                            PriorPrecision)
    # Sample an action and the next state using posteriors
    act[t], sta[t+1] = mdp.sampleNextState( sta[t], P)
    # Prepare inputs for next iteration
    PosteriorLastState = bel[t]
    PastAction = act[t]
    PriorPrecision = W
    
