# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:10:21 2016

@author: dario

Script to test actinfClass
"""
import actinfClass as af
import cbet as cbet
import numpy as np
import time
#import scipy as sp

oldtime = time.time() # Measuring performance
#T = 8  #Number of trials per mini-block


# Initiate the MDP as an instance of the class actinfClass
mdp = cbet.betMDP('kolling compact')
mdp.setMDPMultVar()
actf = af.MDPmodel(mdp)

T = mdp.V.shape[1]

obs = np.zeros(T, dtype=int)    # Observations at time t
act = np.zeros(T, dtype=int)    # Actions at time t
sta = np.zeros(T, dtype=int)    # Real states at time t
bel = np.zeros((T,actf.Ns))      # Inferred states at time t

sta[0] = np.nonzero(actf.S)[0][0]
# Some dummy initial values:
PosteriorLastState = 1
PastAction = 1
PriorPrecision = 1
for t in xrange(T-1):
    # Sample an observation from current state
    obs[t] = actf.sampleNextObservation(sta[t])
    # Update beliefs over current state and posterior over actions and precision
    bel[t,:], P, W = actf.posteriorOverStates(obs[t], t, actf.V,
                                            PosteriorLastState, PastAction,
                                            PriorPrecision)
    # Sample an action and the next state using posteriors
    act[t], sta[t+1] = actf.sampleNextState( sta[t], P)
    # Prepare inputs for next iteration
    PosteriorLastState = bel[t]
    PastAction = act[t]
    PriorPrecision = W
    print 'trial: %d\n' % t
print "Time of execution of ExampleActinf: %f\n" % (time.time() - oldtime)