# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:36:14 2016

@author: dario
"""
import numpy as np
import itertools
import utils

def bet(parameters):
    

    # Working parameters. To be imported later
    nU = 3
    nS = 8
    nT = 8
    risklow = 0.9
    riskhigh = 0.3
    rewlow = 1
    rewhigh = 3
    thres = 1
    obsnoise = 0.001
    

    
    
    # Define the observation matrix
    A = np.eye(nS) + obsnoise
    A = A/sum(A)
    
    # Define transition matrices
#    B = [0 for i in range(nU-1)]
    B = np.zeros((nU-1,nS,nS))
    B[0] = np.diag(risklow*np.ones(nS-rewlow),-rewlow)
    np.fill_diagonal(B[0], 1-risklow)
    B[0][-1,(-1-rewlow):-1] = risklow
    B[0][-1,-1] = 1
    
    B[1] = np.diag(riskhigh*np.ones(nS-rewhigh),-rewhigh)
    np.fill_diagonal(B[1], 1-riskhigh)
    B[1][-1,(-1-rewhigh):-1] = riskhigh
    B[1][-1,-1] = 1

    # Define priors over last state (goal)
    C = np.zeros(nS)
    C[-thres:] = 1
    C = C/sum(C)
    
    # Priors over initial state
    D = np.zeros(nS)
    D[0] = 1
    
    # Real initial state
    S = D.astype(int)
    
    # Policies: all combinations of 2 actions
    V = np.array(list(itertools.product(range(0,nU-1),repeat = nT)))
    
    # Preparing inputs for MDP
    MDP = {}
    MDP['A'] = A
    MDP['B'] = B
    MDP['C'] = C
    MDP['D'] = D
    MDP['S'] = S
    MDP['V'] = V
    MDP['alpha'] = 64
    MDP['beta'] = 4
    MDP['lambda'] = 0.005 # or 1/4?
    MDP['gamma'] = 20
    
    return MDP

def bet_mult_var(parameters):    
    # Working parameters. To be imported later
    nU = 3
    nS = 8
    nT = 8
    risklow = 0.9
    riskhigh = 0.3
    rewlow = 1
    rewhigh = 3
    thres = 1
    obsnoise = 0.001
    
    # Default action pairs
    nP = 5;
    pL = np.linspace(0.7,0.9,nP)
    pH = np.linspace(0.4,0.2,nP)
    rL = np.round(np.linspace(1,nS-3,nP))
    rH = np.round(np.linspace(nS,3,nP))
    # Define observation matrix
    A = np.eye(nS*nP) + obsnoise/nP
    
    # Transition matrices
    # num2coords = np.unravel_index
    # coords2num = np.ravel_multi_index
    B = np.zeros((nU,nS*nP,nS*nP))
    for s in xrange(nS):
        for p in xrange(nP):
            nextsL = np.min((s+rL[p],nS-1)).astype(int)
            nextsH = np.min((s+rH[p],nS-1)).astype(int)
            B[0,np.ravel_multi_index([nextsL,p],(nS,nP),order='F'),
              np.ravel_multi_index([s,p],(nS,nP),order='F')] = pL[p]
            B[0,np.ravel_multi_index([s,p],(nS,nP),order='F'),
              np.ravel_multi_index([s,p],(nS,nP),order='F')] = 1 - pL[p]
            B[1,np.ravel_multi_index([nextsH,p],(nS,nP),order='F'),
              np.ravel_multi_index([s,p],(nS,nP),order='F')] = pH[p]
            B[1,np.ravel_multi_index([s,p],(nS,nP),order='F'),
              np.ravel_multi_index([s,p],(nS,nP),order='F')] = 1 - pH[p]
              
    
    for s in xrange(nS):
        allothers = np.array(utils.allothers([[s],range(nP),[0]],(nS,nP)))
        allothers = allothers.astype(int)
        B[np.ix_([2],allothers,allothers)] = 1
        
    # Define priors over last state (goal)
    C = np.zeros(nS*nP)
    allothers = np.array(utils.allothers([[0],range(nP)],(nS,nP)))
    C[allothers] = 1
    C = C/sum(C)
    
    # Priors over initial state
    D = np.zeros(nS*nP)
    D[0] = 1
    
    # Real initial state
    S = D
    
    # Policies: all combinations of 2 actions
    V = np.array(list(itertools.product(range(0,nU-1),repeat = nT)))
    
    # Preparing inputs for MDP
    MDP = {}
    MDP['A'] = A
    MDP['B'] = B
    MDP['C'] = C
    MDP['D'] = D
    MDP['S'] = S
    MDP['V'] = V
    MDP['alpha'] = 64
    MDP['beta'] = 4
    MDP['lambda'] = 0.005 # or 1/4?
    MDP['gamma'] = 20
    
    return MDP
