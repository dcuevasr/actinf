# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:01:24 2016

@author: dario
"""
import numpy as np
import scipy as sp
import utils


def single(MDP):
    """
    Original implementation of the Active Inference by the Friston group. 
    Translated to Python by Dario Cuevas.
    """
    # Some stuff
    alpha = 8
    beta = 4
    g = 1
    lambd = 0
    N = 4
    T = np.shape(MDP['V'])[1]
    
    # Read some numbers from the inputs
    Ns = np.shape(MDP['B'])[1] # Number of hidden states
    Nu = np.shape(MDP['B'])[0] # Number of actions
    p0 = sp.exp(-16)           # Smallest probability
    
    A = MDP['A'] + p0
    No = np.shape(MDP['A'])[0] # Number of outcomes
    A = sp.dot(A,np.diag(1/np.sum(A,0)))
    lnA = sp.log(A)
    H = np.sum(A*lnA,0)
    
    # transition probabilities
    B = MDP['B'] + p0
    for b in xrange(np.shape(B)[0]):
        B[b] = B[b]/np.sum(B[b],axis=0)
        
    # priors over last state (goals)
    C = sp.tile(MDP['C'],(No,1)).T + p0
    C = C/np.sum(C)
    lnC = sp.log(C)
    
    # priors over initial state
    D = MDP['D']
    D = D + p0
    D = D/np.sum(D)
    lnD = sp.log(D)
    
    # policies and their expectations
    V = MDP['V']
    Np = np.shape(V)[0]
    w = np.array(range(Np))
    
    # initial states and outcomes
    q = np.argmax(sp.dot(A,MDP['S']))
    s = np.zeros((T))
    s[0] = np.nonzero(MDP['S']==1)[0]
    o = np.zeros((T))
    o[0] = q
    S = np.zeros((Ns,T))
    S[s[0]][0] = 1
    O = np.zeros((No,T))
    O[q][0] = 1
    U = np.zeros((Nu,T))
    P = np.zeros((Nu,T))
    x = np.zeros((Ns,T))
    u = np.zeros((Np,T))
    a = np.zeros((T))
    W = np.zeros((T))
    
    #solve
    gamma = []
    b = alpha/g
    for t in xrange(T):
        # Expectations of allowable policies (u) and current state (x)
        if t>0:
            # retain allowable policies (consistent with last action)
            j = utils.ismember(V[:,t-1], a[t-1])
            V = V[j,:]
            w = w[j]
            
            # current state (x)
            v = lnA[o[t]] + sp.log(sp.dot(B[a[t-1]],x[:,t-1]))
            x[:,t] = utils.softmax(v)
        else:
#            pdb.set_trace()
            u[:,t] = np.ones(Np)/Np
            v = lnA[int(o[t]),:] + lnD
            x[:,t] = utils.softmax(v)
            
        # value of policies (Q)
        cNp = np.shape(V)[0]
        Q = np.zeros(cNp)
        for k in xrange(cNp):
            # path integral of expected free energy (...)
            xt = x[:,t]
            for j in xrange(t,T):
                # transition probability from current state
                xt = sp.dot(B[V[k,j]],xt)
                ot = sp.dot(A,xt)
                
                # predicted divergence
                Q[k] += sp.dot(H,xt) + sp.dot(lnC[:,j] - sp.log(ot),ot)
                
        # Variational iterations
        for i in xrange(N):
            # policy (u)
            u[w,t] = utils.softmax(sp.dot(W[t],Q))
            # precision (W)
            b = lambd*b + (1 - lambd)*(beta - sp.dot(u[w,t],Q))
            W[t] = alpha/b
            #simulated dopamine responses (precision as each iteration)
            gamma.append(W[t])
            
        for j in xrange(Nu):
            for k in xrange(t,T):
                P[j,k] = np.sum(u[w[utils.ismember(V[:,k],j)],t])
        # next action
        a[t] = np.nonzero(np.random.rand(1) < np.cumsum(P[:,t]))[0][0]
        # save action
        U[a[t],t] = 1
        
        # sampling of next state (outcome)
        if t<T-1:
            #next sampled state
            s[t+1] = np.nonzero(np.random.rand(1) < 
                np.cumsum(B[a[t],:,s[t]]))[0][0]
            #next observed state
            o[t+1] = np.nonzero(np.random.rand(1) <
                np.cumsum(A[:,s[t+1]]))[0][0]
            # save the outcome and state sampled
            W[t+1] = W[t]
            O[o[t+1]][t+1] = 1
            S[s[t+1],t+1] = 1
    oMDP = {}
    oMDP['P'] = P
    oMDP['Q'] = x
    oMDP['O'] = O
    oMDP['S'] = S
    oMDP['U'] = U
    oMDP['W'] = W
    oMDP['s'] = s
    oMDP['a'] = a
    
    return oMDP

def multi(iMDP, trials):
    rMDP = [{} for i in xrange(trials)]
    for i in xrange(trials):
        rMDP[i] = single(iMDP)
    return rMDP