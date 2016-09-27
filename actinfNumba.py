# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:01:24 2016

@author: dario
"""
import numpy as np
import scipy as sp
import numba as nb


@nb.jit(nopython = True)
def softmax(A):

    maxA = A.max()
    A = A - maxA
    A = np.exp(A)
    A = A/np.sum(A)
    return A


jittypes = ('Tuple((float64[:,:], float64[:,:],int64[:,:], float64[:,:], ' +
        'int64[:,:], float64[:], int64[:], int16[:]))' +
        '(float64[:,:], float64[:,:,:], float64[:], float64[:], int64[:,:], ' +
        'float64[:])')




#@nb.jit('void(float64[:,:], float64[:,:,:], float64[:], float64[:], int16[:,:], float64[:])',nopython = True)
@nb.jit(jittypes, nopython = True)
def single(A, B, C, D, V, iS):
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
    T = V.shape[1]

    # Read some numbers from the inputs
    Ns = B.shape[1] # Number of hidden states
    Nu = B.shape[0] # Number of actions
    p0 = sp.exp(-16)           # Smallest probability

    A = A + p0
    No = A.shape[0] # Number of outcomes
#    A = np.dot(A,np.diag(1/np.sum(A,0)))
    lnA = np.log(A)
    H = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        H[i] = np.sum(A[i,:]*lnA[i,:])
#    H = np.sum(A*lnA,0)

    # transition probabilities
    B = B + p0

    Bsum = np.zeros(Ns)
    for b in range(B.shape[0]):
        for bcol in range(Ns):
            Bsum[bcol] = np.sum(B[b,:,bcol])
            for brow in range(Ns):
                B[b,brow,bcol] /= Bsum[bcol]
#        B[b] = B[b]/Bsum

    # priors over last state (goals)
    C = C + p0
    C = C/np.sum(C)
    lnC = np.log(C)

    # priors over initial state
    D = D + p0
    D = D/np.sum(D)
    lnD = np.log(D)
#
    # policies and their expectations
    Np = V.shape[0]
    w = np.arange(Np)

    # initial states and outcomes
    q = np.argmax(sp.dot(A,iS))
    s = np.zeros((T), dtype = np.int64)
    s[0] = np.nonzero(iS==1)[0][0]
    o = np.zeros((T), dtype = np.int64)
    o[0] = q
    S = np.zeros((Ns,T))
    S[s[0]][0] = 1
    O = np.zeros((No,T), dtype = np.int64)
    O[q][0] = 1
    U = np.zeros((Nu,T), dtype = np.int64)
    P = np.zeros((Nu,T))
    x = np.zeros((Ns,T))
    u = np.zeros((Np,T))
    a = np.zeros((T), dtype = np.int16)
    W = np.zeros((T))

    #solve
    gamma = np.zeros(T*N, dtype = np.float64)
    b = alpha/g
    for t in range(T):
        # Expectations of allowable policies (u) and current state (x)
        if t>0:
            #retain allowable policies (consistent with last action)
            j = (V[:,t-1] == a[t-1])
            V = V[j,:]
            w = w[j]

            #current state (x)
            v = lnA[o[t]] + np.log(sp.dot(B[a[t-1]],x[:,t-1]))
            x[:,t] = softmax(v)
        else:
            u[:,t] = np.ones(Np)/Np
            v = lnA[int(o[t]),:] + lnD
            x[:,t] = softmax(v)
        # value of policies (Q)
        cNp = V.shape[0]
        Q = np.zeros(cNp)
        for k in range(cNp):
            # path integral of expected free energy (...)
            xt = x[:,t]
            for tf in range(t,T):
                # transition probability from current state
                xt = sp.dot(B[V[k,tf]],xt)
                ot = sp.dot(A,xt)

                # predicted divergence
                Q[k] += sp.dot(H,xt) + sp.dot(lnC[:] - np.log(ot),ot)
        # Variational iterations
        for i in range(N):
            # policy (u)
            u[w,t] = softmax(W[t]*Q)
            # precision (W)
            b = lambd*b + (1 - lambd)*(beta - np.dot(u[w,t],Q))
            W[t] = alpha/b
            #simulated dopamine responses (precision as each iteration)
            gamma[t*N + i] = W[t]

        for j2 in range(Nu):
            for k in range(t,T):
                P[j2,k] = np.sum(u[w[V[:,k] == j2],t])
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
#    oMDP = {'P': P, 'Q': x, 'O': O, 'S': S, 'U': U, 'W': W, 's': s, 'a': a}

    return P, x, O, S, U, W, s, a