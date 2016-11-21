#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:56:02 2016

@author: dario
"""

import theano as th
import theano.tensor as TT
import numpy as np

def softmax(X):
    X -= X.max()
    return TT.exp(X)/TT.sum(TT.exp(X))


B = TT.tensor3('B')
S = TT.vector('S')
V = TT.matrix('V', dtype='int8')
lnC = TT.vector('lnC')
H = TT.vector('H')
i0 = TT.vector('i0', dtype='int8')
i1 = TT.vector('i1', dtype='int8')
t = TT.scalar('t', dtype='int8')
NT = TT.scalar('NT', dtype='int8')

def free_energy(V, S, H, lnC, B):
    cS = S
    cfE = 1
    for pi in range(8):
        cS = TT.dot(B[V[pi]],cS)
        cfE = cfE + (TT.dot(cS, H) + TT.dot(cS,(TT.log(cS)-lnC)))
    return cfE

fE, upd_fE_single = th.scan(fn=free_energy,
                                   sequences=V,
                                   non_sequences=[S,H,lnC,B])
get_fE = th.function(inputs=[V,S,H,lnC,B], outputs=fE)

GAMMA_INI = TT.scalar('GAMMA_INI') # Prior for precision
GAMMA     = TT.vector('GAMMA')     # Precision updates
LAMBD     = TT.scalar('LAMBD')     # Learning rate
ALPHA     = TT.scalar('ALPHA')     # Precision parameter
BETA      = TT.scalar('BETA')      # Precision parameter
CNP       = TT.scalar('CNP')       # Number of available policies
fE_all    = TT.vector('fE_all')    # Eh...

def upd_gamma(GAMMA, Q, LAMBD, ALPHA, BETA, CNP):
    """ Calculates a single precision update.
    """
    u = softmax(GAMMA*Q)
    new_gamma = ALPHA/(LAMBD*GAMMA + (1 - LAMBD)*(BETA - TT.dot(u, Q)/CNP))
    epsilon = TT.abs_(new_gamma - GAMMA)/GAMMA
    return new_gamma, th.scan_module.until(epsilon<0.01)
    


GAMMA, upd_GAMMA = th.scan(fn=upd_gamma, 
                           outputs_info=GAMMA_INI,
                           non_sequences=[fE_all, LAMBD, ALPHA, BETA, CNP],
                           n_steps=16)
get_gamma = th.function(inputs=[GAMMA_INI, fE_all, LAMBD, ALPHA, BETA, CNP],
                        outputs = GAMMA)

def posterior_over_actions(lnc, gamma_ini, lambd, alpha, beta, v, s, h, b, t):
    cnp = v.shape[0]
    mafe = get_fE(v,s,h,lnc,b)
    maga = get_gamma(gamma_ini, mafe, lambd, alpha, beta, cnp)
    return softmax(maga[-1]*np.array([mafe[v[:,t]==0].sum(), mafe[v[:,t]==1].sum()]))


def main(t=0):
    import betClass as bc
    
    mabes = bc.betMDP()
    
    gamma_ini = mabes.gamma
    lambd = mabes.lambd
    alpha = mabes.alpha
    beta = mabes.beta
    v = mabes.V.astype('int8')
    cnp = v.shape[0]
    s = mabes.S
    h = mabes.H
    lnc = mabes.lnC
    b = mabes.B
    nt = mabes.nT
    
    mafE = get_fE(v,s,h,lnc,b)
    
    maga = get_gamma(gamma_ini, mafE, lambd, alpha, beta, cnp)
    
    post = posterior_over_actions(lnc, gamma_ini, lambd, alpha, beta, v, s, h, b, 0)
    
    return mafE, maga, post, mabes