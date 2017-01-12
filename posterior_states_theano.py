#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:56:02 2016

@author: dario
"""

import theano as th
import theano.tensor as TT
import numpy as np

# Test values
import betClass as bc
mabes = bc.betMDP('small')
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


def softmax(X):
    X -= X.max()
    return TT.exp(X)/TT.sum(TT.exp(X))


B = TT.tensor3('B')
S = TT.vector('S')
V = TT.matrix('V', dtype='int8')
lnC = TT.vector('lnC')
H = TT.vector('H')
Pi = TT.vector('Pi', dtype='int8')
leofE = TT.matrix('leofE')

B.tag.test_value = b
S.tag.test_value = s
V.tag.test_value = v
lnC.tag.test_value = lnc
H.tag.test_value = h
Pi.tag.test_value = V[0,:]
leofE.tag.test_value = V.astype(th.config.floatX)

def free_energy(Pi, S, H, lnC, B):
    def one_computation(action, S, H, lnC, B):
        SS = TT.dot(B[action,:,:],S)
        return SS, TT.dot(SS, H) + TT.dot(SS,lnC - TT.log(SS))

    [xS, fE], upds = th.scan(fn=one_computation,
                         sequences=Pi,
                         outputs_info=[S, None],
                         non_sequences=[H, lnC, B])

    return fE.sum(), xS

[ofE, oxS], upd_fE_single = th.scan(fn=free_energy,
                                   sequences=V,
                                   non_sequences=[S,H,lnC,B])
get_fE = th.function(inputs=[V,S,H,lnC,B], outputs=ofE)
get_xS = th.function(inputs=[V,S,H,lnC,B], outputs=oxS) # For testing and dbing

leofE = ofE + ofE

GAMMA_INI = TT.scalar('GAMMA_INI') # Prior for precision
GAMMA     = TT.vector('GAMMA')     # Precision updates
LAMBD     = TT.scalar('LAMBD')     # Learning rate
ALPHA     = TT.scalar('ALPHA')     # Precision parameter
BETA      = TT.scalar('BETA')      # Precision parameter
CNP       = TT.scalar('CNP')       # Number of available policies
fE_all    = TT.vector('fE_all')    # Eh...

GAMMA_INI.tag.test_value = mabes.gamma
GAMMA.tag.test_value = np.zeros(mabes.nT)
LAMBD.tag.test_value = lambd
ALPHA.tag.test_value = alpha
BETA.tag.test_value = beta
CNP.tag.test_value = 0
fE_all.tag.test_value = np.zeros(v.shape[0])

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

#get_post = th.function(inputs=[GAMMA_INI, TT.concatenate(fE1, fE2), LAMBD, ALPHA, BETA, CNP], outputs=[softmax(GAMMA[-1]*[fE1.sum(), fE2.sum()])])

D_GAMMA = TT.grad(GAMMA[-1], GAMMA_INI)
FD_GAMMA_GAMMI= th.function([GAMMA_INI, fE_all, LAMBD, ALPHA, BETA, CNP], D_GAMMA)


# Functions that are no longer used:
#def posterior_over_actions(lnc, gamma_ini, lambd, alpha, beta, v, s, h, b, ct):
#    cnp = v.shape[0]
#    mafe = get_fE(v,s,h,lnc,b)
#    maga = get_gamma(gamma_ini, mafe, lambd, alpha, beta, cnp)
#    dist_out = softmax(maga[-1]*np.array([mafe[v[:,ct]==0].sum(),
#                                         mafe[v[:,ct]==1].sum()]))
#    return np.log(dist_out)
#
#def main(t=0):
#    import betClass as bc
#
#    mabes = bc.betMDP()
#
#    gamma_ini = mabes.gamma
#    lambd = mabes.lambd
#    alpha = mabes.alpha
#    beta = mabes.beta
#    v = mabes.V.astype('int8')
#    cnp = v.shape[0]
#    s = mabes.S + 0.000000
#    h = mabes.H
#    lnc = mabes.lnC
#    b = mabes.B
#
#    mase = get_xS(v,s,h,lnc,b)
#
#    mafE = get_fE(v,s,h,lnc,b)
#
#    maga = get_gamma(gamma_ini, mafE, lambd, alpha, beta, cnp)
#
#    post = posterior_over_actions(lnc, gamma_ini, lambd, alpha, beta, v, s, h, b, 0)
#
#    return mafE, maga, post, mase, mabes