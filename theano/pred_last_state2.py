#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:21:40 2016

@author: dario
"""

import betClass as bc
from theano import tensor as T
import theano as th
import numpy as np

def posterior_over_actions_funs():
    # define theano-happy softmax
#    X = T.vector('X')
    def softmax(X):
        X -= X.max()
        return T.exp(X)/T.sum(T.exp(X))
        
    # Definitions for get_Q_fun:
    B   = T.tensor3('B')                    # Transition matrices (nU, nS, nS)
    S   = T.vector('S')                     # Current state
#    Pi  = T.vector(name='Pi', dtype='int8') # Policy to evaluate
#    xS  = T.matrix('xS')                    # Final state of a policy ([t, nS])
    C   = T.matrix('C')                     # Goals of actinf (one per trial)
    V   = T.matrix('V', dtype='int8')       # All policies for actinf
    xS3 = T.tensor3('xS_all')               # All expected states for all policies
    H   = T.vector('H')                     # Term from Actinf
#    t   = T.scalar('t', dtype='int8')       # Current trial
#    nT  = T.scalar('nT')                    # Number of trials
    
    
    def get_Q_fun():
        """ Calculates Q(pi) for all action sequences
        """
        def get_xS_single(Pi, S, B):
            def next_state(action, S, B):
                return th.dot(B[action],S)
            expected_states, update = th.scan(fn=next_state,
                                     sequences=Pi,
                                     outputs_info=S,
                                     non_sequences=B)
            return expected_states
     
        # Calculate the expected states for each policy:
        xS_all, upd_xS_all = th.scan(fn=get_xS_single,
                                     sequences=V,
                                     non_sequences=[S, B])
        get_xS_all = th.function(inputs=[V, S, B], outputs=xS_all)
        
        
        def free_energy(xS, C, H):
            """ Calculates the free energy for all the policies that were used
            to create the xS input variable
            """
            # TODO: Get rid of the fuzzy factors here
            xS += 0.000000001
        #    xS /= T.sum(xS)
            return T.prod((T.dot(xS, H)) + 
                          T.diag(xS*(T.log(xS)-T.log(C))))
        
        # Calculate the free energy for all the policies:
        fE_all, update_fE_all = th.scan(fn=free_energy, 
                                        sequences=xS3,
                                        non_sequences=[C,H])
        get_fE_all = th.function(inputs=[xS3, C, H], outputs = fE_all)
        return get_xS_all, get_fE_all, fE_all, xS_all
    
    # Definitions for the gamma updates
    get_xS_all, get_fE_all, fE_all, xS_all = get_Q_fun()
    
    GAMMA_INI = T.scalar('GAMMA_INI') # Prior for precision
    GAMMA     = T.vector('GAMMA')     # Precision updates
    LAMBD     = T.scalar('LAMBD')     # Learning rate
    ALPHA     = T.scalar('ALPHA')     # Precision parameter
    BETA      = T.scalar('BETA')      # Precision parameter
    CNP       = T.scalar('CNP')       # Number of available policies
    
    def upd_gamma(GAMMA, Q, LAMBD, ALPHA, BETA, CNP):
        """ Calculates a single precision update.
        """
        u = softmax(GAMMA*Q)
        new_gamma = ALPHA/(LAMBD*GAMMA + (1 - LAMBD)*(BETA - T.dot(u, Q)/CNP))
        epsilon = T.abs_(new_gamma - GAMMA)/GAMMA
        return new_gamma, th.scan_module.until(epsilon<0.01)
        
    
    
    GAMMA, upd_GAMMA = th.scan(fn=upd_gamma, 
                               outputs_info=GAMMA_INI,
                               non_sequences=[fE_all, LAMBD, ALPHA, BETA, CNP],
                               n_steps=16)
    # Function to calculate the precision updates
    get_gamma = th.function(inputs=[GAMMA_INI, fE_all, LAMBD, ALPHA, BETA, CNP],
                            outputs = GAMMA)
    
    INDEX_V0 = T.vector('INDEX_V0', dtype='int8')
    INDEX_V1 = T.vector('INDEX_V1', dtype='int8')
    
    fE0 = T.sum(fE_all[INDEX_V0])
    fE1 = T.sum(fE_all[INDEX_V1])
    set_final_Q = th.function(inputs=[GAMMA_INI, fE_all, LAMBD,
                                      ALPHA, BETA, CNP, fE0, fE1],
                                      outputs = softmax(GAMMA[-1]*[fE0, fE1]))
    return get_xS_all, get_fE_all, get_gamma, set_final_Q

#%% Actual computations 
mabes = bc.betMDP('small')

def posterior_actions(mabes):
    nt = mabes.nT
    tt = 0
    b = mabes.B
    c = mabes.C
    crep = np.tile(c, (nt,1))
    v = mabes.V.astype('int8')
    s = mabes.S
    h = mabes.H
    gamma_ini = mabes.gamma
    lambd = mabes.lambd
    alpha = mabes.alpha
    beta = mabes.beta
    cnp = v[:,0].size
    
    get_xS_all, get_fE_all, get_gamma, set_final_Q = posterior_over_actions_funs()
    xs_all = get_xS_all(v,s,b)
    fe_all = get_fE_all(xs_all, crep, h)
#    gamma_updates = get_gamma(gamma_ini, fe_all, lambd, alpha, beta, cnp)
    
    ind0 = np.arange(v[:,0].size)[v[:,tt]==0]
    ind1 = np.arange(v[:,0].size)[v[:,tt]==1]
    
    fe0 = fe_all[ind0].sum()
    fe1 = fe_all[ind1].sum()
    
    return set_final_Q(gamma_ini, fe_all, lambd, alpha, beta, cnp, fe0, fe1)

#pact0, pact1 = get_post_act(xs_all, crep, h, v, tt)
