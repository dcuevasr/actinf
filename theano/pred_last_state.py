#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:21:40 2016

@author: dario
"""

import betClass as bc
from theano import tensor as T
import theano as th

B = T.tensor3('B')   # Transition matrices (nU, nS, nS)
S = T.vector('S')   # Current state
C = T.vector('C')   # Goals
V = T.matrix('V')   # Collection of all policies (row is policy)
Pi = T.vector('Pi', dtype='int8') # Policy to evaluate
fS = T.vector('fS') # Free energy of a policy

nT = 8
t = 4

def next_state(action, S, B):
    return th.dot(B[action],S)

# Loop to apply the actions of a policy in sequence.
expected_states, upd_expected_states = th.scan(fn=next_state,
                                                 sequences=Pi[t:],
                                                 outputs_info=S,
                                                 non_sequences=B)
# Function to calculate the last expected state given a policy
f_pred_final_state = th.function(inputs=[Pi,B,S], outputs=expected_states[-1])



def kl_diff(fS, C):
    return T.sum(T.abs_(fS - C))

# Loop to calculate the free energy of a policy.
fE, updfE = th.scan(fn=kl_diff, 
                    sequences=fS,
                    non_sequences=C)
# Function to calculate the free energy of one policy
free_energy_policy = th.function(inputs=[fS, C], outputs=fE)


fE_all, updfE_all = th.scan(fn=free_energy_policy, 
                            sequences=V,
                            non_sequences=C)

#free_energy_all = th.function(inputs=[V, C], outputs=fE_all)


#
#mabes = bc.betMDP('small')
#
#b = mabes.B
#s = mabes.S
#c = mabes.C
#v = mabes.V.astype('int8')
#pi = mabes.V[0].astype('int8')
#
#
#calc_final_state = f_pred_final_state(pi, b, s)
#calc_free_energy = free_energy_policy(calc_final_state, c)

#calc_fE_all = free_energy_all(v, c)