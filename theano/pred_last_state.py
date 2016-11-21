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
Pi = T.vector(name='Pi', dtype='int8') # Policy to evaluate
xS = T.matrix('xS') # Final state of a policy ([t, nS])
C = T.vector('C')   # Goals of actinf
V = T.matrix('V', dtype='int8')   # All policies for actinf
xS_all = T.tensor3('xS_all')  # All expected states for all policies
nT = 8
t = 0

def next_state(action, S, B):
    return th.dot(B[action],S)


expected_states, update = th.scan(fn=next_state,
                         sequences=Pi,
                         outputs_info=S,
                         non_sequences=B)
f_pred_final_state = th.function(inputs=[Pi, S, B], outputs=expected_states)

def free_energy(xS, C):
    return T.sum(T.abs_(xS - C))

fE_single, updfE_single = th.scan(fn=free_energy, 
                    sequences=xS,
                    non_sequences=C)
free_energy_policy= th.function(inputs=[xS, C], outputs=fE_single.prod())

xS_all, update_xS_all = th.scan(fn=f_pred_final_state, 
                                sequences=V,
                                non_sequences=[S, B])
#f_xS_all = th.function(inputs=[V, S, B], outputs=xS_all)

#fS_all, upd_fS_all = th.scan(fn=free_energy_policy, 
#                             sequences=[V, xS_all],
#                             non_sequences=B)
#f_fS_all = th.function(inputs=[V, xS_all, B], outputs=fS_all)

#mabes = bc.betMDP('small')
#b = mabes.B
#c = mabes.C
#v = mabes.V.astype('int8')
#s = mabes.S
#
#c_exp_states = f_pred_final_state(v[0], s, b)
#c_fE_pol1 = free_energy_policy_seq(c_exp_states, c)
