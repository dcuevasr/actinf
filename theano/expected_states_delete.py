#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:02:34 2016

@author: dario
"""

import betClass as bc
import numpy as np
from theano import tensor as T
import theano as th

mabes2 = bc.betMDP('small')
#pi = mabes2.V[0]
pi = np.array([0,1,1,0,0,0,1,0])
xS = np.zeros((mabes2.nT+1, mabes2.Ns))
xS[0,:] = mabes2.S
for t in range(1,mabes2.nT+1):
    xS[t,:] = mabes2.B[pi[t-1],:,:].dot(xS[t-1,:])    

B = T.tensor3('B')   # Transition matrices (nU, nS, nS)
S = T.vector('S')   # Current state
Pi = T.vector(name='Pi', dtype='int8') # Policy to evaluate
#xS = T.matrix('xS') # Final state of a policy ([t, nS])
C = T.matrix('C')   # Goals of actinf
#lnC = T.matrix('Crep')  # Repetition of lnC vector for all trials.
V = T.matrix('V', dtype='int8')   # All policies for actinf
xS3 = T.tensor3('xS_all')  # All expected states for all policies
H = T.vector('H')  # Term from Actinf
t = T.scalar('t', dtype='int8')
nT = T.scalar('nT')

def next_state(action, S, B):
    return th.dot(B[action],S)
expected_states, update = th.scan(fn=next_state,
                         sequences=Pi,
                         outputs_info=S,
                         non_sequences=B)
get_xS_one = th.function(inputs=[Pi, S, B], outputs=expected_states)

#mabes2 = bc.betMDP('small')
#nt = mabes2.nT
#tt = 0
b = mabes2.B
#c = mabes2.C
#crep = np.tile(c, (nt,1))
v = mabes2.V.astype('int8')
s = mabes2.S
#h = mabes2.H

xS_one = get_xS_one(v[0,:], s, b)