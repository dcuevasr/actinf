#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:54:13 2016

@author: dario
"""

import theano as th
from theano import tensor as T
import numpy as np
import utils
from theano import as_op
from pymc3.distributions import Discrete, Binomial
from pymc3 import Model, find_MAP


@as_op([T.bscalar, T.bscalar, T.bmatrix, 
        T.dvector, T.bscalar,
        T.dscalar, T.dtensor3, 
        T.dmatrix, T.dvector, T.dvector, T.dscalar, T.dscalar, T.dscalar],
        [T.dvector])
def posteriorOverStates(Observation, CurrentTime, Policies,
                            PostPrevState, PastAction,
                            PriorPrecision, TransMat, 
                            ObsMat, Entropy, lnC, Alpha, Beta, lambd):
    """
    Decision model for Active Inference. Takes as input the model
    parameters in MDP, as well as an observation and the prior over the
    current state.

    The input PreUpd decides whether the output should be the final
    value of precision (False) or the vector with all the updates for this
    trial (True).
    """
#        print PriorPrecision, self.gamma
    V = Policies
    cNp, fT = np.shape(V)
    w = np.array(range(cNp))
    x = PostPrevState
    W = PriorPrecision
    a = PastAction
    t = CurrentTime

    u = np.zeros(cNp)
    P = np.zeros(V.max()+1)
    B = TransMat
    A = ObsMat
    H = Entropy
    Nu = B.shape[0]
    lnA = np.eye(B.shape[-1])
    # A 'false' set of transition matrices can be fed to the Agent,
    # depending on the newB input above. No input means that the ones from
    # the actinf class are used:
   
    if t==0:
        v = lnA[Observation,:] + np.log(Observation)
    else:
        v = lnA[Observation,:] + np.log(np.dot(B[a],x))
    x = utils.softmax(v)

    Q = np.zeros(cNp)

    for k in range(cNp):
        xt = x
        for j in range(t,fT):
            # transition probability from current state
            xt = np.dot(B[V[k, j],:,:], xt)
            ot = np.dot(A, xt)
            # Predicted Divergence
            Q[k] += H.dot(xt) + (lnC - np.log(ot)).dot(ot)

    # Variational updates: calculate the distribution over actions, then
    # the precision, and iterate N times.

    b = Alpha/W
    for i in range(4):
        # policy (u)
        u[w] = utils.softmax(W*Q)
        # precision (W)
        b = lambd*b + (1 - lambd)*(Beta -
            np.dot(u[w],Q)/cNp)
        W = Alpha/b
    # Calculate the posterior over policies and over actions.
    for j in range(Nu):
        P[j] = np.sum(u[w[utils.ismember(V[:,t],j)]])

    return np.log(P)
    
class actinf(Discrete):
    def __init__(self, gammi, lnc, lambd, alpha, beta, v, s, h, b, a, ct,
                 *args, **kwargs):
        super(actinf, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.gammi = gammi
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.v = v
#        self.v2 = v2
        self.s = s
        self.h = h
        self.b = b
        self.ct = ct
        self.a = a

    def logp(self, action):
        X = posteriorOverStates(self.s.argmax().astype('int8'), self.ct, self.v,
                                self.s, AT(0), 
                                self.gammi, self.b,
                                self.a, self.h, self.lnc, self.alpha, self.beta, self.lambd)
        return X[action]
#        return [(gammi+1)/gammi, (gammi-1)/gammi]

import betClass as bc
from theano.tensor import as_tensor_variable as AT
from scipy import optimize
#import numpy as np

mabes = bc.betMDP('small')
lnc = mabes.lnC
gammi = mabes.gamma
lambd = mabes.lambd
alpha = np.float64(mabes.alpha)
beta = np.float64(mabes.beta)
v = mabes.V.astype('int8')
s = mabes.S.astype('float64')
h = mabes.H
b = mabes.B
a = mabes.A
ct = 3

v1 = v[v[:,ct]==0]
v2 = v[v[:,ct]==1]

# Generate data
mabes.exampleFull()
s = np.zeros(s.shape)
s[ct] = mabes.Example['RealStates'][ct]
Y = np.random.choice([0,1], 10, replace=True, p=mabes.Example['PostActions'][ct])


actinf_model = Model()

with actinf_model:

    gammi = Binomial('gammi', 10, 1.0)
#    Y_obs = actinf('Y_obs', lnc, gammi, lambd, alpha, beta, v[:,ct:], s, h, b, ct, observed=Y)

    Y_obs = actinf('Y_obs', gammi, AT(lnc, name='lnc'), AT(lambd, name='lambd'), 
                   AT(alpha), AT(beta), AT(v[:,ct:]), AT(s), AT(h), AT(b), AT(a),
                   AT(ct), observed=Y)
    
#    map_estimate = find_MAP(start={gammi:20})
map_estimate = find_MAP(model=actinf_model, vars=actinf_model.vars, disp=True, start={gammi:20})
