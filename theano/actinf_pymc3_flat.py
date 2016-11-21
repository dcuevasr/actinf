#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:01:58 2016

@author: dario
"""

from theano import as_op
import pred_last_state4 as pls
from theano import tensor as T
import numpy as np
import theano as th

from pymc3.distributions import Discrete
from pymc3.distributions import BetaBinomial, HalfNormal
from pymc3 import Model, find_MAP

th.config.compute_test_value = 'raise'

@as_op([T.dvector, T.dscalar, T.dscalar, T.dscalar, T.dscalar,
        T.bmatrix, T.dvector, T.dvector, T.dtensor3, T.bscalar],
        [T.dvector])
def actinf_post(lnc, gammi, lambd, alpha, beta, v, s, h, b, ct):
    return pls.posterior_over_actions(lnc, gammi, lambd, alpha, beta, v, s, h, b, ct)

X = T.vector('X')
X.tag.test_value = [0.1,0.9]
class actinf(Discrete):
    def __init__(self, lnc, gammi, lambd, alpha, beta, v, s, h, b, ct,
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
#        self.mafu = actinf_post(lnc, gammi, lambd, alpha, beta,v, s, h, b, ct)

    def logp(self, action):
        X = actinf_post(self.lnc, self.gammi, self.lambd, self.alpha,
                           self.beta,self.v, self.s, self.h, self.b,
                           self.ct)
        return X[action]



import betClass as bc
from theano.tensor import as_tensor_variable as AT
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
ct = 3

v1 = v[v[:,ct]==0]
v2 = v[v[:,ct]==1]

# Generate data
mabes.exampleFull()
s = np.zeros(s.shape)
s[ct] = mabes.Example['RealStates'][ct]
Y = np.random.choice([0,1], 100, replace=True, p=mabes.Example['PostActions'][ct])


actinf_model = Model()

with actinf_model:

    gammi = HalfNormal('gammi', 10)
#    Y_obs = actinf('Y_obs', lnc, gammi, lambd, alpha, beta, v[:,ct:], s, h, b, ct, observed=Y)

    Y_obs = actinf('Y_obs', AT(lnc, name='lnc'), gammi, AT(lambd, name='lambd'),
                   AT(alpha), AT(beta), AT(v[:,ct:]), AT(s), AT(h), AT(b),
                   AT(ct), observed=Y)

    map_estimate = find_MAP(start={gammi:20}, vars = model.vars)
