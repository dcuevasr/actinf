#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:42:02 2016

@author: dario
"""
from pymc3.distributions import Discrete
from pymc3.distributions import HalfNormal
from pymc3 import Model, find_MAP
import theano as th
import actinfThClass as afOp

class actinf(Discrete):
    def __init__(self, gammi, lnc, lambd, alpha, beta, v, s, h,b,ct,
                 *args, **kwargs):
        super(actinf, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.gammi = gammi
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.s = s
        self.h = h
        self.b = b
        self.ct = ct
        self.mafu = afOp.actinf_post_actions(lnc, lambd,
                                             alpha, beta,v, s, h, b, ct)

    def logp(self, action):
        return self.mafu(self.gammi)[action]


import betClass as bc
import numpy as np

mabes = bc.betMDP('small')
lnc = mabes.lnC
gammi = mabes.gamma
lambd = mabes.lambd
alpha = mabes.alpha
beta = mabes.beta
v = mabes.V.astype('int8')
s = mabes.S
h = mabes.H
b = mabes.B
ct = 3

v1 = v[v[:,ct]==0]
v2 = v[v[:,ct]==1]

# Generate data
mabes.exampleFull()
s = np.zeros(s.shape)
s[mabes.Example['RealStates'][ct]] = 1
Y = np.random.choice([0,1], 10, replace=True, p=mabes.Example['PostActions'][ct])

#actinf.grad = lambda *x: x[0]
actinf_model = Model()

with actinf_model:

    gammi = HalfNormal('gammi', sd=10)
    Y_obs = actinf('Y_obs', gammi, lnc, lambd, alpha, beta,
                   v[:,ct:], s, h, b, ct, observed=Y)
    map_estimate = find_MAP(start={gammi:20})