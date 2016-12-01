#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:42:02 2016

@author: dario
"""
from pymc3.distributions import Discrete
from pymc3.distributions import HalfNormal, Binomial, Uniform, Normal
from pymc3 import Model, find_MAP
import theano as th
import actinfThClass as afOp
from theano import tensor as T

#%%

import betClass as bc
import numpy as np

mabes = bc.betMDP()
lnc = mabes.lnC
gammi = mabes.gamma
lambd = mabes.lambd
alpha = mabes.alpha
beta = mabes.beta
v = mabes.V.astype('int8')
s = mabes.S
h = mabes.H
b = mabes.B
ct = 1

v1 = v[v[:,ct]==0]
v2 = v[v[:,ct]==1]

# Generate data
mabes.exampleFull()
s = np.zeros(s.shape)
s[mabes.Example['RealStates'][ct]] = 1
Y = np.random.choice([0,1], 20, replace=True, p=mabes.Example['PostActions'][ct])
print(mabes.Example['PostActions'][ct])
print(mabes.Example['RealStates']%mabes.nS)
print(Y.sum())

#%%
#actinf.grad = lambda *x: x[0]

mu_to_lnc = afOp.lnC_normal(mabes.nP, mabes.Ns)




actinf_model = Model()

with actinf_model:

    gammi = HalfNormal('gammi', sd=10)
#    lnC = Uniform('lnC', lower=-16, upper=0, shape=(mabes.Ns))
    mu = Normal('mu', mu=mabes.thres, sd = 10)
    sd = HalfNormal('sd', sd = 100)
    lnC = mu_to_lnc(T.as_tensor([mu, sd]))
    Y_obs = afOp.actinf('Y_obs', gammi, lnC, lambd, alpha, beta,
                   v[:,ct:], s, h, b, ct, observed=Y)
#    starts = {'gammi_log_':10, 'lnC_interval_': mabes.lnC}

        
#%%
out = []
for i in range(10):
    startln = np.random.uniform(low=-16, high=0, size=(mabes.lnC.shape))
    starts = {'gammi_log_':10, 'lnC_interval_': mabes.lnC}
    map_estimate = find_MAP(start=starts, model=actinf_model)
    out.append(map_estimate)