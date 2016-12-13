#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:42:02 2016

@author: dario
"""
from pymc3.distributions import Discrete
from pymc3.distributions import HalfNormal, Binomial, Uniform, Normal
from pymc3 import Model, find_MAP, nuts
import theano as th
import actinfThClass as afOp
from theano import tensor as T
import pymc3 as pm
import betClass as bc
import numpy as np
from scipy import optimize


th.config.exception_verbosity = 'high'
#%%

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
#mabes.exampleFull()
#s = np.zeros(s.shape)
#s[mabes.Example['RealStates'][ct]] = 1
#Y = np.random.choice([0,1], 100, replace=True, p=mabes.Example['PostActions'][ct])
##print(mabes.Example['PostActions'][ct])
##print(mabes.Example['RealStates']%mabes.nS)
#print(Y.sum())


#%% Alt-data
nD = 2
nS = mabes.nS
nT = mabes.nT
data = np.zeros(nD*nT, dtype=np.float64)
trial = np.zeros(nD*nT, dtype=np.int64)
state = np.zeros(nD*nT, dtype=np.int64)

for d in range(nD):
    mabes.exampleFull()
    data[d*nT:(d+1)*nT] = mabes.Example['Actions']
    trial[d*nT:(d+1)*nT] = range(nT)
    state[d*nT:(d+1)*nT] = mabes.Example['RealStates']

stateV = np.zeros((nD*nT, mabes.Ns))

for n, st in enumerate(state):
    stateV[n, st] = 1


#%%
#actinf.grad = lambda *x: x[0]

mu_to_lnc = afOp.lnC_normal(mabes.nP, mabes.nS)




actinf_model = Model()

with actinf_model:

#    gammi = HalfNormal('gammi', sd=50)
    gammi = 20
#    lnC = Uniform('lnC', lower=-16, upper=0, shape=(mabes.Ns))
    mu = Normal('mu', mu=20, sd = 30)
    sd = HalfNormal('sd', sd = 100)
    lnC = mu_to_lnc(T.as_tensor([mu, sd]))
    Y_obs = afOp.actinfDist('Y_obs', gammi, lnC, lambd, alpha, beta,
                   v, stateV, h, b, trial, observed=data)
#%%
start =  {'mu': 20, 'sd_log_':30}
map_estimate = find_MAP(model=actinf_model, start = start)
#%%
#manuts = nuts.NUTS(state=map_estimate, model=actinf_model)
#    starts = {'gammi_log_':10, 'lnC_interval_': mabes.lnC}
#%%
#trace = pm.sample(2000, manuts, model=actinf_model)
#%%
#map_estimate = find_MAP(model=actinf_model)

##%%
#out = []
#for i in range(10):
#    startln = np.random.uniform(low=-16, high=0, size=(mabes.lnC.shape))
##    starts = {'gammi_log_':10, 'lnC_interval_': mabes.lnC}
##    map_estimate = find_MAP(start=starts, model=actinf_model)
#    map_estimate = find_MAP
#    out.append(map_estimate)