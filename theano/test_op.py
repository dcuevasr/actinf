#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 01:59:38 2016

@author: dario

Simple example of the creation of an arbitrary, user-defined distribution and 
its use in pymc3's stuff.

"""

import theano as th
import numpy as np

def my_th_fun():
    X = th.tensor.vector('X')
    SCALE = th.tensor.scalar('SCALE')
    
    X.tag.test_value = np.array([1,2,3,4])
    SCALE.tag.test_value = 5.
    
    Scale, upd_sm_X = th.scan(lambda x, scale: scale*(scale+ x),
                               sequences=[X],
                               outputs_info=[SCALE])
    fun_Scale = th.function(inputs=[X, SCALE], outputs=Scale)
    D_out_d_scale = th.tensor.grad(Scale[-1], SCALE)
    fun_d_out_d_scale = th.function([X, SCALE], D_out_d_scale)
    return Scale, fun_Scale, D_out_d_scale, fun_d_out_d_scale

class myOp(th.gof.Op):
    __props__ = ()
    itypes = [th.tensor.dscalar]
    otypes = [th.tensor.dvector]
    def __init__(self, *args, **kwargs):
        super(myOp, self).__init__(*args, **kwargs)
        self.base_dist = np.arange(1,5)
        (self.UPD_scale, self.fun_scale, 
         self.D_out_d_scale, self.fun_d_out_d_scale)= my_th_fun()

    def perform(self, node, inputs, outputs):
        scale = inputs[0]
        updated_scale = self.fun_scale(self.base_dist, scale)
        out1 = self.base_dist[0:2].sum()
        out2 = self.base_dist[2:4].sum()
        maxout = np.max([out1, out2])
        exp_out1 = np.exp(updated_scale[-1]*(out1-maxout))
        exp_out2 = np.exp(updated_scale[-1]*(out2-maxout))
        norm_const = exp_out1 + exp_out2
        outputs[0][0] = np.array([exp_out1/norm_const, exp_out2/norm_const])

    def grad(self, inputs, output_gradients): #working!
        scale = inputs[0]
        X = th.tensor.as_tensor(self.base_dist)
        # Do I need to recalculate all this or can I assume that perform() has
        # always been called before grad() and thus can take it from there?
        # In any case, this is a small enough example to recalculate quickly:
        all_scale, _ = th.scan(lambda x, scale_1: scale_1*(scale_1+ x),
                               sequences=[X],
                               outputs_info=[scale])
        updated_scale = all_scale[-1]
#        updated_scale = 5
        
        
        out1 = self.base_dist[0:1].sum()
        out2 = self.base_dist[2:3].sum()
        maxout = np.max([out1, out2])

        exp_out1 = th.tensor.exp(updated_scale*(out1 - maxout))
        exp_out2 = th.tensor.exp(updated_scale*(out2 - maxout))
        norm_const = exp_out1 + exp_out2

        d_S_d_scale = th.theano.grad(all_scale[-1], scale)
        Jac1 = (-(out1-out2)*d_S_d_scale*
                th.tensor.exp(updated_scale*(out1+out2 - 2*maxout))/(norm_const**2))
        Jac2 = -Jac1
        return Jac1*output_gradients[0][0]+ Jac2*output_gradients[0][1],

#    def grad(self, inputs, output_gradients): #working!
#        return [inputs[0]*2]
        
#    def grad(self, inputs, output_gradients): #Working!
#        invT = inputs[0]
#        X = th.tensor.as_tensor(self.base_dist)
#        Scale, upd_sm_X = th.scan(lambda x, scale: scale*(scale+ x),
#                           sequences=[X],
#                           outputs_info=[invT])
#        D_out_d_scale = th.tensor.grad(Scale[-1], invT)
##        return D_out_d_scale,
#        return D_out_d_scale*output_gradients[0][0],


import pymc3 as pm

class myDist(pm.distributions.Discrete):
    def __init__(self, invT, *args, **kwargs):
        super(myDist, self).__init__(*args, **kwargs)
        self.invT = invT
        self.myOp = myOp()
    def logp(self, value):
        return self.myOp(self.invT)[value]

# Generate some data with the same base_dist as in the class:
base_dist = np.array([3, 7])
real_invT = 0.2
real_dist = np.exp(real_invT*base_dist)
real_dist /= real_dist.sum()

Y_data = np.random.choice([0,1], 100, replace=True, p=real_dist)


from pymc3.distributions import HalfNormal
from pymc3 import Model, find_MAP

my_model = Model()

with my_model:
    invT = HalfNormal('invT', sd = 2)
    Y_obs = myDist('Y_obs', invT=invT, observed=Y_data)
    Y_MAP = find_MAP()