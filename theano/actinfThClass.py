#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:56:06 2016

@author: dario
"""

import theano as th
from theano import tensor as T
import pred_last_state4 as postA
import numpy as np
from utils import softmax


class actinf_post_actions(th.gof.Op):
    __props__ = ()
    itypes = [T.dscalar]
    otypes = [T.dvector]
    check_input = True
    
    def __init__(self, lnc, lambd, alpha, beta, v, s, h, b, ct, *args, **kwargs):
        super(actinf_post_actions, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.v = v
        v0 = (v[:,ct]==0).nonzero()
        v1 = (v[:,ct]==1).nonzero()
        self.s = s
        self.h = h
        self.b = b
        self.ct = ct
        self.mafe = postA.get_fE(v, s, h, lnc, b)
        self.mafe0 = self.mafe[v0].sum()
        self.mafe1 = self.mafe[v1].sum()
        # See the grad() method to see what these two do:
        self.perform_count = 0
        self.grad_count = 0

    def perform(self, node, inputs, outputs):
        """ Calculates the posterior over actions for active inference.
        """
        gammi = inputs[0]
        cnp = self.v.shape[0]
        maga = postA.get_gamma(gammi, self.mafe, self.lambd,
                               self.alpha, self.beta, cnp)
        outputs[0][0] = softmax(maga[-1]*np.array([self.mafe0, self.mafe1]))

    def grad(self, inputs, dCdf):
        GAMMI = inputs[0]
        cnp = self.v.shape[0]
        
        GAMMA, upd_GAMMA = th.scan(fn=postA.upd_gamma,
                           outputs_info=GAMMI,
                           non_sequences=[self.mafe, self.lambd, self.alpha, self.beta, cnp],
                           n_steps=16)
        dGdg = T.grad(GAMMA[-1], GAMMI)
        out1 = self.mafe0
        out2 = self.mafe1
        maxout = np.max([out1, out2])

        exp_out1 = th.tensor.exp(GAMMA[-1]*(out1 - maxout))
        exp_out2 = th.tensor.exp(GAMMA[-1]*(out2 - maxout))
        norm_const = exp_out1 + exp_out2

        Jac1 = (-(out1-out2)*dGdg*
                th.tensor.exp(GAMMA[-1]*(out1+out2 - 2*maxout))/(norm_const**2))
        Jac2 = -Jac1        
        return Jac1*dCdf[0][0]+ Jac2*dCdf[0][1],

# They were made for testing purposes and therefore not necessary for the
# proper functoining of the whole thing. I keep them just in case they are
# needed again:

#def __define_function():
#    lnC = T.vector('lnC')
#    GAMi = T.scalar('GAMi')
#    LAMBD = T.scalar('LAMBD')
#    ALPHA = T.scalar('ALPHA')
#    BETA = T.scalar('BETA')
#    V = T.matrix('V', dtype='int8')
#    S = T.vector('S')
#    H = T.vector('H')
#    B = T.tensor3('B')
#    cT = T.scalar('cT', dtype='int8')
##    actinf = actinf_post_actions()
#    v1, v2 = T.vectors('v1', 'v2')
#    o1 = test_th_class()(v1, v2)
#
##    X = [lnC, GAMi, BETA, V, S, H, B ,cT]
##
#    return th.function(inputs=[lnC, GAMi, LAMBD, ALPHA, BETA, V, S, H, B ,cT],
#                       outputs=actinf_post_actions()(lnC, GAMi, LAMBD, ALPHA, BETA, V, S, H, B ,cT))
##    return th.function(inputs=[GAMi], outputs=[GAMi+1])
##    return th.function(inputs=[lnC, GAMi], outputs=o1)
#
#def __main():
#    import betClass as bc
#    mabes = bc.betMDP()
#
#    lnc = mabes.lnC
#    gami = mabes.gamma
#    lambd = mabes.lambd
#    alpha = mabes.alpha
#    beta = mabes.beta
#    v = mabes.V.astype('int8')
#    s = mabes.S
#    h = mabes.H
#    b = mabes.B
#    ct = 0
#
#    mafu = define_function()
#
#    evalfu = mafu(lnc, gami, lambd, alpha, beta, v, s, h, b, ct)
#
#    return mafu, evalfu
#
#
#class __test_th_class(th.Op):
#    __props__ = ()
#
##    itypes = [T.vector(), T.vector()]
##    otypes = [T.vector()]
#    def make_node(self, in1, in2):
#        return th.gof.Apply(self, [in1, in2], [T.vector()])
#
#    def perform(self, node, inputs, outputs):
#        V, v = inputs
#        outputs[0][0] = v+V
#
#    check_input = True
#def __define_test():
#    INP1, INP2 = T.vectors('inp1','inp2')
#    return th.function(inputs=[INP1, INP2], outputs=test_th_class()(INP1, INP2))