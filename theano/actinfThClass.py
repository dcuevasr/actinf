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
        max_mafe = max([self.mafe0, self.mafe1])
        exp_out0 = np.exp(maga[-1]*(self.mafe0 - max_mafe))
        exp_out1 = np.exp(maga[-1]*(self.mafe1 - max_mafe))
        norm_const = exp_out0 + exp_out1
        outputs[0][0] = np.array([exp_out0, exp_out1])/norm_const

    def grad(self, inputs, dCdf):
        GAMMI = inputs[0]
        cnp = self.v.shape[0]


        GAMMA, upd_GAMMA = th.scan(fn=self.__upd_gamma,
                           outputs_info=[GAMMI],
                           non_sequences=[self.mafe, self.lambd, self.alpha, self.beta, cnp],
                           n_steps=4)
        dGdg = T.grad(GAMMA[-1], GAMMI)
        out1 = self.mafe0
        out2 = self.mafe1
        maxout = np.max([out1, out2])

        exp_out1 = np.exp(GAMMA[-1]*(out1 - maxout))
        exp_out2 = np.exp(GAMMA[-1]*(out2 - maxout))
        norm_const = exp_out1 + exp_out2

        Jac1 = (-(out1-out2)*dGdg*
                np.exp(GAMMA[-1]*(out1+out2 - 2*maxout))/(norm_const**2))
        Jac2 = -Jac1
        return Jac1*dCdf[0][0]+ Jac2*dCdf[0][1],

    def __upd_gamma(self, GAMMA, Q, LAMBD, ALPHA, BETA, CNP):
        """ Calculates a single precision update.
        """
        QG = GAMMA.dot(Q)
        QGmax = QG.max()
        u = (QG - QGmax).exp()
        u = u/u.sum()
        new_gamma = ALPHA/(LAMBD*GAMMA + (1 - LAMBD)*(BETA - u.dot(Q)/CNP))
        # TODO: Apply convergence override
#        epsilon = (new_gamma - GAMMA).__abs__()/GAMMA
        return new_gamma,# th.scan_module.until(epsilon<0.01)

