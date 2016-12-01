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
from pymc3.distributions import Discrete
import pred_last_state4 as postA
import numpy as np
import copy

#from utils import softmax

#%%
class actinf_post_actions(th.gof.Op):
    """ Theano op to calculate the posteriors over actions in active inference,
    as well as its gradient wrt the inputs.
    """
    __props__ = ()
    itypes = [T.dvector]
    otypes = [T.dvector]
    check_input = True

    def __init__(self, lnc, lambd, alpha, beta, v, s, h, b, ct, *args, **kwargs):
        super(actinf_post_actions, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.v0 = (v[:,ct]==0).nonzero()
        self.v1 = (v[:,ct]==1).nonzero()
        self.s = s
        self.h = h
        self.b = b
        self.ct = ct
        # See the grad() method to see what these two do:
        self.perform_count = 0
        self.grad_count = 0

    def perform(self, node, inputs, outputs):
        """ Calculates the posterior over actions for active inference.
        """
        lnc = inputs[0][1:]
        gammi = inputs[0][0]

        mafe = postA.get_fE(self.v, self.s, self.h, lnc, self.b)
        mafe0 = mafe[self.v0].sum()
        mafe1 = mafe[self.v1].sum()
        cnp = self.v.shape[0]
        maga = postA.get_gamma(gammi, mafe, self.lambd,
                               self.alpha, self.beta, cnp)
        max_mafe = max([mafe0, mafe1])
        exp_out0 = np.exp(maga[-1]*(mafe0 - max_mafe))
        exp_out1 = np.exp(maga[-1]*(mafe1 - max_mafe))
        norm_const = exp_out0 + exp_out1
        outputs[0][0] = np.array([exp_out0, exp_out1])/norm_const

    def grad(self, inputs, dCdf):
        lnC = inputs[0][1:]
        GAMMI = inputs[0][0]

        cnp = self.v.shape[0]

        # Gradient of fE wrt the priors over final state
        [ofE, oxS], upd_fE_single = th.scan(fn=self._free_energy,
                                   sequences=self.v,
                                   non_sequences=[self.s,self.h,lnC,self.b])
        ofE0 = ofE[self.v0].sum()
        ofE1 = ofE[self.v1].sum()

        dFE0dlnC = T.jacobian(ofE0, lnC)
        dFE1dlnC = T.jacobian(ofE1, lnC)
        dFEdlnC  = T.jacobian(ofE,  lnC)
        
        ofE_ = T.vector()
        ofE_.tag.test_value = ofE.tag.test_value
        # Gradient of Gamma with respect to its initial condition:
        GAMMA, upd_GAMMA = th.scan(fn=self._upd_gamma,
               outputs_info=[GAMMI],
               non_sequences=[ofE, self.lambd, self.alpha, self.beta, cnp],
               n_steps=4)
        dGdg = T.grad(GAMMA[-1], GAMMI)
        
        dGdfE = T.jacobian(GAMMA[-1], ofE)
        dGdlnC = dGdfE.dot(dFEdlnC)
        
        
        out1 = ofE0
        out2 = ofE1
        maxout = T.max([out1, out2])

        exp_out1 = T.exp(GAMMA[-1]*(out1 - maxout))
        exp_out2 = T.exp(GAMMA[-1]*(out2 - maxout))
        norm_const = exp_out1 + exp_out2
        
        # Derivative wrt the second output (gammi):
        Jac1_gammi = (-(out1-out2)*dGdg*
                T.exp(GAMMA[-1]*(out1+out2 - 2*maxout))/(norm_const**2))
        Jac2_gammi = -Jac1_gammi
#        dfd1_tZ = Jac1_gammi*dCdf[1][0]+ Jac2_gammi*dCdf[1][1]

        # Derivative wrt first input (lnc)
        Jac1_lnC = (T.exp(GAMMA[-1]*(out1 + out2 - 2*maxout))/(norm_const**2)*
                  (-dGdlnC*(out1 - out2) - GAMMA[-1]*(dFE0dlnC - dFE1dlnC)))
        Jac2_lnC = -Jac1_lnC
        
        Jac1 = T.concatenate([T.stack(Jac1_gammi), Jac1_lnC])
        Jac2 = T.concatenate([T.stack(Jac2_gammi), Jac2_lnC])
        
        return Jac1*dCdf[0][0] + Jac2*dCdf[0][1], 

    def _free_energy(self, Pi, S, H, lnC, B):
        def one_computation(action, S, H, lnC, B):
            SS = B[action,:,:].dot(S)
            return SS, T.dot(SS, H) + T.dot(SS,lnC - T.log(SS))
    
        [xS, fE], upds = th.scan(fn=one_computation,
                             sequences=Pi,
                             outputs_info=[S, None],
                             non_sequences=[H, lnC, B])
    
        return fE.sum(), xS

    def _upd_gamma(self, GAMMA, Q, LAMBD, ALPHA, BETA, CNP):
        """ Calculates a single precision update.
        """
        QG = GAMMA.dot(Q)
        QGmax = QG.max()
        u = (QG - QGmax).exp()
        u = u/u.sum()
        new_gamma = ALPHA/(LAMBD*GAMMA + (1 - LAMBD)*(BETA - u.dot(Q)/CNP))
        # TODO: Apply convergence override
#        epsilon = (new_gamma - GAMMA).__abs__()/GAMMA
        return new_gamma#, th.scan_module.until(epsilon<0.01)

class actinf(Discrete):
    """ pymc3 distribution for the posteriors over actions in active inference.
    It takes as inputs the usual MDP stuff from betClass or actinfClass.
    """
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
        self.mafu = actinf_post_actions(lnc, lambd,
                                             alpha, beta,v, s, h, b, ct)

    def logp(self, action):
        return self.mafu(T.concatenate([T.stack(self.gammi), self.lnc]))[action]


#%%
class lnC_normal(th.gof.Op):
    """ Theano Op to define the priors over last state (logC in the model) as a 
    repetitions (sum, really) of offset normals.
    """
    itypes=[T.dvector]
    otypes=[T.dvector]
    __props__ = ()
    
    def __init__(self, nP, nS, *args, **kwargs):
        super(lnC_normal, self).__init__(*args, **kwargs)
        self.nP = nP
        self.nS = nS
        self.Ns = nP*nS # Total number of elements in the output vector
        self._normal()
    
    def perform(self, node, inputs, outputs):
        """ Returns the vector of size Ns=nS*nP for lnC
        """
        mu = inputs[0][0]
        sd = inputs[0][1]
        
#        Ns = self.nS*self.nP
#        x = range(Ns)
        sum_norms = self.Norm(mu, sd)
        
        outputs[0][0] = sum_norms/sum_norms.sum()
    
    def grad(self, inputs, dCdf):
        """ Gradient MTF
        """
        MU = inputs[0][0]
        SD = inputs[0][1]
#        Y = self._normal(just_return = True, MU=MU, SD=SD)
        Y, Y_upd = th.scan(fn=self.norm_fun,
                               sequences=self.counter, non_sequences=[MU, SD])


        dYdMIn = T.jacobian(Y.sum(axis=0), inputs[0])
#        dYdSD = T.jacobian(Y, SD)
#        return dYdMIn[0]*dCdf[0][0] + dYdMIn[1]*dCdf[0][1], 
#        return T.as_tensor([dCdf[0][0]*dYdMIn[0][0] + dCdf[0][1]*dYdMIn[1][0],
#                dCdf[0][0]*dYdMIn[0][1] + dCdf[0][1]*dYdMIn[1][1]]),
        return T.as_tensor([dCdf[0].dot(dYdMIn[:,0]), dCdf[0].dot(dYdMIn[:,1])])
    
    def _normal(self):
        """ Normal distribution.
        """
        nP = self.nP
        nS = self.nS
        
        # Auxiliary Theano variables
        MU = T.scalar('MU')
        SD = T.scalar('SD')
        
        x = np.arange(nP*nS)
        # Test values
#        X.tag.test_value = np.arange(nP*nS)
        MU.tag.test_value = nS
        SD.tag.test_value = 10
#        Y0 = T.vector('Y0')
        counter = np.arange(nP)

        Y = lambda n, mu, sd: 1/T.sqrt(2*np.pi*sd**2)*T.exp(-(x - (mu + n*nS))**2/(2*sd**2))
        Y_all, Y_upd = th.scan(fn=Y, sequences=counter, non_sequences=[MU, SD])

        fun_Y = th.function(inputs=[MU, SD], outputs=Y_all.sum(axis=0))
        self.Norm = fun_Y
        self.norm_fun = Y #For using in Grad
        self.counter = counter #For using in Grad

#%%        
#def gaussian(x, mu, sd):
#    return (2*np.pi*sd**2)**(-1/2)*np.exp(-(x - mu)**2/(2*sd**2))
#    
#def dGdMu(x, mu, sd):
#    return (x-mu)/sd**2*gaussian(x, mu, sd)
#    
#def dGdSd(x, mu, sd):
#    return (-sd**(-1) + (x-mu)**2/sd**3)*gaussian(x, mu, sd)
        
#        #%% Test grad
#Ns = 100
#nP = 1
#mu = 20.
#sd = 10.
#
#A = T.vector('A')
#B = T.scalar('B')
#C = T.vector('C')
#inputs = [A, B]
#ins_lnc = lnC_normal(nP, Ns)
#output_sample = ins_lnc(T.as_tensor([mu, sd])).eval()
#output_sampleT = T.as_tensor(output_sample)
#input_sample = [mu, sd]
#input_sampleT = T.as_tensor(input_sample)
#o1 = ins_lnc.grad([A], [C])
#fo1 = th.function(outputs=[o1[0]], inputs=[A, C])
#fo2 = th.function(outputs=[o1[1]], inputs=[A, C])
#withgrad1 = fo1(input_sample, np.ones(nP*Ns))
#withgrad2 = fo2(input_sample, np.ones(nP*Ns))
#print('with grad():', withgrad1, withgrad2)    
#
#
#import test_sym_grads_delete as tst
#x = np.arange(Ns)
#print('symbolic:', tst.dGdMu(x, mu, sd).sum(), tst.dGdSd(x, mu, sd).sum())
#
##%% Verify grad
#
##def verify_grad(fun, pt, n_tests=2, rng=None, eps=1.0e-7, abs_tol=0.0001, rel_tol=0.0001):
## pt: list of numpy arrays as inputs
## n_tests
## random number vector... "we check the gradient of sum(u*fn) at pt "
## eps: step size for finite difference
## abs_tol: absolute tolerance for identity
## rel_tol: relative tolerance for identity    
#th.config.compute_test_value='off'
#
#rng = np.random.RandomState(42)
#ins_lnc = lnC_normal(2, 30)
##start = np.array([20,10], dtype=np.float64)
##start = (20., 10.) 
##start = [np.array(20.), np.array(10.)]
#start = [[20., 10.]]
#th.gradient.verify_grad(ins_lnc, start, n_tests=1, 
#                        rng=rng, eps=1.0e-7, abs_tol=0.0001, rel_tol=0.0001)