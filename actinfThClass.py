#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:56:06 2016

@author: dario
"""

import theano as th
from theano import tensor as T
import posterior_states_theano as postA
import numpy as np
from pymc3.distributions import Discrete

#from utils import softmax

#%% actinf Elementwise
class actinfOpElewise(th.gof.Op):
    """Wrapper to make the actinf_post_actions respond to multiple trials and
    initial states given as input.
    """

    __props__ = ()
#    itypes = [T.dvector]
#    otypes = [T.dvector]
    check_input = True
    def __init__(self, lnc, lambd, alpha, beta, v, s, h, b, ct, *args, **kwargs):
        super(actinfOpElewise, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.v = v
#        self.v0 = (v[:,ct]==0).nonzero()
#        self.v1 = (v[:,ct]==1).nonzero()
        self.s = s
        self.h = h
        self.b = b
        self.ct = ct
        self._check_inputs()

#        self._create_all_functions()
    def make_node(self, x):
        x = T.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return th.Apply(self, [x], [x.type()])

#    def _create_all_functions(self):
#        self.funs = [self._create_function(n) for n in range(self.nCt)]

    def _check_inputs(self):
        if not isinstance(self.ct, np.ndarray):
            self.ct = np.array((self.ct,))
        if not isinstance(self.s, np.ndarray):
            self.s = np.array(self.s)
        if self.ct.shape[0]>1 and not self.ct.shape[0] == self.s.shape[0]:
            raise ValueError('Dimensions of ct (0) and s (0) do not match')
        self.nCt = self.ct.shape[0]

#    def _create_function(self, n):
#        return actinfOpElewise(self.lnc, self.lambd, self.alpha, self.beta,
#                                   self.v, self.s[n], self.h, self.b,
#                                   self.ct[n])

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

    def _perform_single(self, t, inputs):
#        outputs[0][0] = np.hstack([self.funs[n](T.as_tensor(inputs[0])) for n in range(self.nCt)])
        """ Calculates the posterior over actions for active inference.
        """
        ct = self.ct[t]
        s = self.s[t]

        lnc = inputs[0][1:]
        gammi = inputs[0][0]
        v = self.v[:, ct:]
        v0 = v[v[:,0]==0,:]
        v1 = v[v[:,0]==1,:]
        mafe = postA.get_fE(v, s, self.h, lnc, self.b)
        mafe0 = mafe[v0].sum()
        mafe1 = mafe[v1].sum()
        cnp = self.v.shape[0]
        maga = postA.get_gamma(gammi, mafe, self.lambd,
                               self.alpha, self.beta, cnp)
        max_mafe = max([mafe0, mafe1])
        exp_out0 = np.exp(maga[-1]*(mafe0 - max_mafe))
        exp_out1 = np.exp(maga[-1]*(mafe1 - max_mafe))
        norm_const = exp_out0 + exp_out1
        return np.array([exp_out0, exp_out1])/norm_const

#        outputs[0][0] = np.ones((2*self.nCt))

    def perform(self, node, inputs, outputs):
#        CT = T.vector('CT', dtype='int64')
#        S = T.matrix('S', dtype='float64')
#        CT.tag.test_value = self.ct
#        S.tag.test_value = self.s
#        out, _ = th.scan(fn=self._perform_single,
#                         sequences=[CT, S],
#                         non_sequences = inputs)
#        out_fun = th.function(inputs=[self.ct, self.s, inputs], outputs = out)
#        outputs[0][0] = T.reshape(out_fun(), ndim=1, name='afElperform')

        out = -np.ones((self.nCt, 2))
        for t in range(self.nCt):
            out[t,:] = self._perform_single(t, inputs)

        outputs[0][0] = np.reshape(out, newshape=-1)



    def _grad_single(self, ct, s, lnC2, GAMMI2):
        lnC = lnC2
        GAMMI = GAMMI2
        v = self.v#T.as_tensor(self.v)[:,ct:]
        v0 = T.as_tensor(v[v[:,0]==0, :])
        v1 = T.as_tensor(v[v[:,0]==1, :])

        cnp = v.shape[0]

        # Gradient of fE wrt the priors over final state
        [ofE, oxS], upd_fE_single = th.scan(fn=self._free_energy,
                                   sequences=v,
                                   non_sequences=[s,self.h,lnC,self.b])
        ofE0 = ofE[v0].sum()
        ofE1 = ofE[v1].sum()

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
        self.debug = [Jac1_lnC, Jac2_lnC, Jac2_gammi, Jac1_gammi, dFE0dlnC,
                      dFE1dlnC, dGdg, out1, out2, v0, v1, v, ct]
        return Jac1, Jac2

    def grad(self, inputs, dCdf):
        CT = T.as_tensor(self.ct)
        S = T.as_tensor(self.s)
        (jac1, jac2), _ = th.scan(fn=self._grad_single,
                                  sequences=[CT, S],
                                  non_sequences=[inputs[0][1:], inputs[0][0]])

#        for t in self.ct:
#            out = self._grad_single(t, s)

#        Jac1 = T.reshape(jac1, newshape=(1,-1))
#        Jac2 = T.reshape(jac2, newshape=(1,-1))
        Jac = T.concatenate([jac1, jac2], axis=0)
#        return Jac1*dCdf[0][0] + Jac2*dCdf[0][1],
        return Jac.T.dot(dCdf[0]),

class actinfDist(Discrete):
    """ pymc3 distribution for the posteriors over actions in active inference.
    It takes as inputs the usual MDP stuff from betClass or actinfClass.
    """
    def __init__(self, gammi, lnc, lambd, alpha, beta, v, s, h,b,ct,
                 *args, **kwargs):
        super(actinfDist, self).__init__(*args, **kwargs)
        self.lnc = lnc
        self.gammi = gammi
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.ct = ct
        self.s = s
        self._check_inputs()
        self.v = T.as_tensor(v)
        self.s = T.as_tensor(self.s)
        self.h = T.as_tensor(h)
        self.b = T.as_tensor(b)
        self.conds = np.arange(self.nCt, dtype=np.int8)

        self.mafu = actinfOpElewise(lnc, lambd, alpha, beta, v, s, h, b, ct)

    def _check_inputs(self):
        if not isinstance(self.ct, np.ndarray):
            self.ct = np.array((self.ct,))
#        if not isinstance(self.s, np.ndarray):
#            self.s = np.array(self.s)
        if self.ct.shape[0]>1 and not self.ct.shape[0] == self.s.shape[0]:
            raise ValueError('First dimension of ct and s do not match')
        self.nCt = self.ct.shape[0]

    def _mafu(self,n):
        return actinfOpElewise(self.lnc, self.lambd, self.alpha,
                   self.beta, self.v, T.as_tensor(self.s)[n], self.h,
                   self.b, T.as_tensor(self.ct)[n])(T.concatenate([T.stack(self.gammi),
                   self.lnc]))
    def logp(self, action):
#        return T.log(self.mafu(T.concatenate([T.stack(self.gammi), self.lnc]))[action])
        flat_out = self.mafu(T.concatenate([T.stack(self.gammi), self.lnc]))
        stack_out = T.reshape(flat_out, newshape=(self.nCt, -1))
        return stack_out[:, action]

class actinfDistMany(actinfDist):
    def logp(self, action):
        return T.log(self.mafu(T.concatenate([T.stack(self.gammi), self.lnc]))[action])
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
        return T.as_tensor([dCdf[0].dot(dYdMIn[:,0]), dCdf[0].dot(dYdMIn[:,1])]),

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


#%%
#class actinfOp(th.gof.Op):
#    """ Theano op to calculate the posteriors over actions in active inference,
#    as well as its gradient wrt the inputs.
#    """
#    __props__ = ()
#    itypes = [T.dvector]
#    otypes = [T.dvector]
#    check_input = True
#
#    def __init__(self, lnc, lambd, alpha, beta, v, s, h, b, ct, *args, **kwargs):
#        super(actinfOp, self).__init__(*args, **kwargs)
#        self.lnc = lnc
#        self.lambd = lambd
#        self.alpha = alpha
#        self.beta = beta
#        self.v = v[:, ct:]
#        self.v0 = (v[:,0]==0) # starting from 0 because previous ones were deleted
#        self.v1 = (v[:,0]==1)
#        self.s = s
#        self.h = h
#        self.b = b
#        self.ct = ct
#
#    def perform(self, node, inputs, outputs):
#        """ Calculates the posterior over actions for active inference.
#        """
#        lnc = inputs[0][1:]
#        gammi = inputs[0][0]
#
#        mafe = postA.get_fE(self.v, self.s, self.h, lnc, self.b)
#        mafe0 = mafe[self.v0].sum()
#        mafe1 = mafe[self.v1].sum()
#        cnp = self.v.shape[0]
#        maga = postA.get_gamma(gammi, mafe, self.lambd,
#                               self.alpha, self.beta, cnp)
#        max_mafe = max([mafe0, mafe1])
#        exp_out0 = np.exp(maga[-1]*(mafe0 - max_mafe))
#        exp_out1 = np.exp(maga[-1]*(mafe1 - max_mafe))
#        norm_const = exp_out0 + exp_out1
#        outputs[0][0] = np.array([exp_out0, exp_out1])/norm_const
#
#    def grad(self, inputs, dCdf):
#        lnC = inputs[0][1:]
#        GAMMI = inputs[0][0]
#        cnp = self.v.shape[0]
#        lnC.tag.test_value = self.lnc
#        GAMMI.tag.test_value = 5
#
#        # Gradient of fE wrt the priors over final state
#        [ofE, oxS], upd_fE_single = th.scan(fn=self._free_energy,
#                                   sequences=self.v,
#                                   non_sequences=[self.s,self.h,lnC,self.b])
#        ofE0 = ofE[self.v0].sum()
#        ofE1 = ofE[self.v1].sum()
#
#        dFE0dlnC = T.jacobian(ofE0, lnC)
#        dFE1dlnC = T.jacobian(ofE1, lnC)
#        dFEdlnC  = T.jacobian(ofE,  lnC)
#
#        ofE_ = T.vector()
#        ofE_.tag.test_value = ofE.tag.test_value
#        # Gradient of Gamma with respect to its initial condition:
#        GAMMA, upd_GAMMA = th.scan(fn=self._upd_gamma,
#               outputs_info=[GAMMI],
#               non_sequences=[ofE, self.lambd, self.alpha, self.beta, cnp],
#               n_steps=4)
#        dGdg = T.grad(GAMMA[-1], GAMMI)
#
#        dGdfE = T.jacobian(GAMMA[-1], ofE)
#        dGdlnC = dGdfE.dot(dFEdlnC)
#
#
#        out1 = ofE0
#        out2 = ofE1
#        maxout = T.max([out1, out2])
#
#        exp_out1 = T.exp(GAMMA[-1]*(out1 - maxout))
#        exp_out2 = T.exp(GAMMA[-1]*(out2 - maxout))
#        norm_const = exp_out1 + exp_out2
#
#        # Derivative wrt the second output (gammi):
#        Jac1_gammi = (-(out1-out2)*dGdg*
#                T.exp(GAMMA[-1]*(out1+out2 - 2*maxout))/(norm_const**2))
#        Jac2_gammi = -Jac1_gammi
##        dfd1_tZ = Jac1_gammi*dCdf[1][0]+ Jac2_gammi*dCdf[1][1]
#
#        # Derivative wrt first input (lnc)
#        Jac1_lnC = (T.exp(GAMMA[-1]*(out1 + out2 - 2*maxout))/(norm_const**2)*
#                  (-dGdlnC*(out1 - out2) - GAMMA[-1]*(dFE0dlnC - dFE1dlnC)))
#        Jac2_lnC = -Jac1_lnC
#
#        Jac1 = T.concatenate([T.stack(Jac1_gammi), Jac1_lnC])
#        Jac2 = T.concatenate([T.stack(Jac2_gammi), Jac2_lnC])
#
#        return Jac1*dCdf[0][0] + Jac2*dCdf[0][1],
#
#    def _free_energy(self, Pi, S, H, lnC, B):
#        def one_computation(action, S, H, lnC, B):
#            SS = B[action,:,:].dot(S)
#            return SS, T.dot(SS, H) + T.dot(SS,lnC - T.log(SS))
#
#        [xS, fE], upds = th.scan(fn=one_computation,
#                             sequences=Pi,
#                             outputs_info=[S, None],
#                             non_sequences=[H, lnC, B])
#
#        return fE.sum(), xS
#
#    def _upd_gamma(self, GAMMA, Q, LAMBD, ALPHA, BETA, CNP):
#        """ Calculates a single precision update.
#        """
#        QG = GAMMA.dot(Q)
#        QGmax = QG.max()
#        u = (QG - QGmax).exp()
#        u = u/u.sum()
#        new_gamma = ALPHA/(LAMBD*GAMMA + (1 - LAMBD)*(BETA - u.dot(Q)/CNP))
#        # TODO: Apply convergence override
##        epsilon = (new_gamma - GAMMA).__abs__()/GAMMA
#        return new_gamma#, th.scan_module.until(epsilon<0.01)