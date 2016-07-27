# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:50:25 2016

@author: dario
"""
import numpy as np
import scipy as sp
import utils
class MDPmodel(object):
    def __init__(self,MDP):
        # TODO: Fix these parameters properly.
        self.alpha = 8.0
        self.Beta = 4.0
        self.g = 1.0
        self.lambd = 0.0
        self.N = 4
        self.T = np.shape(MDP.V)[1]
        
        self.Ns = np.shape(MDP.B)[1] # Number of hidden states
        self.Nu = np.shape(MDP.B)[0] # Number of actions
        self.Np = np.shape(MDP.V)[0]
        self.No = np.shape(MDP.A)[0]
        
        # If the matrices A, B, C or D have any zero components, add noise and 
        # normalize again.
        p0 = np.exp(-16.0) # Smallest probability
        self.A = MDP.A + (np.min(MDP.A)==0)*p0
        self.A = sp.dot(self.A,np.diag(1/np.sum(self.A,0)))
        
        self.B = MDP.B + (np.min(MDP.B)==0)*p0
        for b in xrange(np.shape(self.B)[0]):
            self.B[b] = self.B[b]/np.sum(self.B[b],axis=0)
            
        C = sp.tile(MDP.C,(self.No,1)).T
        self.C = C + (np.min(C)==0)*p0
        self.C = self.C/np.sum(self.C)
        self.lnC = np.log(self.C)
        
        self.D = MDP.D + (np.min(MDP.D)==0)*p0
        self.D = self.D/np.sum(self.D)
        self.lnD = np.log(self.D)
        
        self.S = MDP.S
        
        self.lnA = sp.log(self.A)
        self.H = np.sum(self.A*self.lnA,0)
        
        self.V = MDP.V
    def posteriorOverStates(self, Observation, CurrentTime, Policies,
                            PosteriorLastState, PastAction, 
                            PriorPrecision, newB = None):
        """
        Decision model for Active Inference. Takes as input the model parameters 
        in MDP, as well as an observation and the prior over the current state.
        """
        V = self.V
        w = np.array(range(self.Np))
        x = PosteriorLastState
        W = PriorPrecision
        a = PastAction
        t = CurrentTime
        T = self.T
        u = np.zeros(self.Np)
        P = np.zeros(self.Nu)
        # A 'false' set of transition matrices can be fed to the Agent, 
        # depending on the newB input above. No input means that the ones from
        # the actinf class are used:
        if newB is None:
            B = self.B
        else:
            if np.shape(newB) != np.shape(self.B):
                raise Exception('BadInput: The provided transition matrices'+
                                ' do not have the correct size')
            B = newB
            
        # Check whether the policies given are for all time points T or just 
        # for the remaining ones. If for all times T, create an offset so that
        # only the last actions are used.
        if t==0:
            v = self.lnA[Observation,:] + self.lnD
        else:
            v = self.lnA[Observation,:] + sp.log(sp.dot(B[a],x))
        x = utils.softmax(v)                                             
#        print x
        cNp = np.shape(V)[0]
        Q = np.zeros(cNp)
        for k in xrange(cNp):
            xt = x
            for j in xrange(t,T):
                # transition probability from current state
                xt = sp.dot(B[V[k,j]],xt)
                ot = sp.dot(self.A,xt)
                # Predicted Divergence
                Q[k] += sp.dot(self.H,xt) + sp.dot(self.lnC[:,j] - 
                        sp.log(ot),ot)
                        
        # Variational updates: calculate the distribution over actions, then the
        # precision, and iterate N times.
        b = self.alpha/W # Recover b 'posterior' (or prior or whatever) from precision
        for i in xrange(self.N):
            # policy (u)
            u[w] = utils.softmax(sp.dot(W,Q))
            # precision (W)
            b = self.lambd*b + (1 - self.lambd)*(self.Beta - 
                sp.dot(u[w],Q))            
            W = self.alpha/b
        # Calculate the posterior over policies and over actions.
        for j in xrange(self.Nu):
            P[j] = np.sum(u[w[utils.ismember(V[:,t],j)]])
#        print '\n'
#        print x
        return x, P, W
     
    def sampleNextState(self,CurrentState, PosteriorAction):
        """
        Samples the next action and next state based on the Posteriors
        """
        s = CurrentState
        P = PosteriorAction
        NextAction = np.nonzero(np.random.rand(1) < np.cumsum(P))[0][0]
        NextState = np.nonzero(np.random.rand(1) < 
                    np.cumsum(self.B[NextAction,:,s.astype(int)]))[0][0]
        return NextAction, NextState
    def sampleNextObservation(self,CurrentState):
        """
        Samples the next observation given the current state.
        
        The state can be given as an index or as a vector of zeros with a
        single One in the current state.
        """
        if np.size(CurrentState) != 1:
            CurrentState = np.nonzero(CurrentState)[0][0]
        Observation = np.nonzero(np.random.rand(1) <
                        np.cumsum(self.A[:,CurrentState]))[0][0]
        return Observation
