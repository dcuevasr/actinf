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
        self.importMDP(MDP)
    def importMDP(self):
        # TODO: Fix these parameters properly.
        self.alpha = 8.0
        self.Beta = 4.0
        self.g = 1.0
        self.lambd = 0.0
        self.N = 4
        self.T = np.shape(self.V)[1]
        
        self.Ns = np.shape(self.B)[1] # Number of hidden states
        self.Nu = np.shape(self.B)[0] # Number of actions
        self.Np = np.shape(self.V)[0]
        self.No = np.shape(self.A)[0]
        
        # If the matrices A, B, C or D have any zero components, add noise and 
        # normalize again.
        p0 = np.exp(-16.0) # Smallest probability
        self.A += (np.min(self.A)==0)*p0
        self.A = sp.dot(self.A,np.diag(1/np.sum(self.A,0)))
        
        self.B += (np.min(self.B)==0)*p0
        for b in xrange(np.shape(self.B)[0]):
            self.B[b] = self.B[b]/np.sum(self.B[b],axis=0)
            
        self.C = sp.tile(self.C,(self.No,1)).T
        self.C += (np.min(self.C)==0)*p0
        self.C = self.C/np.sum(self.C)
        self.lnC = np.log(self.C)
        
        self.D += (np.min(self.D)==0)*p0
        self.D = self.D/np.sum(self.D)
        self.lnD = np.log(self.D)
        
        self.S = self.S
        
        self.lnA = sp.log(self.A)
        self.H = np.sum(self.A*self.lnA,0)
        
        self.V = self.V
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
    def exampleFull(self):
        """ This is a use example for the Active Inference class. It performs
        inference for all trials in one go.
        
        For this example to work, the class must be already initiated with a
        specific task (e.g. bet task from Kolling 2014).
        
        The resulting Observations, States, inferred states, taken Actions and
        the posterior distribution over actions at each trial are saved in the
        dictionary Example.
        """
        # Check that the class has been fully initiated with a task:
        if hasattr(self, 'lnA') is False:
            raise Exception('NotInitiated: The class has not been initiated'+
                            ' with a task')
        T = self.V.shape[1]

        obs = np.zeros(T, dtype=int)    # Observations at time t
        act = np.zeros(T, dtype=int)    # Actions at time t
        sta = np.zeros(T, dtype=int)    # Real states at time t
        bel = np.zeros((T,self.Ns))      # Inferred states at time t
        P   = np.zeros((T, self.Nu))
        W   = np.zeros(T)
        sta[0] = np.nonzero(self.S)[0][0]
        # Some dummy initial values:
        PosteriorLastState = 1
        PastAction = 1
        PriorPrecision = 1
        for t in xrange(T-1):
            # Sample an observation from current state
            obs[t] = self.sampleNextObservation(sta[t])
            # Update beliefs over current state and posterior over actions 
            # and precision
            bel[t,:], P[t,:], W[t] = self.posteriorOverStates(obs[t], t, self.V,
                                                    PosteriorLastState, 
                                                    PastAction,
                                                    PriorPrecision)
            # Sample an action and the next state using posteriors
            act[t], sta[t+1] = self.sampleNextState( sta[t], P[t,:])
            # Prepare inputs for next iteration
            PosteriorLastState = bel[t]
            PastAction = act[t]
            PriorPrecision = W[t]
        self.Example = {'Obs':obs, 'RealStates':sta, 'InfStates':bel, 
                        'Precision':W, 'PostActions':P, 'Actions':act}
        print 'See the Example dictionary for the results\n'