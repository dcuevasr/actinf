# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:50:25 2016

@author: dario

Base class for Active Inference. It sets up the parameters necessary for
Actinf based on the input MDP. Add small non-zero probabilities to the required
matrices to avoid division by zero and extracts parameters from these matrices
for later use.

By making use of the methods, a full simulation of active inference over all
trials is performed. So see an example, see exampleFull().

Uses:
    While it is possible to use this class on its own by giving the __init__
    function an appropriate MDP object (see below for a definition of an MDP
    object), the intention is that subclasses are created from it, where this
    MDP object is created and used.

Inputs:
    Regardless of if the class is to be used alone or to generate subclasses,
    the MDP object is necessary and must contain the following fields:
        A       [nO,nS] Observation matrix
        B       [nU,nS,nS] Transition matrices, where the transitions are
                defined from the third dimension to the second one, that is,
                B[0,5,6] is the probability of transitioning from state 6 to
                state 5, having chosen the action 0.
        C       [nO] Priors over last observation (goals).
        D       [nS] Priors over initial state.
        S       [nS] Real initial state. Must be a vector of zeros, with a
                single 1 in the element representing the real initial state.
        V       [nV,nT] All possible policies.




"""
import numpy as np
import scipy as sp
import utils

np.seterr(divide = 'ignore')

class Actinf(object):
    def __init__(self,MDP):
        self.A = MDP.A
        self.B = MDP.B
        self.C = MDP.C
        self.D = MDP.D
        self.S = MDP.S
        self.V = MDP.V

        self.importMDP()
    def importMDP(self, SmallestProbability = True):
        """ Adds a smallest probability of exp(-16) to all the matrices, to
        avoid real infinitums. It also defines lnA, lnC, lnD and H.

        Setting SmallestProbability to False adds nothing to the matrices but
        still defines the logarithms and H. H is a matrix of zeros in this
        case, since lim(x->0)xlogx = 0.
        """
        # TODO: Fix these parameters properly.
#        self.alpha = 8.0
#        self.Beta = 4.0
#        self.g = 20.0
#        self.lambd = 1.0
        self.N = 4
        self.T = np.shape(self.V)[1]

        self.Ns = np.shape(self.B)[1] # Number of hidden states
        self.Nu = np.shape(self.B)[0] # Number of actions
        self.Np = np.shape(self.V)[0]
        self.No = np.shape(self.A)[0]

        # If the matrices A, B, C or D have any zero components, add noise and
        # normalize again.
        if SmallestProbability is True:
            p0 = np.exp(-16.0) # Smallest probability
        else:
            p0 = 0
        #self.A += (np.min(self.A)==0)*p0
        #self.A = sp.dot(self.A,np.diag(1/np.sum(self.A,0)))

        self.B += (np.min(self.B)==0)*p0
        for b in range(np.shape(self.B)[0]):
            self.B[b] = self.B[b]/np.sum(self.B[b],axis=0)

        self.C += (np.min(self.C)==0)*p0
        self.C = self.C/self.C.sum(axis=0)

        self.D += (np.min(self.D)==0)*p0
        self.D = self.D/np.sum(self.D)


        self.lnA = sp.log(self.A)
        self.lnC = np.log(self.C)
        self.lnD = np.log(self.D)

        if SmallestProbability is True:
            self.H = np.sum(self.A*np.nan_to_num(self.lnA),0)
        else:
            self.H = np.zeros(self.Ns)

    def posteriorOverStates(self, Observation, CurrentTime, Policies,
                            PosteriorLastState, PastAction,
                            PriorPrecision, newB = None, PreUpd = False):
        """
        Decision model for Active Inference. Takes as input the model
        parameters in MDP, as well as an observation and the prior over the
        current state.

        The input PreUpd decides whether the output should be the final
        value of precision (False) or the vector with all the updates for this
        trial (True).
        """
#        print PriorPrecision, self.gamma
        V = Policies
        cNp = np.shape(V)[0]



        w = np.array(range(cNp))
        x = PosteriorLastState
        W = PriorPrecision
        a = PastAction
        t = CurrentTime
        T = self.T
        u = np.zeros(cNp)
        P = np.zeros(self.Nu)


        # V can be given as the action sequences starting at trial t, instead
        # of the full array (starting at trial 0). This is done to avoid doing
        # extra calculations on ''repeated'' entries on V, when observations
        # are being processed as independent (as oppossed to as part of a
        # game). In this case, pad the left-hand side of the array with zeros,
        # to make it consistent with everything else. These values will not be
        # used for any calculations:
        if V.shape[1] != T:
            V = np.hstack([-np.ones((V.shape[0],T - V.shape[1]), dtype=int), V])


        # A 'false' set of transition matrices can be fed to the Agent,
        # depending on the newB input above. No input means that the ones from
        # the actinf class are used:
        if newB is None:
            B = self.B
        else:
            if np.shape(newB) != np.shape(self.B):
                raise ValueError('The provided transition matrices'+
                                ' do not have the correct size')
            B = newB

        if t==0:
            v = self.lnA[Observation,:] + self.lnD
        else:
            v = self.lnA[Observation,:] + sp.log(sp.dot(B[a],x))
        x = utils.softmax(v)

        Q = np.zeros(cNp)

        for k in range(cNp):
            xt = x
            for j in range(t,T):
                # transition probability from current state
                xt = sp.dot(B[V[k, j],:,:], xt)
#                raise Exception('stooooop')
                ot = sp.dot(self.A, xt)
                # Predicted Divergence
                Q[k] += self.H.dot(xt) + (self.lnC - np.log(ot)).dot(ot)

#        self.oQ.append(Q)
#        self.oV.append(V)
        # Variational updates: calculate the distribution over actions, then
        # the precision, and iterate N times.
        precisionUpdates = []
        b = self.alpha/W
        for i in range(self.N):
            # policy (u)
            u[w] = utils.softmax(W*Q)
            # precision (W)
            b = self.lambd*b + (1 - self.lambd)*(self.beta - sp.dot(u[w],Q))
            W = self.alpha/b
            precisionUpdates.append(W)
        # Calculate the posterior over policies and over actions.
        for j in range(self.Nu):
            P[j] = np.sum(u[w[utils.ismember(V[:,t],j)]])


        if PreUpd is True:
            return x, P, precisionUpdates
        else:
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
    def exampleFull(self, printTime = False, PreUpd = False):
        """ This is a use example for the Active Inference class. It performs
        inference for all trials in one go.

        For this example to work, the class must be already initiated with a
        specific task (e.g. bet task from Kolling 2014).

        The resulting Observations, States, inferred states, taken Actions and
        the posterior distribution over actions at each trial are saved in the
        dictionary Example.
        """

        from time import time

        t1 = time()
        # Check that the class has been fully initiated with a task:
        if hasattr(self, 'lnA') is False:
            raise Exception('NotInitiated: The class has not been initiated'+
                            ' with a task')
        T = self.V.shape[1]
        wV = self.V   # Working set of policies. This will change after choice
        obs = np.zeros(T, dtype=int)    # Observations at time t
        act = np.zeros(T, dtype=int)    # Actions at time t
        sta = np.zeros(T, dtype=int)    # Real states at time t
        bel = np.zeros((T,self.Ns))      # Inferred states at time t
        P   = np.zeros((T, self.Nu))
        W   = np.zeros(T)
        # Matrices for diagnostic purposes. If deleted, also deleted the
        # corresponding ones in posteriorOverStates:
        self.oQ = []
        self.oV = []
        self.oH = []

        sta[0] = np.nonzero(self.S)[0][0]
        # Some dummy initial values:
        PosteriorLastState = self.D
        PastAction = 1
        PriorPrecision = self.gamma
        Pupd = []
        for t in range(T-1):
            # Sample an observation from current state
            obs[t] = self.sampleNextObservation(sta[t])
            # Update beliefs over current state and posterior over actions
            # and precision
            bel[t,:], P[t,:], Gamma = self.posteriorOverStates(obs[t], t, wV,
                                                    PosteriorLastState,
                                                    PastAction,
                                                    PriorPrecision,
                                                    PreUpd = PreUpd)
            if PreUpd is True:
                W[t] = Gamma[-1]
                Pupd.append(Gamma)
            else:
                W[t] = Gamma
            # Sample an action and the next state using posteriors
            act[t], sta[t+1] = self.sampleNextState( sta[t], P[t,:])
            # Remove from pool all policies that don't have the selected action
            tempV = []
            for seq in wV:
                if seq[t] == act[t]:
                    tempV.append(seq)
            wV = np.array(tempV, dtype = int)
            # Prepare inputs for next iteration
            PosteriorLastState = bel[t]
            PastAction = act[t]
            PriorPrecision = W[t]
        xt = time() - t1
        self.Example = {'Obs':obs, 'RealStates':sta, 'InfStates':bel,
                        'Precision':W, 'PostActions':P, 'Actions':act,
                        'xt':xt}
        if PreUpd is True:
            self.Example['PrecisionUpdates'] = np.array(Pupd)
        if printTime is True:
            pass
#            print 'Example finished in %f seconds' % xt

#        print 'See the Example dictionary for the results\n'

    def full_inference(self, obs_f = None, act_f = None, V_f = None,
                       sta_f = None, preupd = False, just_return = False):
        """ Performs active inference over all trials for one game, as defined
        in the particular sub-class in use (e.g. betClass).

        The observations, choices and available policies can be forced by
        setting the appropriate inputs.

        The resulting Observations, States, inferred states, taken Actions and
        the posterior distribution over actions at each trial are saved in the
        dictionary Example or just returned.

        Parameters
        ----------
        obs/act/sta/_f: arrays, size = {nT,}
            Numpy arrays (can also be lists or any indexable array) which
            contain the forced observations/actions/states for a full game. If
            not provided, they will be calculated.
        V_f: array, size = {nT,-1}
            Action sequences to consider for active inference. If none provided,
            defaults to self.V.
        preupd: bool
            If True, the precision updates are saved and returned. If False,
            only the final value for each trial.
        just_return: bool
            If True, the dictionary Example, with all the results, is returned.
            If False, it is written to self.Example.

        Returns
        -------
        Dictionary with the following keys:
        Obs: numpy array, size = {nT}
            Observations by the agent.
        RealStates: numpy array, size = {nT,}
            Actual states visited by the agent.
        InfStates: numpy array, size = {nT,}
            States inferred by the agent.
        Precision, numpy array, size = {nT,}
            Final value of Precision at every trial.
        PostActions: numpy array, size = {nT, nA}
            Posteriors over actions for each trial
        Actions: numpy array, size = {nT,}
            Action taken by the agent at every trial.
        PrecisionUpdates: list, size = {nT*4,}
            Precision updates at every trial. The 4 comes from the fact that
            the inference over actions is iterated 4 times at each trial.
        xt: int
            Execution time for this game.

        """

        from time import time

        t1 = time()
        # Check that the class has been fully initiated with a task:
        if hasattr(self, 'lnA') is False:
            raise Exception('NotInitiated: The class has not been initiated'+
                            ' with a task')
        if V_f is not None:
            wV = V_f
        else:
            wV = self.V

        T = wV.shape[1]

        if obs_f is None:
            obs = np.zeros(T, dtype=int)    # Observations at time t
        else:
            obs = obs_f

        if act_f is None:
            act = np.zeros(T, dtype=int)    # Actions at time t
        else:
            act = act_f

        if sta_f is None:
            sta = np.zeros(T, dtype=int)    # Real states at time t
        else:
            sta = sta_f

        bel = np.zeros((T,self.Ns))      # Inferred states at time t
        P   = np.zeros((T, self.Nu))
        W   = np.zeros(T)
        # Matrices for diagnostic purposes. If deleted, also deleted the
        # corresponding ones in posteriorOverStates:
        self.oQ = []
        self.oV = []
        self.oH = []

        sta[0] = np.nonzero(self.S)[0][0]
        # Some dummy initial values:
        PosteriorLastState = self.D
        PastAction = 1
        PriorPrecision = self.gamma
        Pupd = []
        for t in range(T-1):
            # Sample an observation from current state
            if obs_f is None:
                obs[t] = self.sampleNextObservation(sta[t])
            # Update beliefs over current state and posterior over actions
            # and precision
            bel[t,:], P[t,:], Gamma = self.posteriorOverStates(obs[t], t, wV,
                                                    PosteriorLastState,
                                                    PastAction,
                                                    PriorPrecision,
                                                    PreUpd = preupd)
            if preupd is True:
                W[t] = Gamma[-1]
                Pupd.append(Gamma)
            else:
                W[t] = Gamma
            # Sample an action and the next state using posteriors
            act_tmp, sta_tmp = self.sampleNextState( sta[t], P[t,:])
            if act_f is None:
                act[t] = act_tmp
            if sta_f is None:
                sta[t+1] = sta_tmp
            # Remove from pool all policies that don't have the selected action
            tempV = []
            for seq in wV:
                if seq[t] == act[t]:
                    tempV.append(seq)
            wV = np.array(tempV, dtype = int)
            # Prepare inputs for next iteration
            PosteriorLastState = bel[t]
            PastAction = act[t]
            PriorPrecision = W[t]
        xt = time() - t1
        Example = {'Obs':obs, 'RealStates':sta, 'InfStates':bel,
                            'Precision':W, 'PostActions':P, 'Actions':act,
                            'xt':xt}
        if preupd is True:
            Example['PrecisionUpdates'] = np.array(Pupd)
        if just_return is True:
            return Example
        else:
            self.Example = Example


    def plot_action_posteriors(self, posterior_over_actions = None,
                               fignum = None, ax = None):
        """ Stacked bar plot of the posteriors over actions at each trial.

        If the posteriors are provided in the input, those are plotted.
        Otherwise, the function attempts to get them from the Example dict,
        which is the output of the exampleFull() method.
        """
        import matplotlib.pyplot as plt
        if posterior_over_actions is not None:
            postA = posterior_over_actions
        else:
            postA = self.Example['PostActions']


        nT, nU = postA.shape

        width = 0.5
        bottoms = np.zeros(nT)
        if ax is None:
            if fignum is None:
                plt.figure()
            else:
                plt.figure(fignum)
            ax = plt.gca()

        for act in range(nU):
            ccolor = (act%2)*'g' + ((act+1)%2)*'y'
            plt.bar(range(1,nT+1), postA[:,act], width, bottom = bottoms,
                    color = ccolor)
            plt.ylim([0, 1])
            plt.xlim([1,8])
            plt.xticks([])
            bottoms += postA[:,act]
        plt.show()

    def plot_real_states(self, real_states = None, actions = None,
                         which_real = None, fignum = None, ax = None):
        """ Plots the real states visited by the agent.

        When a which_real parameter is provided, only the physical states are
        plotted, which are assumed to be the first dimension of an array (in
        fortran ordering). The value of which_real, when provided, is taken
        to be the number of physical states.

        If the real states are provided (in the real_states input), those are
        plotted. Otherwise, the function attempts to plot from the Example
        dictionary, which is the output from exampleFull().
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if real_states is not None:
            S = real_states
        else:
            S = self.Example['RealStates']
            A = self.Example['Actions']

        if actions is not None:
            A = actions
        else:
            A = np.zeros(self.nT)


        nS = self.Ns
        nT = S.size
        if which_real is not None:
            rem = which_real
        else:
            rem = self.nS


        S = S%rem
        nS = rem
        Smatrix = np.zeros((nS,nT))
        Smatrix[S[:-1], range(nT-1)] = 1

        if hasattr(self, 'thres'):
            thres = self.thres
        else:
            thres = S.max()

        maxy_plot = max(S.max(), self.thres)*1.2
        if ax is None:
            if fignum is None:
                plt.figure()
            else:
                plt.figure(fignum)
            ax = plt.gca

        plt.plot(S,'+-')
        plt.plot(range(nT), np.ones(nT)*thres)
        plt.ylim([0, maxy_plot])
        plt.ylabel('Accumulated points')
        plt.xlabel('Trial')

        plt.show

    def expected_states(self, V = None, S = None):
        """ Returns all the states expected to be visited in the future given
        a set of action sequences V and an initial state S. Both S and the
        output are probability distributions over all possible states.

        The initial state S is not returned as part of the array.
        Note: nV = number of action sequences. nT = number of trials.
        nS = number of possible states.

        Inputs:
          V             {nV, nT} Set of action sequences to use. Defaults to
                        the one contained in self.
          S             {nS} Initial state distribution. Defaults to self.

        Outputs
          xS            {nV, nT, nS} Expected states in the future. Does not
                        include the initial state.
        """
        if V is None:
            V = self.V
        B = self.B
        if S is None:
            S = self.S
        nv, nt = V.shape
        # Expected visited states:
        xS = np.zeros((nv, nt+1, self.Ns))
        xS[:,0,:] = S
        for v in range(nv):
            for t in range(0,nt):
                xS[v, t+1, :] = np.dot(B[V[v,t],:,:], xS[v, t, :])

        return xS[:, 1:, :]



class MDPmodel(Actinf):
    """ For compatibility with older implementations of tasks that might still
    want to call with the old name.
    """
    def __init__(self, MDP):
        super(MDPmodel,self).__init__(MDP)





class CurrentState(object):
    """ Holds the current state for active inference.
    Both real states and inferred states can be represented with this class.
    """
    def __init__(self, nSr, nSi):
        self.nSr = nSr
        self.nSi = nSi
        self.nStot = nSr * nSi

    def separate_states(self, fullState, just_return = False):
        """ Separates the current full state into relevant and irrelevant. It
        is assumed that the first index to move is the relevant one; this means
        that states 1 to nSr are taken to be all the relevant states for the
        first value of the irrelevant states.

        To separate, it is assumed that the probability of a given state is
        given as the sum over all irrelevant states:
        p(s^{rel}_x) = \displaystyle \sum _{n=1}^{N} p(s^{rel}_x, s^{irr}_n)

        Equivalently for irrelevant states.
        """
        S = np.reshape(fullState,(self.nSr, self.nSi), order='F')
        Sr = S.sum(axis=1)
        Si = S.sum(axis=0)
        if just_return is False:
            self.Sr = Sr
            self.Si = Si
        else:
            return Sr, Si

    def join_states(self, relStates, irrStates, just_return = False):
        """ Generates the full state from the current separated states. To
        calculate the probability of any (Srel, Sirr) combined state, the
        probabilities of the separated states are multiplied.

        This implementation assumes that p(rel,irr) = p(rel)p(irr), which might
        not be valid for a given generative model.

        To calculate the joint state without setting it to self.S, use the
        optional input just_return=True
        """
        # TODO: Add the possibility of calculating the joint states'
        #       probabilities with another function.
        S = np.matrix(relStates).T*np.matrix(irrStates)
        S = np.reshape(S, (self.nStot,1), order='F')
        if just_return is False:
            self.S = S
        else:
            return S

    def separate_actions(self, B, just_return = False):
        """ Separates the transition matrices B into relevant and irrelevant.
        """
        nU = B.shape[0]

        Brel = np.zeros((nU, self.nSr, self.nSr))
        for u in range(nU):
            for rel in range(self.nSr):
                rel_indices = utils.allothers([range(self.nSr),[0]],
                                              (self.nSr,self.nSr))
                Brel[u,:,rel] = B[u, rel_indices, rel]
        if just_return is False:
            self.Brel = Brel
        else:
            return Brel


