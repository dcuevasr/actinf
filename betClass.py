# -*- coding: utf-8 -*-
"""
Subclass of actinfClass

Sets up the necessary matrices to simulate the Kolling_2014 experiments. There
are two variants: setMDP and setMDPMultVar. The difference between them is in
whether the action pairs (probabilities and rewards of the risky and safe
options) are fixed or are allowed to change.

Note that as a subclass of actinfClass, the exampleFull() method is available
and works with this particular paradigm with no further modification.

Uses:
    The initiation of the class accepts four optional inputs:
        paradigm        {'kolling','kolling compact','small'} controlls which
                        type of paradigm is used. For 'kolling', the exact
                        numbers are used as in kolling_2014. For 'kolling
                        compact' similar numbers are used, but total points
                        and possible rewards are all divided by 10. This
                        GREATLY reduces number of states and improves
                        performance, while maintaining the same scheme. For
                        'small', only 12 states are used; this is mostly for
                        testing, as it works very quickly.
        changePairs     {bool} Controls whether action pairs change from trial
                        to trial (and thus, are many of them) or whether there
                        is just one action pair and is fixed throughout the
                        experiment. 'True' for the former.
        actionPairs     dict{risklow, riskhigh, rewlow,rewhigh} dictionary with
                        the action pair to be used; use only with
                        changePairs = False. If not provided, defaults of
                        {0.9,0.3,1,3} will be used.

"""
from __future__ import print_function # Probably unnecessary.

import numpy as np
import itertools
import utils
import actinfClass as afc
class betMDP(afc.Actinf):
    def __init__(self, paradigm = 'kolling compact', changePairs = True,
                 actionPairs = None, nS = None, nT = None, thres = None):
        """
        Initializes the instance with parameter values depending on the inputs.
        See the class's help for details on optional inputs.

        Additionally, calls setMDPMultVar() or setMDP, depending on inputs, to
        set all the MDP matrices for Active Inference.
        """
        if paradigm == 'kolling':
            self.paradigm = 'kolling'
            self.pdivide = 1
            if nS is not None:
                self.nS = nS
            else:
                self.nS = 2000 #max number of attainable points
            if thres is not None:
                self.thres = thres
            else:
                self.thres = 1000
        elif paradigm == 'kolling compact':
            self.paradigm = 'kolling'
            self.pdivide = 10
            if nS is not None:
                self.nS = nS
            else:
                self.nS = 200 #max number of attainable points
            if thres is not None:
                self.thres = thres
            else:
                self.thres = 60
        else:
            if nS is not None:
                self.nS = nS
            else:
                self.nS = 20 #max number of attainable points
            if thres is not None:
                self.thres = thres
            else:
                self.thres = 5
            self.paradigm = 'small'


        if nT is not None:
            self.nT = nT
        else:
            self.nT = 8

        self.nU = 2

        self.obsnoise = 0.001 #0.0001

        if changePairs is True:
            self.setMDPMultVar()
        else:
            self.setMDP(actionPairs)


    def setMDP(self,parameters = None):
        """ Sets the observation and transition matrices, as well as the goals,
        the priors over initial states, the real initial state and the possible
        policies to evaluate.

        A single action pair is used for all trials. The probabilities and
        reward magnitudes of this pair is set by the actionPairs input to
        __init__. It can also be taken from the input 'parameters', when
        calling the method manually.

        If the action pair is to be specified, the input 'parameter' must be
        a dict with entries risklow, riskhigh, rewlow and rewhigh.
        """
        # checking inputs and assigning defaults if necessary
        if parameters != None:
            if not isinstance(parameters,dict):
                raise Exception('BadInput: ''parameters'' is not a dict')
            expectedParameters = ['risklow','riskhigh','rewlow','rewhigh']
            compareSets = set(parameters) & set(expectedParameters)
            if compareSets != set(expectedParameters):
                raise Exception('BadInput: ''Paremeters'' does not contain the \
                required parameters')
            else:
                risklow = parameters['risklow']
                riskhigh = parameters['riskhigh']
                rewlow = parameters['rewlow']
                rewhigh = parameters['rewhigh']
        else:
            risklow = 0.9
            riskhigh = 0.3
            rewlow = 1
            rewhigh = 3


        # Define the observation matrix
        A = np.eye(self.nS) + self.obsnoise
        A = A/sum(A)

        # Define transition matrices
        B = np.zeros((self.nU,self.nS,self.nS))
        B[0] = np.diag(risklow*np.ones(self.nS-rewlow),-rewlow)
        np.fill_diagonal(B[0], 1-risklow)
        B[0][-1,(-1-rewlow):-1] = risklow
        B[0][-1,-1] = 1

        B[1] = np.diag(riskhigh*np.ones(self.nS-rewhigh),-rewhigh)
        np.fill_diagonal(B[1], 1-riskhigh)
        B[1][-1,(-1-rewhigh):-1] = riskhigh
        B[1][-1,-1] = 1

        # Define priors over last state (goal)
        C = np.zeros(self.nS)
        C[self.thres:] = 1
        C = C/sum(C)

        # Priors over initial state
        D = np.zeros(self.nS)
        D[0] = 1

        # Real initial state
        S = D.astype(int)

        # Policies: all combinations of 2 actions
        V = np.array(list(itertools.product(range(0,self.nU),repeat = self.nT)))

        # Preparing inputs for MDP
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.S = S
        self.V = V.astype(int)
        self.alpha = 64
        self.beta = 4
        self.lambd = 0.005 # or 1/4?
        self.gamma = 20

        self.importMDP()

#        self.H = np.zeros(self.H.shape)

    def setMDPMultVar(self,parameters=None):
        """Sets the observation and transition matrices, as well as the goals,
        the priors over initial states, the real initial state and the possible
        policies to evaluate.

        A set of action pairs is used here, and are to be selected randomly
        for each trial during simulations. To do this, a third action is
        created, B[2], which is called the evolution action. When chosen, the
        action pair changes to a random one (the distribution is not
        necessarily constant).

        If the action pairs are not supplied, they are taken from the method
        setActionPairs(). Otherwise, they must be provided in a dict with
        keys pL, pH, rL, rH, which must be numpy arrays of the same length.
        """
        # Working parameters. To be imported later
        nU = self.nU
        nS = self.nS
        nT = self.nT
#        thres = self.thres
        obsnoise = self.obsnoise
        if parameters != None:
            if not isinstance(parameters,dict):
                raise ValueError('Input ''parameters'' is not a dict')
            expectedParameters = ['pL','pH','rL','rH']
            compareSets = set(parameters) & set(expectedParameters)
            if compareSets != set(expectedParameters):
                raise ValueError(' ''paremeters'' does not contain the \
                required parameters')
            else:
                self.pL = parameters['pL']
                self.pH = parameters['pH']
                self.rL = parameters['rL']
                self.rH = parameters['rH']
                self.nP = nP = len(self.pL)
        else:
            # Default action pairs
            self.setActionPairs()
            nP = self.nP

        # Define observation matrix
        A = np.eye(nS*nP) + obsnoise/nP
        # Transition matrices
        # num2coords = np.unravel_index
        # coords2num = np.ravel_multi_index
        self.set_transition_matrices()

        # Define priors over last state (goal)
        self.set_prior_goals()
        # Priors over initial state
        D = np.zeros(nS*nP)
        D[0] = 1

        # Real initial state
        S = D.astype(int)

        # Policies: all combinations of 2 actions
        V = np.array(list(itertools.product(range(0,nU),repeat = nT)))
        np.random.shuffle(V)


        # Preparing inputs for MDP
        self.A = A
#        self.B = B
#        self.C = C
        self.D = D
        self.S = S
        self.V = V.astype(int)
        self.alpha = 64
        self.beta = 4
        self.lambd = 0.005 # or 1/4?
        self.gamma = 20

        self.importMDP()

#        self.H = np.zeros(self.H.shape)


    def priors_over_goals(self, threshold = None, just_return = False):
        """
        xxx DEPRECATED in favor of set_prior_goals() xxx

        Not deleted just in case...
        TODO: Delete if you are brave
        """
        import utils
        import numpy as np
#        print 'This function has been deprecated in favor of set_prior_goals.\n'

        if threshold is None:
            thres = self.thres
        else:
            thres = threshold

        C = np.zeros(self.nS*self.nP)
        allothers = np.array(utils.allothers([range(thres,self.nS),
                                              range(self.nP)],
                                              (self.nS,self.nP)))
        C[allothers] = 1
        C = C/sum(C)

        if just_return is False:
            self.C = C
            self.thres = thres
        elif just_return is True:
            return C

    def setActionPairs(self):
        """ Sets the action pairs to be used for this instance. Which ones are
        selected depends on the instance's parameters.

        There is probably no use of this from the outside. Might be good to
        make it internal.
        """
        if self.paradigm == 'small':
            nP = 3;
            pL = np.array([0.9, 0.6, 0.8])
            pH = np.array([0.3, 0.3, 0.2])
            rL = np.array([1, 2, 1])
            rH = np.array([3, 4, 4])
        elif self.paradigm == 'kolling':
            nP = 8
            pL = np.array([90, 60, 75, 55, 90, 60, 75, 80], dtype=float)/100
            pH = np.array([35, 35, 35, 20, 45, 45 ,40, 30], dtype=float)/100
            rL = np.array([100, 180, 145, 145, 115, 150, 170, 120],
                          dtype = int)/self.pdivide
            rH = np.array([265 ,260 ,245 ,350, 240, 190, 245, 210],
                          dtype = int)/self.pdivide

        self.nP = nP
        self.pL = pL
        self.pH = pH
        self.rL = rL
        self.rH = rH

    def set_transition_matrices(self, priorActPairs = None,
                                reward = None, probability = None,
                                just_return = False):
        """ Defines the transition matrices (actions) for the task, using
        the input priorActPairs as priors over the probability of transitioning
        to a new action pair.

        The input priorActPairs must be a vector of size nP, normalized, whose
        elements represent the biases towards each of the action pairs.
        """
        if reward is None:
            nU, nS, nP = self.nU, self.nS, self.nP
            pL, pH, rL, rH = self.pL, self.pH, self.rL, self.rH
        else:
            nU = 2
            nS = self.nS
            nP = 1
            pL = pH = [probability]
            rL = rH = [reward]

        if priorActPairs is None:
            pAP = np.ones(nP)/nP
        else:
            pAP = priorActPairs

        B = np.zeros((nU,nS*nP,nS*nP))
        for s in range(nS):
            for p in range(nP):
                nextsL = np.min((s+rL[p],nS-1)).astype(int)
                nextsH = np.min((s+rH[p],nS-1)).astype(int)
                mixL = utils.allothers([[nextsL],range(nP)],(nS,nP))
                mixH = utils.allothers([[nextsH],range(nP)],(nS,nP))
                this = utils.allothers([[s]     ,range(nP)],(nS,nP))
                ifrom = [np.ravel_multi_index([s,p],(nS,nP),order='F')]
                B[np.ix_([0], mixL, ifrom)] = (pL[p]*pAP).reshape(1,nP,1)
                B[np.ix_([0], this, ifrom)] = ((1 - pL[p])*pAP).reshape(1,nP,1)
                B[np.ix_([1], mixH, ifrom)] = (pH[p]*pAP).reshape(1,nP,1)
                B[np.ix_([1], this, ifrom)] = ((1 - pH[p])*pAP).reshape(1,nP,1)
        if just_return is False:
            self.B = B
        else:
            return B

    def set_single_trans_mat(self, reward, probability):
        """ Creates a single transition matrix for the current task, given the
        reward and probability given as input.

        If the inputs are vectors (same length), a number of transition
        matrices is created that equals the number of elements in the vectors.

        Wrapper for set_transition_matrices
        """

        nB = len(reward)
        nS = self.nS

        B = np.zeros((nB, nS, nS))
        for b in range(nB):
            twoBs = self.set_transition_matrices(reward = reward[b],
                                            probability = probability[b],
                                            just_return = True)
            B[b] = twoBs[0]


        return B


    def set_prior_goals(self, selectShape='flat', rampX1 = None,
                  Gmean = None, Gscale = None, Scenter = None, Sslope = None,
                  convolute = True, cutoff = True, just_return = False):
        """ Wrapper for the functions prior_goals_flat/Ramp/Unimodal.

        Sets priors over last state (goals) in different shapes for testing
        the effects on the agent's behavior.

        The three options are 'flat', which is what is normally set in cbet.py,
        'ramp', which is a ramping-up that starts at threshold, and 'unimodal',
        which uses a Gaussian to set up a 'hump' after threshold.

        Uses:
            goals = setPriorGoals(mdp [,selectShape] [, rampX1] [, Gmean]
                                  [, Gscale] [,convolute] [,just_return])

        Inputs:
        selectShape         {'flat','ramp','unimodal','sigmoid'} selects which
                            shape is to be used. When selecting 'ramp', the
                            optional input rampX1 can be selected (default 1).
                            When using 'unimodal', Gmean and Gscale can be set to
                            change the mean (in Trial number) and scale of the
                            Gaussian. Selecting a Gmean pre-threshold
                            and a value for cutoff of False cause the 'hump' to
                            be invisible and the priors will be an exponential
                            ramp down. 'sigmoid' requires the two parameters
                            for center and slope.
        rampX1              {x} determines the initial point for the ramp,
                            which uniquely determines the slope of the ramp.
        Gmean, Gscale       {x}{y} determine the mean and the scale of the
                            Gaussian for the unimodal version.
        Scenter, Sslope     Center and slope for using when 'sigmoid' is used.
        convolute           {bool} If True, C will be calculated for the full
                            state space nS*nU, where nU is the number of action
                            pairs. If False, C will be in the nS space.
        cutoff              {bool} If True, the priors over last state (C) will
                            be set to zero pre-threshold and re-normalized.
        just_return         {bool} If True, the calculated value for C will be
                            returned and no further action taken. If False,
                            both C and lnC will be written into self.
        Outputs:
        (Note: when just_return is False, nothing is returned)
        goals               [nS] are the resulting priors over last state.
        """
        if selectShape == 'flat':
            goals = self.prior_goals_flat(convolute, just_return = True)
        elif selectShape == 'ramp':
            if rampX1 is None:
                raise ValueError('A value for rampX1 must be provided when using'+
                                ' ''ramp''')
            goals = self.prior_goals_ramp(rampX1 = rampX1,
                                   convolute = convolute, just_return = True)
        elif selectShape == 'unimodal':
            if Gscale is None or Gmean is None:
                raise ValueError('Values for Gmean and Gscale must be provided '+
                                 'when using ''unimodal''')
            goals = self.prior_goals_unimodal(Gmean, Gscale,
                                    convolute = convolute,
                                    cutoff = cutoff, just_return = True)
        elif selectShape == 'sigmoid':
            if Scenter is None or Sslope is None:
                raise ValueError('Values for Scenter and Sslope must be '+
                                 'provided when using ''sigmoid''')
            goals = self.prior_goals_sigmoid(Scenter, Sslope,
                                     convolute = convolute, just_return = True)
        else:
            raise ValueError('selectShape can only be ''flat'', ''ramp'' or '+
                            '''unimodal''')
        if just_return is True:
            return goals
        elif just_return is False and convolute is True:
            self.C = goals
            self.C += (np.min(self.C)==0)*np.exp(-16)
            self.C = self.C/self.C.sum(axis=0)
            self.lnC = np.log(self.C)
        else:
            raise ValueError('Bad combination of just_return and convolute')

    def prior_goals_sigmoid(self, Scenter, Sslope, convolute = True,
                            just_return = True):
        """ To be called from set_prior_goals()."""
        sigmoid = lambda C,S,X: 1/(1 + np.exp(-S*(X - C)))
        points = np.arange(self.nS)
        goals = sigmoid(Scenter, Sslope, points)
        if convolute:
            goals = np.tile(goals, self.nP)
            goals /= goals.sum()

        if just_return:
            return goals
        else:
            self.C = goals
            self.lnC = np.log(goals)

    def prior_goals_flat(self, convolute = True, just_return = True):
        """To be called from set_prior_goals()."""
        from utils import allothers

        if convolute is True:
            goals = np.zeros(self.nS*self.nP, dtype = float)

            indices = np.array(allothers([range(self.thres,self.nS),
                                      range(self.nP)], (self.nS,self.nP)),
                                     dtype = int)
            goals[indices] = 1.0/indices.size
        elif convolute is False:
            goals = np.zeros(self.nS, dtype = float)
            goals[self.thres:] = 1.0/goals[self.thres:].size

        if just_return is True:
            return goals
        if just_return is False and convolute is True:
            self.C = goals
            self.lnC = np.log(goals)
        else:
            raise ValueError('Bad combination of just_return and convolute')


    def prior_goals_ramp(self, rampX1, convolute = True, just_return = True):
        """ Creates goals as an increasing or decreasing ramp, depending on the
        value given for rampX1.

        rampX1 is the initial value. That is, the value of the first point
        after threshold. If rampX1 is smaller than M (where M is the number of
        points past-threshold), then the ramp is increasing. The slope is
        calculated automatically (since rampX1 determines it uniquely).

        To be called from set_prior_goals().
        """
        from utils import allothers


        thres = self.thres
        nS = self.nS
        pastThres = nS - thres
        nP = self.nP

        minX1 = 0
        maxX1 = 2.0/pastThres

        if rampX1<minX1 or rampX1>maxX1:
            raise ValueError ('Initial point X1 is outside of allowable '+
                              'limits for this task. min = %f, max = %f'
                              % (minX1, maxX1))
        if rampX1 == 1.0/pastThres:
            raise ValueError('rampX1 is invalid. For this value, use ''flat'''+
                                ' instead')

        slope = (2.0/pastThres - 2.0*rampX1)/(pastThres-1)

        stateRamp = rampX1 + slope*np.arange(pastThres)

        istateR = np.arange(self.thres, self.nS)

        if convolute is False:
            goals = np.zeros(nS)
            goals[thres:] = stateRamp
        else:
            goals = np.zeros(nS*nP)
            for ix,vx in enumerate(istateR):
                indices = np.array(allothers([[vx],range(self.nP)],
                                              (self.nS,self.nP)))
                goals[indices] = stateRamp[ix]
            goals = goals/goals.sum()
        if just_return is True:
            return goals
        elif just_return is False and convolute is True:
            self.C = goals
            self.lnC = np.log(goals)
        else:
            raise ValueError('Bad combination of just_return and convolute')



    def prior_goals_unimodal(self, Gmean, Gscale,
                           convolute = True, cutoff = True, just_return = True):
        """ Sets the priors over last state (goals) to a Gaussian distribution,
        defined by Gmean and Gscale. To be called from set_prior_goals().
        """
#        from utils import allothers
        from scipy.stats import norm

        points = np.arange(self.nS)
        npoints = norm.pdf(points, Gmean, Gscale)
        if cutoff is True:
            npoints[:self.thres] = 0
        if convolute is False:
            goals = npoints
        else:
#            goals = np.zeros(self.Ns)
#            istateR = np.arange(self.thres, self.nS, dtype=int)
#            for ix,vx in enumerate(istateR):
#                indices = np.array(allothers([[vx],range(self.nP)],
#                                              (self.nS,self.nP)))
#                goals[indices] = npoints[vx]
            goals = np.tile(npoints, self.nP)
            goals = goals/goals.sum()

        if just_return is True:
            return goals
        elif just_return is False and convolute is True:
            self.C = goals
            self.lnC = np.log(goals)
        else:
            raise ValueError('Bad combination of just_return and convolute')



    def print_table_results(self, results = None):
        """ Prints a table using the results from the Example from actinfClass.

        Data for other runs can be passed as optional arguments. For this, the
        input 'results' should be a dictionary with the following entries:
            RealStates      [nT] a vector containing the real state for each
                            trial. It is assumed that this is in the full
                            space, not just the physical space.
            Actions         [nT] vector with actions at each trial.
            PostActions     [nU, nT] matrix with the posteriors over actions at
                            each trial.

        """

        from tabulate import tabulate
        import numpy as np
        trial_number = range(self.nT)
        pH = self.pH
        pL = self.pL
        rH = self.rH
        rL = self.rL

        expH = pH*rH
        expL = pL*rL

        if results is not None:
            Results = results
        else:
            Results = self.Example

        real_state, action_pair = np.unravel_index(Results['RealStates'],
                                                   (self.nS,self.nP), order='F')

        actions = Results['Actions']
        post_actions = Results['PostActions']

        table_data = []
        for t in range(self.nT):
            table_data.append([trial_number[t], real_state[t], action_pair[t],
                               actions[t], expL[action_pair[t]],
                               expH[action_pair[t]],
                               post_actions[t][0], post_actions[t][1]])

        table_headers = ['Trial','State','Act Pair', 'Action', 'expL', 'expH',
                         'ProbAct0', 'ProbAct1']
        print(tabulate(table_data, headers = table_headers))


    def all_points_and_trials(self, preserve_all = False):
        """ Wrapper to do all_points_and_trials_small/kolling, depending on the
        paradigm
        """
        if self.paradigm=='small':
            points, trials_needed = self.all_points_and_trials_small(
                                                                preserve_all)
            return points, trials_needed
        elif self.paradigm=='kolling':
            points, trials_needed = self.all_points_and_trials_kolling(
                                                                preserve_all)
            return points, trials_needed
        else:
            raise ValueError('Unknown paradigm')
            return None, None


    def all_points_and_trials_small(self, preserve_all = False):
        """ Calculates all the possible points attainable during this task, and
        all the possible number of trials in which the agent could have gotten
        each of these points.

        The 'runs' in which the points go over the threshold are eliminated, so
        that the ouputs are not so big. This can be toggled on and off with the
        input preserve_all.

        Outputs:
            points              Array with all the possible number of points
                                that can be gained in this game.
            trials_needed       Array that, for every element of 'points' (see
                                above) has the minimum number of trials needed
                                to attain them.
        """
        import itertools

        rL = self.rL
        rH = self.rH
        actPairs = np.unique(np.concatenate(([0],rL, rH)))
        numberActPairs = actPairs.size
        # All possible combinations of all actions:
        Vb = np.array(list(itertools.product(range(numberActPairs),
                                             repeat = self.nT)))

        # Number of trials needed for a given number of points:
        trialsNeeded = np.zeros(Vb.shape[0], dtype = int)
        # Calculate these points (save in V).
        for r, row in enumerate(Vb):
            trialsNeeded[r] = self.nT - Vb[r,Vb[r,:]==0].size
        points = np.sum(Vb, axis=1) #Points gained

        if preserve_all is False:
            under_thres = points<self.thres
            points = points[under_thres]
            trialsNeeded = trialsNeeded[under_thres]
        else:
            #Eliminate those past nS
            under_nS = points<self.nS
            points = points[under_nS]
            trialsNeeded = trialsNeeded[under_nS]
#        return points, trialsNeeded


        uniquePoints = np.unique(points)
        uniqueTrials = np.zeros(uniquePoints.shape, dtype = np.int64)
        for i, up in enumerate(uniquePoints):
            uniqueTrials[i] = min(trialsNeeded[points == up])

        return uniquePoints, uniqueTrials

    def all_points_and_trials_kolling(self, preserve_all = False):
        """ Calculates all the possible points attainable during this task, and
        all the possible number of trials in which the agent could have gotten
        each of these points.

        The 'runs' in which the points go over the threshold are eliminated, so
        that the ouputs are not so big. This can be toggled on and off with the
        input preserve_all.

        Outputs:
            points              Array with all the possible number of points
                                that can be gained in this game.
            trials_needed       Array that, for every element of 'points' (see
                                above) has the minimum number of trials needed
                                to attain them.
        """
        import numpy as np
        import itertools

        rL = self.rL
        rH = self.rH
        nP = self.nP

        Vb = np.array(list(itertools.product(range(3), repeat=nP)))
        V = np.zeros(Vb.shape)
        points = np.zeros(Vb.shape[0], dtype = int)
        trials = np.zeros(Vb.shape[0], dtype = int)
        for r, row in enumerate(Vb):
            for t, val in enumerate(row):
                V[r, t] = (val == 1)*rL[t] + (val == 2)*rH[t]
                points[r] = V[r,:].sum()
                trials[r] = self.nT - Vb[r, Vb[r,:]==0].size

        if preserve_all is False:
            under_thres = points<self.thres
            points = points[under_thres]
            trials = trials[under_thres]
        else:
            #Eliminate those past nS
            under_nS = points<self.nS
            points = points[under_nS]
            trials = trials[under_nS]

        uniquePoints = np.unique(points)
        uniqueTrials = np.zeros(uniquePoints.shape, dtype = int)
        for i, up in enumerate(uniquePoints):
            uniqueTrials[i] = min(trials[points == up])

        return uniquePoints, uniqueTrials