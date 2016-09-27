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
import numpy as np
import itertools
import utils
import actinfClass as afc
class betMDP(afc.Actinf):
    def __init__(self, paradigm = 'kolling compact', changePairs = True,
                 actionPairs = None, number_aps = None):
        """
        Initializes the instance with parameter values depending on the inputs.
        See the class's help for details on optional inputs.

        Additionally, calls setMDPMultVar() or setMDP, depending on inputs, to
        set all the MDP matrices for Active Inference.
        """
        if paradigm == 'kolling':
            self.nS = 2000 #max number of attainable points
            self.paradigm = 'kolling'
            self.pdivide = 1
            self.thres = 1000
            self.nT = 8
        elif paradigm == 'kolling compact':
            self.nS = 200
            self.paradigm = 'kolling'
            self.pdivide = 10
            self.nT = 8
            self.thres = 100
        else:
            self.nS = 12
            self.thres = 6
            self.paradigm = 'small'
            self.nT = 8

        self.nU = 2
        # Testing. Can be deleted now:
        self.number_aps = number_aps
        # END testing

        self.obsnoise = 0.001 #0.0001

        if changePairs is True:
            self.setMDPMultVar()
        else:
            self.setMDP(actionPairs)

    def priors_over_goals(self, threshold = None, just_return = False):
        import utils
        import numpy as np

        if threshold is None:
            thres = self.thres
        else:
            thres = threshold

        C = np.zeros(self.nS*self.nP)
        allothers = np.array(utils.allothers([range(self.nS-thres,self.nS),
                                              range(self.nP)],
                                              (self.nS,self.nP)))
        C[allothers] = 1
        C = C/sum(C)

        if just_return is False:
            self.C = C
            self.thres = thres
        elif just_return is True:
            return C

    def setActionPairs(self, number_aps = None):
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
        # The following is for testing purposes. Can be eliminated now:
        if number_aps is not None:
            nP = number_aps
            pL = pL[:nP]
            pH = pH[:nP]
            rL = rL[:nP]
            rH = rH[:nP]
        # END testing crap

        self.nP = nP
        self.pL = pL
        self.pH = pH
        self.rL = rL
        self.rH = rH

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
        C[-self.thres:] = 1
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
        necessarily constant). All the policies are then injected with a forced
        third action (B[2]) every other trial, thus doubling the number of
        trials; however, since all policies have B[2] every other trial, this
        action is always chosen and the agent can count on it.

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
                raise Exception('BadInput: ''parameters'' is not a dict')
            expectedParameters = ['pL','pH','rL','rH']
            compareSets = set(parameters) & set(expectedParameters)
            if compareSets != set(expectedParameters):
                raise Exception('BadInput: Paremeters does not contain the \
                required parameters')
            else:
                pL = parameters['pL']
                pH = parameters['pH']
                rL = parameters['rL']
                rH = parameters['rH']
                nP = len(pL)
        else:
            # Default action pairs

            self.setActionPairs(self.number_aps)
            nP, pL, pH, rL, rH = self.nP, self.pL, self.pH, self.rL, self.rH

        # Define observation matrix
        A = np.eye(nS*nP) + obsnoise/nP
        # Transition matrices
        # num2coords = np.unravel_index
        # coords2num = np.ravel_multi_index
        B = np.zeros((nU,nS*nP,nS*nP))
        for s in xrange(nS):
            for p in xrange(nP):
                nextsL = np.min((s+rL[p],nS-1)).astype(int)
                nextsH = np.min((s+rH[p],nS-1)).astype(int)
                mixL = utils.allothers([[nextsL],range(nP)],(nS,nP))
                mixH = utils.allothers([[nextsH],range(nP)],(nS,nP))
                this = utils.allothers([[s]     ,range(nP)],(nS,nP))
                ifrom = [np.ravel_multi_index([s,p],(nS,nP),order='F')]
                B[np.ix_([0], mixL, ifrom)] = pL[p]/nP
                B[np.ix_([0], this, ifrom)] = (1 - pL[p])/nP
                B[np.ix_([1], mixH, ifrom)] = pH[p]/nP
                B[np.ix_([1], this, ifrom)] = (1 - pH[p])/nP



        # Define priors over last state (goal)
        self.priors_over_goals()
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
        self.B = B
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
        for t in xrange(self.nT):
            table_data.append([trial_number[t], real_state[t], action_pair[t],
                               actions[t], expL[action_pair[t]], expH[action_pair[t]],
                               post_actions[t][0], post_actions[t][1]])

        table_headers = ['Trial','State','Act Pair', 'Action', 'expL', 'expH',
                         'ProbAct0', 'ProbAct1']
        print tabulate(table_data, headers = table_headers)