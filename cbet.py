# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:14:49 2016

@author: dario
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:59:42 2016

@author: dario
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:36:14 2016

@author: dario
"""
import numpy as np
import itertools
import utils

class betMDP():
    def __init__(self, paradigm = 'kolling compact'):
        """
        Initializes the instance with some required values for the other
        methods. It can be executed with an extra input, a dictionary with keys
        nU, nS, nT, thres and obsnoise.
        """
        if paradigm == 'kolling':
            self.nS = 2000 #max number of attainable points
            self.paradigm = 'kolling'
            self.pdivide = 1
        elif paradigm == 'kolling compact':
            self.nS = 200
            self.paradigm = 'kolling'
            self.pdivide = 10
        else:
            self.nS = 8
            self.paradigm = 'small'
        
        self.nU = 3
        self.nT = 8
        self.thres = 1
        self.obsnoise = 0.001
        
                
        
    def setActionPairs(self):
        # fromwhere \in {'kolling','flat'}
        if self.paradigm == 'small':
            nP = 5;
            pL = np.linspace(0.7,0.9,nP)
            pH = np.linspace(0.4,0.2,nP)
            rL = np.round(np.linspace(1,self.nS-3,nP))
            rH = np.round(np.linspace(self.nS,3,nP))
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
#        return nP, pL, pH, rL, rH
            
    def setMDP(self,parameters = None):
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
    
        # Working parameters. To be imported later

        # Define the observation matrix
        A = np.eye(self.nS) + self.obsnoise
        A = A/sum(A)
        
        # Define transition matrices
        B = np.zeros((self.nU-1,self.nS,self.nS))
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
        V = np.array(list(itertools.product(range(0,self.nU-1),repeat = self.nT)))
        
        # Preparing inputs for MDP
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.S = S
        self.V = V
        self.alpha = 64
        self.beta = 4
        self.lambd = 0.005 # or 1/4?
        self.gamma = 20
        

    def setMDPMultVar(self,parameters=None):    
        # Working parameters. To be imported later
        nU = self.nU
        nS = self.nS
        nT = self.nT
        thres = self.thres
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
            self.setActionPairs()
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
                B[0,np.ravel_multi_index([nextsL,p],(nS,nP),order='F'),
                  np.ravel_multi_index([s,p],(nS,nP),order='F')] = pL[p]
                B[0,np.ravel_multi_index([s,p],(nS,nP),order='F'),
                  np.ravel_multi_index([s,p],(nS,nP),order='F')] = 1 - pL[p]
                B[1,np.ravel_multi_index([nextsH,p],(nS,nP),order='F'),
                  np.ravel_multi_index([s,p],(nS,nP),order='F')] = pH[p]
                B[1,np.ravel_multi_index([s,p],(nS,nP),order='F'),
                  np.ravel_multi_index([s,p],(nS,nP),order='F')] = 1 - pH[p]
                  
        
        for s in xrange(nS):
            allothers = np.array(utils.allothers([[s],range(nP),[0]],(nS,nP)))
            allothers = allothers.astype(int)
            B[np.ix_([2],allothers,allothers)] = 1
            
        # Define priors over last state (goal)
        C = np.zeros(nS*nP)
        allothers = np.array(utils.allothers([range(nS-thres,nS),range(nP)],
                                              (nS,nP)))
        C[allothers] = 1
        C = C/sum(C)
        
        # Priors over initial state
        D = np.zeros(nS*nP)
        D[0] = 1
        
        # Real initial state
        S = D.astype(int)
        
        # Policies: all combinations of 2 actions
        V = np.array(list(itertools.product(range(0,nU-1),repeat = nT)))
        Vs = V.copy()
        # Put the evolution action after every action
        v1, v2 = np.shape(V)
        twos = 2*np.ones((v1,v2))
        V = np.reshape(np.concatenate((V,twos)),(v1,2*v2), order='F')
        V = np.delete(V,V.shape[1]-1,axis=1)
        
        # Preparing inputs for MDP
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.S = S
        self.V = V
        self.Vs = Vs
        self.alpha = 64
        self.beta = 4
        self.lambd = 0.005 # or 1/4?
        self.gamma = 20


