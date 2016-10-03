# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:33:11 2016

@author: dario
"""
import numpy as np
def setForcedObservations(mdp, nPerms=10, sPerms = 10):
    """
    Of all the possible permutations of the action pairs shown, the choices
    made and the success of the bet, for each trial, this method takes a
    sample and creates observations for each one of these cases. For example,
    one of these samples would be a case in which the action pairs were shown
    in a given order (e.g. 2,3,5,8,1,6,4,7), at each trial the agent makes a
    choice (e.g. 1, 2, 2, 2, 1, 2, 1,1; where 1 is for safe and 2 for risky)
    and at each trial the bet is either successful or failed
    (e.g. 0,0,0,1,1,0,1,1) where 1 is success, 0 failure.

    The algorithm goes through every single action available to the MDP agent,
    as generated by cbet.setMDPMultVar, and takes as many as nPerms different
    action pair permutations and sPerms different, randomly chosen fail/succeed
    sequences.
    Uses:

    Output:
        states          [trials,nPerms,No. action sequences, sPerms] contains
                        every sequence of observations (points accumulated) for
                        every trial.
        sucPerms        [trials,sPerms] All success/failure sequences used.
        ranPerms        [trials,nPerms] All permutations of action pairs used.
    """

    ranPerms = np.array([np.random.choice(range(mdp.nP), size=mdp.nT)
                            for i in xrange(nPerms)])
    V = mdp.V
    nV, nT = V.shape
    sPIn = np.arange(nV, dtype=int)
    np.random.shuffle(sPIn)
    sPIn = sPIn[:sPerms]
    sucPerms = V[sPIn,:]
    state = np.zeros((mdp.nT,nPerms,nV,sPerms), dtype = int)
    # Calculate the different sequences
    for tr in xrange(1,mdp.nT):
        for per in xrange(nPerms):
            for v in xrange(nV):
                for ac in xrange(sPerms):
                    state[tr,per,v,ac] = (state[tr-1,per,v,ac]
                                +(mdp.rL[ranPerms[per,tr-1]]*(V[v,tr-1]==0)
                                +mdp.rH[ranPerms[per,tr-1]]*(V[v,tr-1]==1))
                                *(sucPerms[ac,tr-1]==1))
    return state, sucPerms, ranPerms
def setPriorGoals(mdp,selectShape='flat', rampX1 = None,
                  Gmean = None, Gscale = None,
                  convolute = True, just_return = True):
    """ Wrapper for the functions priorGoalsFlat/Ramp/Unimodal.

    Sets priors over last state (goals) in different shapes for testing
    the effects on the agent's behavior.

    The three options are 'flat', which is what is normally set in cbet.py,
    'ramp', which is a ramping-up that starts at threshold, and 'unimodal',
    which uses a Gaussian to set up a 'hump' after threshold.

    Uses:
        goals = setPriorGoals(mdp [,selectShape] [, rampX1] [, Gmean]
                              [, Gscale] [,convolute] [,just_return])

    Inputs:
        selectShape         {'flat','ramp','unimodal'} selects which shape is
                            to be used. When selecting 'ramp', the optional
                            input rampX1 can be selected (default 1). When
                            using 'unimodal', Gmean and Gscale can be set to
                            change the mean (in Trial number) and scale of the
                            Gaussian. Selecting a Gmean pre-threshold will
                            cause the 'hump' to be invisible and the priors
                            will be an exponential ramp down.
        rampX1              {x} determines the initial point for the ramp,
                            which uniquely determines the slope of the ramp.
        Gmean, Gscale       {x}{y} determine the mean and the scale of the
                            Gaussian for the unimodal version.
    Outputs:
        (Note: when just_return is False, nothing is returned)
        goals               [nS] are the resulting priors over last state.
    """
    if selectShape == 'flat':
        goals = priorGoalsFlat(mdp, convolute, just_return = True)
    elif selectShape == 'ramp':
        if rampX1 is None:
            raise ValueError('A value for rampX1 must be provided when using'+
                            ' ''ramp''')
        goals = priorGoalsRamp(mdp, rampX1 = rampX1,
                               convolute = convolute, just_return = True)
    elif selectShape == 'unimodal':
        if Gscale is None or Gmean is None:
            raise ValueError('Values for Gmean and Gscale must be provided '+
                             'when using ''unimodal''')
        goals = priorGoalsUnimodal(mdp, Gmean, Gscale, convolute = convolute,
                                   just_return = True)
    else:
        raise ValueError('selectShape can only be ''flat'', ''ramp'' or '+
                        '''unimodal''')
    if just_return is True:
        return goals
    elif just_return is False and convolute is True:
        mdp.C = goals
    else:
        raise ValueError('Bad combination of just_return and convolute')


def priorGoalsFlat(mdp, convolute = True, just_return = True):
    from utils import allothers

    if convolute is True:
        goals = np.zeros(mdp.nS*mdp.nP, dtype = float)

        indices = np.array(allothers([range(mdp.thres,mdp.nS),
                                  range(mdp.nP)], (mdp.nS,mdp.nP)),
                                 dtype = int)
        goals[indices] = 1.0/indices.size
    elif convolute is False:
        goals = np.zeros(mdp.nS, dtype = float)
        goals[mdp.thres:] = 1.0/goals[mdp.thres:].size

    if just_return is True:
        return goals
    if just_return is False and convolute is True:
        mdp.C = goals
    else:
        raise ValueError('Bad combination of just_return and convolute')


def priorGoalsRamp(mdp, rampX1, convolute = True, just_return = True):
    """ Creates goals as an increasing or decreasing ramp, depending on the
    value given for rampX1.

    rampX1 is the initial value. That is, the value of the first point
    after threshold. If rampX1 is smaller than M (where M is the number of
    points past-threshold), then the ramp is increasing. The slope is
    calculated automatically (since rampX1 determines it uniquely).
    """
    from utils import allothers


    thres = mdp.thres
    nS = mdp.nS
    pastThres = nS - thres
    nP = mdp.nP

    minX1 = 0
    maxX1 = 2.0/pastThres

    if rampX1<minX1 or rampX1>maxX1:
        raise ValueError ('Initial point X1 is outside of allowable '+
                          'limits for this task. min = %f, max = %f' % (minX1,
                          maxX1))
    if rampX1 == 1.0/pastThres:
        raise ValueError('rampX1 is invalid. For this value, use ''flat'''+
                            ' instead')

    slope = (2.0/pastThres - 2.0*rampX1)/(pastThres-1)

    stateRamp = rampX1 + slope*np.arange(pastThres)

    istateR = np.arange(mdp.thres, mdp.nS)

    if convolute is False:
        goals = np.zeros(nS)
        goals[thres:] = stateRamp
    else:
        goals = np.zeros(nS*nP)
        for ix,vx in enumerate(istateR):
            indices = np.array(allothers([[vx],range(mdp.nP)],(mdp.nS,mdp.nP)))
            goals[indices] = stateRamp[ix]
        goals = goals/goals.sum()
    if just_return is True:
        return goals
    elif just_return is False and convolute is True:
        mdp.C = goals
    else:
        raise ValueError('Bad combination of just_return and convolute')



def priorGoalsUnimodal(mdp, Gmean, Gscale,
                       convolute = True, just_return = True):
    """ Sets the priors over last state (goals) to a Gaussian distribution,
    defined by Gmean and Gscale.
    """
    from utils import allothers
    from scipy.stats import norm

    points = np.arange(mdp.nS)
    npoints = norm.pdf(points, Gmean, Gscale)
    npoints[:mdp.thres] = 0
    if convolute is False:
        goals = npoints
    else:
        goals = np.zeros(mdp.Ns)
        istateR = np.arange(mdp.thres, mdp.nS, dtype=int)
        for ix,vx in enumerate(istateR):
            indices = np.array(allothers([[vx],range(mdp.nP)],(mdp.nS,mdp.nP)))
            goals[indices] = npoints[vx]
        goals = goals/goals.sum()

    if just_return is True:
        return goals
    elif just_return is False and convolute is True:
        mdp.C = goals
    else:
        raise ValueError('Bad combination of just_return and convolute')


def setGammaPriors():
    # TODO: implement
    raise NotImplementedError('Maybe on Tuesday I will get around to it...')

def setForcedObservationMatrix(selectShape='gaussian'):
    # selectShape selects how to modify the observation matrix
    # TODO: implement
    raise NotImplementedError('Stop nagging me! I''ll do it when I do it')

def setForcedTransitionMatrix():
    # TODO: implement
    raise NotImplementedError('Yeah, yeah, I know, lazy...')

def setPriorsActionPairs(mdp,priorBeliefs = None):
    ''' Sets the belief used by the subject regarding which action pairs are
    more likely to be seen in the future. The change reflects in the third
    action available to the agent, which is the 'evolution' action. This action
    sets which action pair is to be seen in the next trial, so by altering its
    outcomes, the agent would expect to see some action pairs more than others.

    Note that the original idea is that these modified transition matrices be
    used only as the expectations of the agent, and not as the real ones;
    however, there is nothing in them that prevents them from being used as the
    real transition matrices.

    Uses:
        B = setPriorsActionPairs(mdp [, priorBeliefs])

    Inputs:
        priorBeliefs        [nP] is the desired prior distribution to be used
                            by the agent. If none is provided, the default is
                            flat.
    Outputs:
        B                   [3, nS, nS] set of three transition matrices: two
                            for the safe/risky choices, and one for the
                            evolution action. Note that B[1] and B[2] are
                            unchanged here.
    Imported modules:
        utils               Toolbox with some random tools like softmax.
    '''
    from utils import allothers
    if priorBeliefs is None:
        priorBeliefs = np.arange(mdp.nP)
    B = np.copy(mdp.B)
    for s in xrange(mdp.nS):
        indices = np.array(allothers([[s],range(mdp.nP),[0]],
                                     (mdp.nS,mdp.nP)), dtype=int)
        B[np.ix_([2],indices,indices)] = priorBeliefs
    return B

def runManyTrials(newActionPairs = False, apPriors = [],
                  newGoals = False, goalShape = [],
                  rampX1 = 1, Gmean = [], Gscale = 100,
                  forceObs = True, nPerms = 10, sPerms = 10,
                  newGammaPriors = False,
                  newObsMatrix = False,
                  newTranMatrix = False,
                  paradigm = 'small'):
    """ Performs many iterations of Active Inference modifying the
    'default' parameters from actinfClass. Which parameters are modified
    depends on the inputs passed.

    The number of repetitions will depend on the parameters nPerms and
    sPerms, as well as in the number of policies in the mdp class.

    Uses:
        There are certain inputs that should be given together, though if
        not, defaults will be used. The following inputs are examples:

        1.- [outputs] = runManyTrials(forceObs = True, nPerms = 100,
                                    sPerms = 100)
            This will create 100x100 permutations of chosen actions and
            forced outcomes. The forced outcomes will be passed through
            the observation matrix to generate forced observations, which
            will be shown to the agent at each trial. Since this case is
            the reason for this method, it is the only one that defaults
            to true.
        2.- [outputs] = runManyTrials(newGoals = True, goalShape = 'ramp',
                                        rampX1 = 2)
            This option changes the default 'flat' priors over last state
            (goals) into a ramp shape, with a 'slope' of 2. Without giving
            any values for forceObs, the defaults of yes will be used and
            many mini-blocks will be used (see above). See the help of
            setPriorGoals for more details.

        All other 'switches' (e.g. newGammaPriors) require no additional
        parameters. All these options can be used in combination with one
        another with no special considerations.

    Inputs:
        forceObs        {bool} Creates (a great number of) observations which
                        will be shown to the agent, removing the random
                        nature of the task (the agent is unaware of this).
                        Use in combination with nPerms and sPerms. The
                        forced observations are generated with
                        setForcedObservations; see its help file for more
                        details.
        nPerms          {x} To be used in combination with forceObs=True.
                        Determines the number of permutations of the
                        action pairs used for the different mini-blocks.
        sPerms          {x} To be used in combination with forceObs=True.
                        Determines the number of forced outcome
                        (win/lose) for every trial in every miniblock of
                        nPerms.
        newActionPairs  {bool} Changes the priors over action pairs from
                        the actinfClass default of 'flat' to the
                        distribution given by apPriors. See the help file
                        of setPriorsActionPairs for details.
        apPriors        [nS] Use in combination with newActionPairs=True.
                        Determines the priors over action pairs to be
                        used by the agent.
        newGoals        {bool} Changes the shape of the priors over the
                        last state (goals) to a different shape, given by
                        the goalShape input. See the help file for
                        setPriorGoals for details.
        goalShape       {'flat','ramp','unimodal'} Used in combination
                        with newGoals, sets the desired shape of the
                        goals. 'ramp' has additional parameter rampX1,
                        and 'unimodal' has additional parameters Gmean
                        and Gscale.
        rampX1       {x} Used in combination with newGoals = True and
                        goalShape = 'ramp', sets the slope (ish) of the
                        used ramp. See setPriorGoals for details.
        Gmean, Gscale   {x} Used in combination with newGoals=True and
                        goalShape = 'unimodal', they set the mean and the
                        scale of the Gaussian used. See setPriorGoals for
                        details.
        newGammaPriors  Not yet implemented.
        newObsMatrix    Not yet implemented.
        newTranMatrix   Not yet implemented.
    Outputs:
    Note: ntPerm is the total number of permutations of forced obs used
        asta            [ntPerm, Trials] States visited by the agent for every
                        permutation and every trial.
        abel            [ntPerm, Trials] Updated beliefs at every trial.
        aact            [ntPerm, Trials] Action chosen at every trial.
        aobs            [ntPerm, Trials] Observations at every trial.

    Imported modules:
        cbet            Class holding the mdp parameters for the Kolling 2014
                        paradigm.
        actinfClass     Class holding the methods for calculating all the
                        posteriors and stuff for Active Inference.

    """
    import betClass
#    import actinfClass as af
    # for testing purposes:
    import sys
    import tqdm

    # setup of the MDP parameters for the task and the actinf class
    mdp = betClass.betMDP(paradigm)

    T = mdp.V.shape[1]


    # here the modification of default stuff using the methods above
    if newActionPairs == True:
        if len(apPriors) != T:
            raise Exception('BadInput: apPriors is the wrong size')
        newB = setPriorsActionPairs(mdp,apPriors)
    else:
        newB = mdp.B

    if newGoals == True:
        if goalShape not in ('flat','ramp','unimodal'):
            raise Exception('BadInput: goalShape must be in {''flat'','+
                            ' ''ramp'','' unimodal''}' )
        # Warning: overwrites previous goals:
        mdp.C = setPriorGoals(mdp, goalShape, rampX1, Gmean, Gscale)

    if forceObs == True:
        fstate, sucPerms, ranPerms = setForcedObservations(mdp,
                                                           nPerms,
                                                           sPerms)
        # Reshape the fstate array to [perm, trials] size:
        s1,s2,s3,s4 = np.shape(fstate)
        fstate = np.reshape(fstate,(s1,s2*s3*s4),order='F').T
#        fstate = fstate.T
        fV = np.zeros((s1,1,s3,1))
        fV[:,0,:,0] = mdp.V.T
        fV = np.tile(fV,(1,s2,1,s4))
        fV = np.reshape(fV,(s1,s2*s3*s4), order='F').T
#        return fstate, fV
    if newGammaPriors == True:
        # TODO:
        pass
    if newObsMatrix == True:
        # TODO:
        pass
    if newTranMatrix == True:
        # TODO:
        pass



    T = mdp.V.shape[1]
    nPerms = np.shape(fstate)[0]

    asta = np.zeros((nPerms,T),dtype=int)
    abel = np.zeros((nPerms,T,mdp.Ns),dtype=float)
    aact = np.zeros((nPerms,T),dtype=int)
    aobs = np.zeros((nPerms,T),dtype=int)
    print 'Finished setting up. Ready for simulations.\n'
    sys.stdout.flush()
    for cper in tqdm.trange(nPerms):
        obs = np.zeros(T, dtype=int)    # Observations at time t
        act = np.zeros(T, dtype=int)    # Actions at time t
        sta = np.zeros(T, dtype=int)    # Real states at time t
        bel = np.zeros((T,mdp.Ns))      # Inferred states at time t
        sta[0] = np.nonzero(mdp.S)[0][0]
        # Dummy values:
        PosteriorPreviousState = 0
        PastAction = 0
        PriorPrecision = 1.0

        # Generate the observations from fstate:
        if forceObs is True:
            for t,st in enumerate(fstate[cper,:]):
                obs[t] = mdp.sampleNextObservation(st)
                act[t] = fV[cper,t]
        for t in xrange(T-1):
            # Sample an observation from current state if no forced obs.
            if forceObs is False:
                obs[t] = mdp.sampleNextObservation(sta[t])
            # Update beliefs over current state and posterior over actions and
            # precision
            bel[t,:], P, W = mdp.posteriorOverStates(obs[t], t, mdp.V,
                                                    PosteriorPreviousState,
                                                    PastAction,
                                                    PriorPrecision,
                                                    newB)
            # Sample an action and the next state using posteriors
            if forceObs is False:
                act[t], sta[t+1] = mdp.sampleNextState( sta[t], P)
            # Prepare inputs for next iteration
            PosteriorPreviousState = bel[t]
            PastAction = act[t]
            PriorPrecision = W
#        print 'trial: %d\n' % t
        # Save the stuff permanently:
        asta[cper,:] = sta
        abel[cper,:,:] = bel
        aact[cper,:] = act
        aobs[cper,:] = obs
#        print 'Check 2: went through one iteration of cper'
        sys.stdout.flush()
#        if cper == 1:
#            print 'Check 3: went through the second iteration of cper'
#            sys.stdout.flush()
#            return 0

    return asta,abel,aact,aobs

def posteriors_all_obs(newGoals = False, goalShape = [],
                       rampX1 = 1, Gmean = [], Gscale = 100):
    """ Generates the posteriors over actions for every possible observation
    that the agent can get during the task of 'small', in betClass.

    Output:
        postActions         Dictionary whose keys are tuples
                            (trial, state, action pair), representing the
                            current trial number, the state observed (number
                            of points) and the action pair observed. The
                            elements corresponding to each tuple are the
                            posterior distributions over actions.
        postPrecision       Dictionary whose keys are tuples
                            (trial, state, action pair), representing the
                            current trial number, the state observed (number
                            of points) and the action pair observed. The
                            elements corresponding to each tuple are the
                            posterior precision.
    """

    import betClass as bc
    import tqdm



    mabe = bc.betMDP(paradigm='small')
    Ns = mabe.Ns # Number of hidden states (without actions)
    nP = mabe.nP # Number of action pairs
    nS = mabe.nS # Number of physical states = Ns/nP


    if newGoals == True:
        if goalShape not in ('flat','ramp','unimodal'):
            raise Exception('BadInput: goalShape must be in {''flat'','+
                            ' ''ramp'','' unimodal''}' )
        # Warning: overwrites previous goals:
        mabe.C = setPriorGoals(mabe, goalShape, rampX1, Gmean, Gscale)


    forcedStates, fTrials = mabe.all_points_and_trials(preserve_all = True)
    postPrecision = {}
    postActions = {}
    for s, sta in enumerate(tqdm.tqdm(forcedStates)):
        for ap in range(nP):
            cState = ap*nS + sta # Current hidden state
            cStateVector = np.zeros(Ns) # for actinfClass
            cStateVector[cState] = 1
            obs = mabe.sampleNextObservation(cState) # Current observation
            for tr in range(fTrials[s], mabe.nT):
                dummybel, P, W = mabe.posteriorOverStates(obs, tr, mabe.V,
                                        cStateVector, 1, mabe.gamma,
                                        newB = None)
                postActions[(tr, sta, ap)] = P
                postPrecision[(tr, sta, ap)] = W
    return postActions, postPrecision, mabe







#%% Run stuff here
def compare_goal_shapes():
    """ Script to compare the effects of the three goal shapes (flat, ramp and
    unimodal) on the posteriors over actions and precision.

    The comparison of precisions is a funny thing to do, as precision is
    something that evolves as the mini-block progresses (not so with posteriors
    over actions). For every forced observation, the same prior on precision is
    used, and as such, only the change in precision should be compared.
    """
    import betClass as bc
    import TestParameters as tp

    mabes = bc.betMDP('small')

    outDict = {}
    # flat
    postActFlat, postPreFlat, mabeFlat = tp.posteriors_all_obs(newGoals = True,
                                            goalShape = 'flat')
    outDict['flat'] = {'postAct':postActFlat, 'postPre':postPreFlat,
                       'mdp':mabeFlat}
    # ramp up
    postActRup, postPreRup, mabeRup = tp.posteriors_all_obs(newGoals = True,
                                            goalShape = 'ramp', rampX1 = 0.001)
    outDict['rampup'] = {'postAct':postActRup, 'postPre':postPreRup,
                       'mdp':mabeRup}
    # ramp down
    postActRdo, postPreRdo, mabeRdo = tp.posteriors_all_obs(newGoals = True,
                                            goalShape = 'ramp', rampX1 = 0.2)
    outDict['rampdown'] = {'postAct':postActRdo, 'postPre':postPreRdo,
                       'mdp':mabeRdo}
    # unimodal
    postActUni, postPreUni, mabeUni = tp.posteriors_all_obs(newGoals = True,
                                            goalShape = 'unimodal',
                                            Gmean = mabes.thres+1,
                                            Gscale = 2)
    outDict['unimodal'] = {'postAct':postActUni, 'postPre':postPreUni,
                       'mdp':mabeUni}

    return outDict
