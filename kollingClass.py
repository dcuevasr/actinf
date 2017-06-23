#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:17:53 2017

@author: dario
"""
import itertools as it
import numpy as np

class kolling(object):
    def __init__(self, max_points = 200, nT = 8):
        """ Use init_opt_rbs = True as long as the optimal risk bonus scale
        does not depend on the current points. It will save a lot of repeated
        computations if many posteriors are being calculated."""
        self.max_points = max_points
        self.nT = nT
        self._action_pairs()
        self.point_progression = []
        self.seen_pair = []
        self.all_orbs = {}
        self.count_forb = 0
    def _action_pairs(self):
        nP = 8
        pL = np.array([90, 60, 75, 55, 90, 60, 75, 80], dtype=float)/100
        pH = np.array([35, 35, 35, 20, 45, 45, 40, 30], dtype=float)/100
        rL = np.array([100, 180, 145, 145, 115, 150, 170, 120],
                      dtype = int)/10
        rH = np.array([265 ,260 ,245 ,350, 240, 190, 245, 210],
                      dtype = int)/10

        self.nP = nP
        self.pL = pL
        self.pH = pH
        self.rL = rL
        self.rH = rH

    def _ap_left(self):
        """ Returns, as a list of indices, the APs that haven't been seen."""
        return [x for x in np.arange(self.nP) if x not in self.seen_pair]

    def reset_agent(self, hard = False):
        """ Resets the accumulated observations of the instance."""
        self.point_progression = []
        self.seen_pair = []
        if hard:
            self.all_orbs = {}

    def _all_actions(self, trial):
        """ Returns all combinations of 2 actions in trial trials."""
        return np.array(list(it.product([0,1], repeat=self.nT - trial)))

    def _risk_pressure(self, trial, cpoints, threshold):
        """ Returns risk pressure."""
        return (threshold - cpoints)/(self.nT - trial)

    def find_optimal_risk_bonus(self, trial, thres, cpoints, cpair):
        """ Calculates the optimal risk bonus scale for the given context."""

        opt_rb = 0
        max_points = 0
        for risk_bonus in np.arange(0, 1.1, 0.1):
            apt_left = self._ap_left()
            for ap_seq in [apt_left]:#it.permutations(apt_left,r = (self.nT - trial)):
                npoints = cpoints
                for t in np.arange(len(ap_seq)):
                    pL = self.pL[ap_seq[t]]
                    pH = self.pH[ap_seq[t]]
                    rL = self.rL[ap_seq[t]]
                    rH = self.rH[ap_seq[t]]
                    vL = pL*rL + risk_bonus*(1-pL)*rL
                    vH = pH*rH + risk_bonus*(1-pH)*rH
                    npoints = npoints + rH*(vL < vH) + rL*(vL >= vH)

                if npoints > max_points:
                    max_points = npoints
                    opt_rb = risk_bonus
        self.point_progression.append(cpoints)
#        self.seen_pair.append(cpair)
        return opt_rb

    def posterior_over_actions(self, trial, thres, cpoints, cpair, inv_temp):
        """ Calculate the posterior over actions with a softmax."""
#        if (trial, cpair) in self.all_orbs:
#            opt_rbs = self.all_orbs[(trial, cpair)]
#        else:
#            opt_rbs = self.find_optimal_risk_bonus(trial, thres, cpoints, cpair)
#            self.all_orbs[(trial, cpair)] = opt_rbs
        opt_rbs = self.find_optimal_risk_bonus(trial, thres, cpoints, cpair)

        vL = self.pL[cpair]*self.rL[cpair] + opt_rbs*(1-self.pL[cpair])*self.rL[cpair]
        vH = self.pH[cpair]*self.rH[cpair] + opt_rbs*(1-self.pH[cpair])*self.rH[cpair]
        tmp_prob = np.exp([inv_temp*vL, inv_temp*vH])
        return tmp_prob/tmp_prob.sum()

    def get_likelihood(self, subjects = None, inv_temp_vec = None, tini=0):
        """ Imports behavioral data and calculates the data likelihood for the
        given model values.
        """

        import import_data as imda

        data, _ = imda.main()

        if subjects is None:
            subjects = range(len(data))
        elif isinstance(subjects, int):
            subjects = subjects,

        data = [data[s] for s in subjects]
        logli = {}
        for inv_temp in inv_temp_vec:
            logli[inv_temp] = 0
            for s, datum in enumerate(data):
                for g, game in enumerate(datum['points']):
                    game_ut = game[datum['points'][g,:]<datum['points'][g]]
                    for t, points in enumerate(game_ut):
                        thres = datum['threshold'][g]
                        cpair = datum['reihe'][g,t] - 1
                        post_act = self.posterior_over_actions(t, thres,
                                                       points, cpair, inv_temp)

                        logli[inv_temp] += np.log(post_act[0]**((datum['choice'][g,t]==0)*1) *
                                       post_act[1]**((datum['choice'][g,t]==1)*1))
                    self.reset_agent()

        return logli
