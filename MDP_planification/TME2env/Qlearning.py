#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:37:39 2020

@author: yass
"""

import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld2
from gym import wrappers, logger
import numpy as np
import copy
from collections import defaultdict 



if __name__ == '__main__':


    env = gym.make("gridworld-v1")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    
    
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    #print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    