#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:19:51 2020

@author: yass
"""
import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld2
from gym import wrappers, logger
import numpy as np
import cop


env = gym.make("gridworld-v1")
env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})


env.seed(0)  # Initialise le seed du pseudo-random
print(env.action_space)  # Quelles sont les actions possibles

#print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
 
env.render()  # permet de visualiser la grille du jeu 

env.render(mode="human") #visualisation sur la console

MDP = env.getMDP()  # recupere le mdp : statedic

statedic,p=MDP
{

def iteration_value(MDP,eps=0.01,gama=0.99):
    statedic, p = MDP
    n_states=len(statedic.keys())
    pi=np.zeros(n_states)-1
    V=np.zeros(n_states)
    act_vals=list()
    act_values=list()

    while True:
        Vprv=V.copy()
        for s in p.keys():
            act_values=list()
            for a in p[s]:
                transitions=p[s][a]
                va=0
                for transition in transitions:
                    prob,s_dest,r,done=transition
                    va=va+prob*(r+gama*Vprv[s_dest])
                act_values.append(va)
            print(act_values)
            V[s]=max(act_values)
        
        if np.abs(V-Vprv).max() < eps:
            break

    
    for s in p.keys():
        print("eatat: {}".format(s))
        act_vals=list()
        for a in p[s]:
            print(a)
            transitions=p[s][a]
            va=0
            for transition in transitions:
                    prob,s_dest,r,done=transition
                    va+=prob*(r+gama*V[s_dest])
            print("val_act= {}".format(va))
            act_vals.append(va)
            print(act_vals)
        
        pi[s]=act_vals.index(max(act_vals))
        
    return V,pi

V,pi=iteration_value(MDP,eps=0.01,gama=0.99)


            
            
            
            
                    
                
                
            
    
    
    
    