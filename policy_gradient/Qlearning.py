#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:37:39 2020

@author: yass
"""

import matplotlib

matplotlib.use("TkAgg")
import gym
import numpy as np
from collections import defaultdict 
import gridworld2


def greedy(opt,**kwargs):
    return np.argmax(opt)

def egreedy(opt,eps=0.1,**kwargs):
    a= np.argmax(opt)
    if np.random.rand()<eps:
        return a
    return np.random.choice(np.array(
                                    [i for i in range(len(opt)) if i !=a]
                                    )
                            )


class AgentQl:
    def __init__(self,env,eve=None,**kwargs):
        self.env=env
        self.num_action=env.action_space.n
        self.Q=defaultdict(lambda:np.zeros(self.num_action))
        print(self.Q)
        self.alpha=0.05
        self.discount_factor=0.99
        self.eve=eve if eve is not None else egreedy
        self.play_greedy=True
        self.stat=env.reset()
        self.action=self.choose(self.stat)
        print(self.action)
        
    def act(self,state,rew,done):
        best_next_action = self.choose(state) 
        td_target = rew + self.discount_factor * self.Q[state][best_next_action] 
        td_delta = td_target - self.Q[self.stat][self.action] 
        self.Q[self.stat][self.action] += self.alpha * td_delta 
        
        self.action=self.choose(state)
        print("action"+str(self.action))
        self.stat=state
        
        return self.action
    
    def choose(self,state):
        if self.play_greedy:
            return greedy(self.Q[state])
        return self.eve(self.Q[state])

def play(env,agent,i,FPS=0.0001):
    state=env.reset()
    reward,done=0,False
    rsum=0
    cpt=0
    while not done:
        action=agent.act(state,reward,done)
        state,reward,done,_=env.step(action)
     
        rsum += reward
        cpt += 1
     
        if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(cpt) + " actions")
                break

    print("done")
    env.close()


if __name__ == '__main__':


    env = gym.make("gridworld-v1")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    
    
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    #print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    
    ql_agent=AgentQl(env)
    episode_count = 100
    FPS=0.0001
    rewards=list()
    for i in range(episode_count):
        state=env.reset()
        reward,done=0,False
        rsum=0
        cpt=0
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            	env.render(FPS)
        while not done:
            action=ql_agent.act(state,reward,done)
            state,reward,done,_=env.step(action)
         
            rsum += reward
            cpt += 1
            rewards.append(reward)
            if done:
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(cpt) + " actions"+", Final reward ="+str(reward))
                    break

    print("done")
    print("rewards",np.array(rewards).mean())
    env.close()
	

    