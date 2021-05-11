#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:37:39 2020

@author: yassine
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random

from torch.autograd import Function


class AgentQL(object):

	def __init__(self, env, alpha=1, gamma=0.99, eps=0.0001, sarsa=False):
		self.env = env
		self.nb_act = 4
		self.state = str(env.reset())
		self.sarsa = sarsa

		# On initialise la Q-table
		self.Q = dict()
		if self.state not in self.Q : 
			self.Q[self.state] = np.zeros(self.nb_act)
		self.alpha = alpha
		self.gamma = gamma
		self.action = int(np.random.randint(4, size=1)[0])
		self.eps = eps
		
		self.t = 0

	def act(self, nxtState, rt, done):
		if str(nxtState) not in self.Q : 
			self.Q[str(nxtState)] = np.zeros(self.nb_act)

		Q_state_a = self.Q[self.state][self.action]
		a_suiv_val = np.max(self.Q[str(nxtState)])
		self.Q[self.state][self.action] = Q_state_a + self.alpha*(rt+self.gamma*(a_suiv_val-Q_state_a))
		self.action = self.choose(nxtState)
		self.state = str(nxtState)
		self.t += 1

		return self.action

	def choose(self, obs): # epsilon greedy

		# renvoie l'action en fonction de l'observation
		if self.sarsa : # SARSA
			if np.random.rand(1) < self.eps : 
				return np.random.randint(4, size=1)
			return np.argmax(self.Q[str(obs)])

		return np.argmax(self.Q[str(obs)]) # Q-Learning


class AgentDynaQ(object):

	def __init__(self, env, alpha=1, gamma=0.99, steps = 10, eps=0.0001, sarsa=False):
		self.steps = steps
		self.env = env
		self.nb_act = 4
		self.state = str(env.reset())
		self.eps = eps
		self.sarsa = sarsa
		
		# On initialise la Q-table
		self.Q = dict()
		if self.state not in self.Q : 
			self.Q[self.state] = np.zeros(self.nb_act)

		# On initialise le modele
		self.model = dict()
		if self.state not in self.model : 
			self.model[self.state] = dict()

		self.alpha = alpha
		self.gamma = gamma
		self.action = int(np.random.randint(4, size=1)[0])
		self.t = 0


	def updateModel(self, state, nxtState, action, reward):
		if self.state not in self.model : 
			self.model[self.state] = dict()

		for a in range(self.nb_act):
			if a != action:
				self.model[state][a] = (0, state)

		self.model[state][action] = (reward, str(nxtState))

	def act(self, nxtState, rt, done):

		if str(nxtState) not in self.Q : 
			self.Q[str(nxtState)] = np.zeros(self.nb_act)

		Q_state_a = self.Q[self.state][self.action]
		a_suiv_val = np.max(self.Q[str(nxtState)])
		self.Q[self.state][self.action] = Q_state_a + self.alpha*(rt+self.gamma*(a_suiv_val-Q_state_a))
		
		self.action = self.choose(nxtState)
		self.state = str(nxtState)
		self.t += 1

		# Partie Dyna-Q

		self.updateModel(self.state, nxtState, self.action, rt)
		
		for _ in range(self.steps):
			# on choisi un etat au hasard

			ind = np.random.choice(range(len(self.model.keys())))
			_st = list(self.model)[ind]

			# on choisi une action au hasard
			ind = np.random.randint(len(self.model[_st]), size=1)[0]
			_act = list(self.model[_st])[ind]

			_rew, _nxtState = self.model[_st][_act]


			if str(_nxtState) not in self.Q : 
				self.Q[str(_nxtState)] = np.zeros(self.nb_act)

			Q_state_a = self.Q[_st][_act]
			a_suiv_val = np.max(self.Q[str(_nxtState)])

			self.Q[_st][_act] = Q_state_a + self.alpha *(_rew + self.gamma*(a_suiv_val-Q_state_a))
		
		return self.action

	def choose(self, obs): # epsilon greedy
		# renvoie l'action en fonction de l'observation

		if self.sarsa : # SARSA
			if np.random.rand(1) < self.eps : 
				return np.random.randint(self.nb_act, size=1)[0]
			return np.argmax(self.Q[str(obs)])

		return np.argmax(self.Q[str(obs)]) # Q-Learning



if __name__ == '__main__':


	env = gym.make("gridworld-v0")
	env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

	env.seed(0)  # Initialise le seed du pseudo-random
	
	#print("Actions possibles :",env.action_space)  # Quelles sont les actions possibles

	episode_count = 5000
	reward = 0
	done = False
	rsum = 0
	FPS = 0.0001
	alph = 0.05
	eps = 0.0001
	sarsa = False

	liste_tps = []
	liste_rew = []
	liste_action = []
	
	agent = AgentQL(env, alpha=alph, gamma=0.99, sarsa=sarsa)
	#agent = AgentDynaQ(env, alpha=alph, gamma=0.99)
	#liste_agents = [AgentQL(env, alpha=alph, gamma=0.99, sarsa=False)]


	for i in range(episode_count): # pour chaque episode 
		obs = env.reset()
		env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
		if env.verbose:
			env.render(FPS)
		j = 0
		rsum = 0
		while True:
			action = agent.act(obs, reward, done)
			obs, reward, done, _ = env.step(action)
			rsum += reward
			if env.verbose:
				env.render(FPS)
			if done:
				print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
				liste_tps.append(i)
				liste_rew.append(rsum)
				liste_action.append(j)
				break
	

	
	
	plt.show()
	plt.plot(liste_tps, liste_rew, color='blue')
	plt.title("Somme des rewards en fonction des époques")
	plt.xlabel("Epoques")
	plt.ylabel("Rewards")
	plt.savefig("tme2_reward")
	plt.show()

	plt.plot(liste_tps, liste_action, color='red')
	plt.title("Nombre d'action en fonction des époques")
	plt.xlabel("Itérations")
	plt.ylabel("Actions")
	plt.savefig("actions")
	plt.show()

	print("done")
	env.close()