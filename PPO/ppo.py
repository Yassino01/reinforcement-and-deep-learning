import time
import subprocess
from collections import namedtuple,defaultdict
import logging
import json
import os
import yaml


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *
from torch.utils.tensorboard import SummaryWriter

import sys
import threading
import argparse
import random
import gym

import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from memory import *


matplotlib.use("TkAgg")


class PPO:
    def __init__(self, env, insize, layers, config, device=torch.device('cpu'), lr=0.0005, gamma=0.99, beta=0.01, K=3, lanbda=0.95):

        inSize = insize
        self.device = device
        self.env = env
        self.action_space = env.action_space.n
        self.discount_factor = gamma
        
        #losses
        self.loss_v = torch.nn.SmoothL1Loss()
        self.kl = torch.nn.KLDivLoss()
        
        #feature_extractor
        self.fe = config["featExtractor"](env)
        
        #Sampling
        self.stateSample = [] 
        self.actionSample = [] 
        self.rewardSample = [] 
        self.nextStateSample = [] 
        self.logitsSample = [] 
        self.doneSample = []
        self.data = []

        #params
        self.state = None
        self.t = 0
        self.beta = beta
        self.delta = 1
        self.K = K
        self.lanbda = lanbda

        self.pi = nn.ModuleList([])
        
        for x in layers:
            self.pi.append(nn.Linear(inSize, x).float())
            inSize = x
        self.pi.append(nn.Linear(inSize, 2).float())

        self.arrdkl = []
        
        inSize = insize
        self.old_pi = nn.ModuleList([])
        for x in layers:
            self.old_pi.append(nn.Linear(inSize, x).float())
            inSize = x
        self.old_pi.append(nn.Linear(inSize, 2).float())
        
        inSize = insize
        self.v = nn.ModuleList([])
        for x in layers:
            self.v.append(nn.Linear(inSize, x).float())
            inSize = x
        self.v.append(nn.Linear(inSize, 1).float())
        
        self.optimizer = torch.optim.Adam(all_params(self.pi.parameters(), self.v.parameters()), lr=lr)

    ## Defining NN networks for pi old_pi and vi

    def Pi(self, x, softmax_dim = 0):
        for layer in self.pi[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.pi[-1](x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def old_Pi(self, x, softmax_dim = 0):
        for layer in self.old_pi[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.old_pi[-1](x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def V(self, x):
        for layer in self.v[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.v[-1](x)
        return x

    def act(self, obs, reward, done):
        prob = self.Pi(torch.tensor(obs).float())
        m = torch.distributions.Categorical(prob)
        action = m.sample().item()
        self.prob = prob
        return action
    

    def add_transition(self, state, action, reward, next_state, done):
        proba = self.prob[action]
        transition = (state, action, reward, next_state, proba, done)
        
        self.stateSample.append(state)
        self.actionSample.append([action])
        self.rewardSample.append([reward])
        self.nextStateSample.append(next_state)
        self.logitsSample.append([proba])
        self.doneSample.append([int(not done)])
        
    
    
    def update(self):
        stateSample = torch.tensor(self.stateSample)
        actionSample = torch.tensor(self.actionSample)
        rewardSample = torch.tensor(self.rewardSample)
        nextStateSample = torch.tensor(self.nextStateSample)
        doneSample = torch.tensor(self.doneSample)
        logitsSample = torch.tensor(self.logitsSample)  


        for i in range(self.K):
            new_pi_s = self.Pi(stateSample.float())
            old_pi_s = self.old_Pi(stateSample.float())
            dkloss = self.kl(old_pi_s, new_pi_s)
        
            td_lambda = rewardSample + self.discount_factor * self.V(nextStateSample.float()) * doneSample
            delta = td_lambda - self.V(stateSample.float())
            delta = delta.detach().numpy()
            advantageArr = []
            advantage = 0
           
            for delta_t in reversed(delta):
                advantage = self.discount_factor * self.lanbda * advantage + delta_t[0]
                advantageArr.append([advantage])
            advantageArr.reverse()
            advantage = torch.tensor(advantageArr, dtype=torch.float)
            
            pi = self.Pi(stateSample.float(), softmax_dim=1)
            pi_action = pi.gather(1,actionSample)
            ratio = (pi_action) / (logitsSample)
            surr1 = ratio * advantage
            loss = -surr1 + F.smooth_l1_loss(self.V(stateSample.float()) , td_lambda.float().detach()) + self.beta * dkloss
           
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.old_pi.load_state_dict(self.pi.state_dict())
           
            self.optimizer.step()
    
        self.arrdkl.append(dkloss)
        self.stateSample = [] 
        self.actionSample = []
        self.rewardSample = [] 
        self.nextStateSample = [] 
        self.logitsSample = [] 
        self.doneSample = []        

def all_params(*args):
    for gen in args:
        yield from gen



if __name__ == '__main__':
   
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    #freqTest = config["freqTest"]
    freqTest = 20 # 100

    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    envName = config["env"]
    if envName == "CartPole-v1":
        insize = 4
    if envName == "LunarLander-v2":
        insize = 8
    freqSave = config["freqSave"]
  

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/PPO" + "-" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    previous_ob = env.reset()

    agent = PPO(env, insize=insize, layers=[32], config=config)

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = False
        else:
            verbose = False
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("End of test, mean reward over the last {} actions: {}".format(freqTest,mean / freqTest))
            itest += 1
            print("New test time! ")
            mean = 0
            agent.test = True

            if i % freqTest == nbTest and i > freqTest:
                logger.direct_write("rewardTest", mean / nbTest, itest)
                agent.test = False
            
            if i % freqSave == 0:
                agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        done = False
        while not done:
            for t in range(20):
                if verbose:
                    env.render()
                action = agent.act(previous_ob, reward, done)
                ob, reward, done, _ = env.step(action)
                agent.updateArrays(previous_ob, action, reward/100.0, ob, done)
                previous_ob = ob
                j += 1
                rsum += reward
                if done:
                    reward = 0
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    logger.direct_write("reward", rsum, i)
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                    previous_ob = env.reset()
                    break
            agent.update()
    
    env.close()
    