
import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
from collections import deque
import torch
from torch.distributions import *

class Buffer(object):
    def __init__(self, size, device="cpu"):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        states = torch.as_tensor(np.array(states), device=self.device)
        actions = torch.as_tensor(np.array(actions), device=self.device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(np.array(next_states), device=self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)

        return states, actions, rewards, next_states, dones



class AgentDQN:
    def __init__(self,env,opt,epochs,play_greedy=False,test=False,laten_dim=32,nb_layers=2,batch_size=50,gama=0.99,lr=5*1e-4,buffer_size=10000,steps=500):
        
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.opt=opt
        self.env=env
        self.epochs=epochs
        self.discount_factor=gama
        self.play_greedy=play_greedy  # eps-greedy
        self.test=test
        self.eps0=1 # for choice e-greedy
        self.epsilon=lr
        self.mu = 100/self.epochs
        self.t=0

        # Nural networks for Q,Qhat and optimizer
        layers=[laten_dim for i in range(nb_layers)]
        self.q = NN(self.featureExtractor.outSize, self.env.action_space.n,layers)
        self.optim = torch.optim.Adam(params=self.q.parameters(),lr=self.epsilon)
        self.qhat = copy.deepcopy(self.q)
        
        self.criterion = torch.nn.SmoothL1Loss()
        self.steps4qhat = steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = Buffer(self.buffer_size)
        self.prev_state = None
        self.prev_action = None
    
    def act(self,state,reward,done,epoch):
        
        if self.t % self.steps4qhat == 0: # qhat update
            self.qhat = copy.deepcopy(self.q)
        
        state = torch.tensor(self.featureExtractor.getFeatures(state), dtype=torch.float)
        qs = self.q(state)

        #If beginig of sequence
        if self.prev_state == None:
            self.prev_state = state
            self.prev_action  = self.env.action_space.sample()
            return self.prev_action

        # epsilon greedy if not test
        eps = self.eps0 / (1 + self.mu * epoch)
        if self.test==False:
            action=self.choose(qs,eps)
        else:
            action=self.greedy(qs)

        # put into buffer
        self.buffer.add(self.prev_state, self.prev_action, reward, state, done)
        
        # Training V
        if len(self.buffer) > 3*self.batch_size and self.test == False: # start training when enough example availble in buffer
       
            states,actions,rewards,next_states,dones = self.buffer.sample(self.batch_size)

            #Calculating predictd values for states qs
            qs = self.q(states)
            mask = F.one_hot(actions, self.env.action_space.n)
            pred_qs = (mask * qs).sum(dim=-1)
            
            #Calculating targets : (max_next_qhat if not final  else just reward) 
            with torch.no_grad():
                max_next_qhats = self.qhat(next_states).max(dim=-1).values
                target = rewards + (1.0 - dones) * self.discount_factor* max_next_qhats
            
            loss = self.criterion(pred_qs, target)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        # Update state and action
        self.prev_state = state
        self.prev_action = action
        self.t+=1
        return action

    
    def choose(self,q_state,eps):
        if self.play_greedy:
            return greedy(q_state)
        return self.egreedy(q_state,eps)
    
    def greedy(self,opt):
        return int(torch.argmax(opt))

    def egreedy(self,opt,eps,**kwargs):
        if np.random.rand()<eps:
            return int(torch.argmax(opt))
        return self.env.action_space.sample()
    
    def save(self,outputDir):
        pass
    
    def load(self,inputDir):
        pass




if __name__ == '__main__':
   
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/Actor_Critic" + "-" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    agent_DQN = AgentDQN(env,config,episode_count)
   

    agent = agent_DQN

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
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done, i)
            ob, reward, done, _ = env.step(action)
            j+=1

            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()