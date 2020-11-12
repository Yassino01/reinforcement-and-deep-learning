import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
#from PIL import Image as Image
import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete
from itertools import groupby
from operator import itemgetter
from contextlib import closing
from six import StringIO, b

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red

COLORS = {0:[0,0,0], 1:[128,128,128], \
          2:[0,0,255], 3:[0,255,0], \
          5:[255,0,0], 6:[255,0,255], \
          4:[255,255,0]}
COLORSDIC = {0: "white", 1:"gray", 2:"blue",3:"green",4:"cyan",5:"red",6:"magenta"} 



def insideBox(coords, xmax,ymax):
    return coords[0] >= 0 and coords[1] >=0 and coords[0]< xmax and coords[1] < ymax
def str_color(s):
    return utils.colorize(" ",COLORSDIC[int(s)],highlight=True)
   
class GridworldEnv(discrete.DiscreteEnv):
    """ Environnement de Gridworld 2D avec le codage suivant : 
            0: case vide
            1: mur
            2: joueur
            3: sortie
            4: objet a ramasser
            5: piege mortel
            6: piege non mortel
        actions : 
            0: South
            1: North
            2: West
            3: East
    """

    metadata = {
        'render.modes': ['human', 'ansi','rgb_array'], #, 'state_pixels'],
        'video.frames_per_second': 1
    }
    num_env = 0
    actions={0:[1,0],1:[-1,0],2:[0,-1],3:[0,1]}
    NBMAXSTEPS = 1000
    ALEA_ACTION = 0.2

    def __init__(self,plan='gridworldPlans/plan0.txt',rewards={0:0,3:1,4:1,5:-1,6:-1}):
        """
        Initialise GridWorld
        
        :param plan: un plan de gridworldPlans
        :param rewards: un dictionnaire de rewards pour chaque case 0 à 6 (sauf 1 et 2)
        """
        self.setPlan(plan,rewards)
    
    def setPlan(self,plan,rewards):
        """ Fixe un scénario
            
        :param plan: un plan de gridworldPlans
        :param rewards: un dictionnaire de rewards pour chaque case 0 à 6 (sauf 1 et 2)
        """
        self._make(plan,rewards)
    
    @property
    def nA(self):
        return len(self.actions)

    @staticmethod
    def state2str(state):
        return str(state.tolist())
    
    @staticmethod
    def str2state(s):
        return np.array(eval(s))

    def _make(self,plan,rewards):
        self.states = dict()
        self.id2states = dict()
        self.rewards, self.nbMaxSteps, self.alea_action = rewards, self.NBMAXSTEPS, self.ALEA_ACTION
        self.action_space = spaces.Discrete(self.nA)
        grid_map_path=plan
        if not os.path.exists(plan):
            grid_map_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), plan)
        
        self.obs_shape = [128, 128, 3]
        self.start_grid_map = self._read_grid_map(grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.startPos=self._get_agent_pos(self.current_grid_map)
        self.reset()
        self.P=None
        self.nS=None
        self.observation_space = None
        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env
    
    def reset(self):
        """ Reinitialise l'environnement
        :returns: l'id de l'observation
        """
        self.currentPos = copy.deepcopy(self.startPos)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.nbSteps=0
        self.lastaction = None
        return self.get_state_id(self.current_grid_map)

    def step(self, action):
        """ Retourne l'observation apres avoir effectuer l'action
        :param action: action a effectuer        
        :returns: un 4-uplet nouvel etat, reward, done, info (dico vide)
        """
        self.nbSteps += 1
        c = self.start_grid_map[self.currentPos[0],self.currentPos[1]]
        if c==3 or c==5 : ## Done == True au coup d'avant
            return self.get_state_id(self.current_grid_map),0,True,{}
        action = int(action)
        p = np.random.rand()
        if p<self.alea_action:
            p = np.random.rand()
            if action == 0 or action == 1:
                action = 2 if p < 0.5 else 3
            else:
                action = 0 if p < 0.5 else 1

        npos = (self.currentPos[0] + self.actions[action][0], self.currentPos[1] + self.actions[action][1])
        rr=-1*(self.nbSteps>self.nbMaxSteps)
        if not insideBox(npos,self.current_grid_map.shape[0], self.current_grid_map.shape[1]) or self.current_grid_map[npos[0],npos[1]]==1:
            return (self.get_state_id(self.current_grid_map), self.rewards[0]+rr, self.nbSteps>self.nbMaxSteps, {})

        c=self.current_grid_map[npos]
        r = self.rewards[c]+rr
        done = (c == 3 or c == 5 or self.nbSteps>self.nbMaxSteps)
        self.current_grid_map[self.currentPos[0],self.currentPos[1]] = 0
        self.current_grid_map[npos[0],npos[1]] = 2
        self.currentPos = npos
        self.lastaction = action
        return (self.get_state_id(self.current_grid_map),r,done,{})

    def get_state_id(self,state):
        """ Retourne l'id correspondant a l'etat. Si l'état n'existe pas, il est ajouté au dictionnaire.
        :params state: l'etat à décoder
        :returns: l'identifiant de l'etat
        """
        strstate = self.state2str(state)
        if not strstate in self.states:
            self.states[strstate] = len(self.states)
            self.id2states[self.states[strstate]] = state
        return self.states[strstate]
    
    def get_state(self,id):
        """ Retourne l'etat associé à l'id. Si l'identifiant n'existe pas, None est renvoyé.
        :params id: l'identifiant de l'état
        :returns: l'id associé à l'état
        """
        if not id in self.id2states:
            return None
        return self.id2states[id]

    def getMDP(self):
        if self.P is None:
            self.P={}
            self.states=dict()
            self._known  = set()
            self._getMDP(self.start_grid_map, self.startPos)
            self.nS = len(self.states)
            self.observation_space = spaces.Discrete(self.nS)
        return (self.id2states,self.P)

   
    def _getMDP(self,gridmap,state):
        cur = self.get_state_id(gridmap)
        if cur in self._known: return
        self.P[cur]={0:[],1:[],2:[],3:[]}
        self._known.add(cur)
        for k,v in self.actions.items():
            self._exploreDir(gridmap,state,v,k)
        
    def _exploreDir(self,gridmap,state,dir,id_dir):
        cur=self.states[self.state2str(gridmap)]
        gridmap = copy.deepcopy(gridmap)
        succs=self.P[cur]
        nstate = copy.deepcopy(state)
        nstate[0]+=dir[0]
        nstate[1] += dir[1]

        if insideBox(nstate,gridmap.shape[0],gridmap.shape[1]) and gridmap[nstate[0],nstate[1]]!=1:
                oldc=gridmap[nstate[0],nstate[1]]
                gridmap[state[0],state[1]] = 0
                gridmap[nstate[0],nstate[1]] = 2
                done = (oldc == 3 or oldc == 5)
                ns = self.get_state_id(gridmap)
                if not done:
                    self._getMDP(gridmap,nstate)
                r = self.rewards[oldc]
                succs[id_dir].append((0.8, ns, r,done))
                if id_dir<2:
                    succs[2].append((0.1, ns, r, done))
                    succs[3].append((0.1, ns, r, done))
                else:
                    succs[0].append((0.1,ns,r,done))
                    succs[1].append((0.1,ns,r,done))
        else:
            succs[id_dir].append((0.8,cur,self.rewards[0],False))
            if id_dir<2:
                succs[2].append((0.1, cur, self.rewards[0], False))
                succs[3].append((0.1, cur, self.rewards[0], False))
            else:
                succs[0].append((0.1, cur, self.rewards[0], False))
                succs[1].append((0.1, cur, self.rewards[0], False))

    def _get_agent_pos(self, grid_map):
        state = list(map(
                 lambda x:x[0] if len(x) > 0 else None,
                 np.where(grid_map == 2)
             ))
        return state

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        grid_map_array = np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grid_map
            ))
        )
        return grid_map_array



    def _gridmap_to_img(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.uint8)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[grid_map[i, j]])
        return observation

    def render(self, pause=0.00001, mode='rgb_array', close=False):
        if mode =='human' or mode =='ansi':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            desc = self.current_grid_map.tolist()
            desc = [[str_color(c) for c in line] for line in desc]
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(["South","North","West","East"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")
            if mode != 'human':
                with closing(outfile):
                    return outfile.getvalue()            
            return
        img = self._gridmap_to_img(self.current_grid_map)
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        if pause>0:
            plt.pause(pause)
        return img

    def _close_env(self):
        plt.close(self.this_fig_num)
        return

    def close(self):
        super(GridworldEnv,self).close()
        self._close_env()
    def changeState(self,gridmap):
        self.current_grid_map=gridmap
        self.currentPos=self._get_agent_pos(gridmap)
