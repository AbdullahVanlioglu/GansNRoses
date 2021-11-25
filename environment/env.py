from numpy.core.arrayprint import dtype_short_repr
import gym
import pickle
import os
import time
import sys
import numpy as np


from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from os import path
from environment.quadrotor_dynamics import Drone
from numpy.random import uniform
from time import sleep
from PIL import Image
import random

class QuadrotorFormation(gym.Env):

    def __init__(self, map_type="random", visualization=False,data_percent=10):
        super(QuadrotorFormation, self).__init__()

        self.seed()
        self.n_action = 8
        self.visualization = visualization

        self.agent = None
        self.viewer = None
        self.rewPos_list = []
        self.map_type = map_type
        
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(2, 4, 4), dtype=np.uint8)

        self.x_lim = 3
        self.y_lim = 3
        self.iteration = 0
        self.map_index = 0
        self.map_iter = 0
        self.reward = 0

        self.index_set=random.sample(range(1, 2**16-1), int((2**16-2)*data_percent/100))

        self.reward_map = np.zeros((self.y_lim, self.x_lim))
        #self.path_map = np.zeros((self.y_lim, self.x_lim))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.reward = -1
        self.iteration += 1

        if self.visualization:
            self.render()

        self.update_agent_pos(action)

        if int(self.reward_map[int(self.agent.y), int(self.agent.x)]) == 1:
            self.reward_map[int(self.agent.y),int( self.agent.x)] = 0
            #self.reward += 8

        self.reward_wall_num()
        state = self.get_observation()

        if self.visualization:
            self.render()

        if np.all(self.reward_map == 0) or self.iteration >= 100:
            done = True
            self.map_iter += 1
            self.close()

        info = {"map_index": self.map_index}
        
        return state, self.reward, done, info

    def get_observation(self):

        state = np.zeros((1,2,4,4))

        #state[:,:,0] = self.path_map*255.0
        state[:,0,:,:] = self.reward_map
        state[:,1,:,:] = self.agent.state
        
        return np.array(state, dtype=np.uint8)


    def generate_agent_position(self, agentY, agentX):
        state0 = np.zeros((4,4))

        state0[agentY, agentX] = 1.0

        self.agent = Drone(state0)
        self.agent.x = agentX
        self.agent.y = agentY

    def get_init_map(self, index):

        if self.map_type == "train":
            with open('/home/avsp/Masaüstü/GansNRoses/all_possible_maps.pkl', 'rb') as f:
                map_dataset = pickle.load(f) 

        elif self.map_type == "test":
            with open('/home/avsp/Masaüstü/GansNRoses/all_test_maps.pkl', 'rb') as f:
                map_dataset = pickle.load(f)

        return map_dataset[index].copy()

    def reward_wall_num(self):
        self.rewPos_list = []

        reward_row, reward_col = np.where(self.reward_map == 1)

        for x in zip(reward_row, reward_col):
            self.rewPos_list.append(x)

    def reset(self):

        self.iteration = 0
        self.reward = 0

<<<<<<< HEAD
        self.map_index = np.random.randint(low=1, high=30)
        init_map = self.get_init_map(self.map_index)
=======
        self.map_index = np.random.randint(low=0, high=len(self.index_set)-1)
        init_map = self.get_init_map(self.index_set[self.map_index])
>>>>>>> 7e727519872f29f1f6be012997d58d1533b824cf
        #if self.map_iter % 60 == 0 and self.map_iter !=0:
        #self.map_index += 1
        #self.map_index %= 600

        agent_initX = 0
        agent_initY = 0
        #self.path_map = init_map[0]
        self.reward_map = init_map[1]

        self.reward_wall_num()

        self.generate_agent_position(agent_initY, agent_initX)

        if int(self.reward_map[int(self.agent.y), int(self.agent.x)]) == 1:
            self.reward_map[int(self.agent.y),int( self.agent.x)] = 0

        state = self.get_observation()

        return state

    def update_agent_pos(self, discrete_action):

        self.agent.state = np.zeros((4,4))

        if discrete_action == 0: # action=0, x += 1.0
            self.agent.x += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)

        elif discrete_action == 1: # action=1, x -= 1.0
            self.agent.x -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)

        elif discrete_action == 2: # action=2, y += 1.0
            self.agent.y += 1.0
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 3: # action=3, y -= 1.0
            self.agent.y -= 1.0
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 4: # action=4, x,y += 1.0
            self.agent.x += 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 5: # action=5, x,y -= 1.0
            self.agent.x -= 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 6: # action=6, x += 1.0 ,y -= 1.0
            self.agent.x += 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 7: # action=7, x -= 1.0 ,y += 1.0
            self.agent.x -= 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        else:
            print ("Invalid discrete action!")

        self.agent.state[int(self.agent.y),int(self.agent.x)] = 1.0
        # time.sleep(1)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(400, 400)
            self.viewer.set_bounds(0, self.x_lim, 0, self.y_lim)
            # fname = path.join(path.dirname(__file__), "sprites/drone.png")
            mapsheet = Image.open(os.path.join(path.dirname(__file__), 'sprites/mapsheet.png'))

            drone_path = os.path.join(path.dirname(__file__), 'sprites/drone.png')
            reward_path = os.path.join(path.dirname(__file__), 'sprites/reward.png')
            road_path = os.path.join(path.dirname(__file__), 'sprites/path.png')

            sprite_dict = dict()
            sprite_dict['D'] = mapsheet.crop((4*16, 0, 5*16, 1*16))
            sprite_dict['X'] = mapsheet.crop((7*16, 1*16, 8*16, 2*16))
            sprite_dict['O'] = mapsheet.crop((2*16, 0, 3*16, 1*16))
            sprite_dict['-'] = mapsheet.crop((2*16, 5*16, 3*16, 6*16))

            sprite_dict['D'].save(drone_path)
            sprite_dict['X'].save(reward_path)
            sprite_dict['-'].save(road_path)

            self.drone_transforms = []
            self.drones = []

            self.reward_transform = []
            self.render_rew = []

            for i in range(1):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(drone_path, 0.2, 0.2))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(len(self.rewPos_list)):
                self.reward_transform.append(rendering.Transform())
                self.render_rew.append(rendering.Image(reward_path, 0.2, 0.2))
                self.render_rew[i].add_attr(self.reward_transform[i])
        
        for i in range(1):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(self.agent.x, self.y_lim-self.agent.y) 
        
        for i in range(len(self.rewPos_list)):
            if not len(self.rewPos_list)==0:
                self.viewer.add_onetime(self.render_rew[i])
                self.reward_transform[i].set_translation(self.rewPos_list[i][1], self.y_lim-self.rewPos_list[i][0])
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
