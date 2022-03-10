import numpy as np
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv
from environment.env import QuadrotorFormation

env = QuadrotorFormation(map_type="train") 

# num_envs = 8
# env_name = "CartPole-v0"

# def make_env():
#     def _thunk():
#         env = gym.make(env_name)
#         return env
#     return _thunk

# plt.ion()
# envs = [make_env() for i in range(num_envs)]
# envs = SubprocVecEnv(envs) # 8 env

# env = gym.make(env_name) # a single env

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        num_channel = obs_shape[0] # 2
        img_size = obs_shape[1] # 4
        
        self.critic = nn.Sequential(
            nn.Conv2d(num_channel, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(hidden_size*2*img_size*img_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Conv2d(num_channel, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(hidden_size*2*img_size*img_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, num_outputs),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value


def test_env():
    state = env.reset()

    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def plot(frame_idx, rewards):
    plt.plot(rewards,'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)


obs_shape  = env.observation_space.shape
print(obs_shape)
num_outputs = env.action_space.n

#Hyper params:
hidden_size = 64
lr          = 7e-4
num_steps   = 100

model = ActorCritic(obs_shape, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

total_maps   = 20000
total_frames = 2**16-1
frame_idx    = 0
test_rewards = []

test_index = 0
iteration = 0

for map_idx in range(total_maps):

    state = env.reset()

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    # rollout trajectory
    while True:
        iteration += 1
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).to(device))
        masks.append(torch.FloatTensor([1 - done]).to(device))
        
        state = next_state
        
        test_index += 1

        if test_index >= 6553:
            test_index = 0
            test_rewards.append(test_env())
            plot(frame_idx, test_rewards)

        if done:
            break
        
        if iteration % 5 == 0:
                  
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = model(next_state)
            returns = compute_returns(next_value, rewards, masks)
            
            log_probs = torch.cat(log_probs)
            returns   = torch.cat(returns).detach()
            values    = torch.cat(values)

            advantage = returns - values

            actor_loss  = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
