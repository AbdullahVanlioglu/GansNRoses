import pickle
import torch
import numpy as np
import torch.nn as nn
import gym

from stable_baselines3 import A2C, DQN,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.env import QuadrotorFormation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


max_steps = 1e6

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        #print(n_flatten)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),)


# Train with GAN Generated Maps
def main():
    #vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="gan", visualization=False), n_envs=18, vec_env_cls=SubprocVecEnv)
    #model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, exploration_fraction = 0.8, verbose=1, tensorboard_log="./dqn_tensorboard/")
    #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/gan")

    #model.learn(total_timesteps=max_steps)
    #model.save("./weights/a2c_gan_curr2")

    # Train with GAN Random Maps
    vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="train", visualization=False,data_percent=10), n_envs=1, vec_env_cls=SubprocVecEnv)
    model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, exploration_fraction = 0.8, verbose=1, tensorboard_log="./dqn_tensorboard/")
    #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/random")

    model.learn(total_timesteps=max_steps)
    model.save("./weights/DQN-10-random")
    
if __name__ == '__main__':
    main()

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render()