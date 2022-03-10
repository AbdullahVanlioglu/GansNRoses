import torch
import gym
import torch.nn as nn

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.base_env import CurriculumEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

max_steps = 2e5

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
            nn.Conv2d(n_input_channels, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten()
            )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, int(n_flatten/2)), 
            nn.ReLU(),
            nn.Linear(int(n_flatten/2), int(n_flatten/4)), 
            nn.ReLU(),
            nn.Linear(int(n_flatten/4), features_dim), 
            nn.ReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),)


def train_agent(map_type, agent_type, tensorboard):
    print("Agent Training")
    vecenv = make_vec_env(lambda: CurriculumEnv(map_type=f"{map_type}", visualization=False),  n_envs=1, vec_env_cls=SubprocVecEnv)
    model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, verbose=0, learning_rate = 0.0003, exploration_fraction=0.65, tensorboard_log=f"./dqn_tensorboard/{tensorboard}")
    #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/trap")

    model.learn(total_timesteps=max_steps)
    model.save(f"./weights/{agent_type}")
