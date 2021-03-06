import torch
import gym
import torch.nn as nn

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.env import QuadrotorFormation
from stable_baselines3.common.vec_env import SubprocVecEnv


#max_steps = 3e5
max_steps = 1e5

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
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, int(n_flatten/2)), 
            nn.ReLU(),
            nn.Linear(int(n_flatten/2), features_dim), 
            nn.ReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),)


# Train with GAN Generated Maps
def main():
    #vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="gan", visualization=False), n_envs=18, vec_env_cls=SubprocVecEnv)
    #model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, exploration_fraction = 0.8, verbose=1, tensorboard_log="./dqn_tensorboard/")
    #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/gan")

    #model.learn(total_timesteps=max_steps)
    #model.save("./weights/a2c_gan_curr2")

    for i in range(1):

        vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="train", visualization=False, data_percent=(i+1)*0.02), n_envs=1, vec_env_cls=SubprocVecEnv)
        model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, verbose=1, learning_rate = 0.0003, exploration_fraction=0.65, tensorboard_log="./dqn_tensorboard/")
        #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/random")

        model.learn(total_timesteps=max_steps)
        model.save(f"./weights/dqn_{(i+1)*0.02}")


def single_map_train():

    for i in range(4):
        vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="train", visualization=False),  n_envs=1, vec_env_cls=SubprocVecEnv)
        model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, verbose=1, learning_rate = 0.0003, exploration_fraction=0.65, tensorboard_log="./dqn_tensorboard/")
        #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, ent_coef = 0.5, verbose=1, tensorboard_log="./a2c_tensorboard/random")

        model.learn(total_timesteps=max_steps)
        model.save(f"./weights/single_map_dqn_{i+1}")

if __name__ == '__main__':
    #main()
    single_map_train()

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render()