from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from environment.base_env import ExperimentalTestEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np

def main():
    
    allRewards = []
    for i in range(1):
        env = ExperimentalTestEnv(map_type="test", visualization=False, agent_idx=i)
        model = A2C.load("./weights/a2c_experimental_{}".format(i+1), env = env)

        total_rew = 0
    
        for idx in range(30):
            done = False
            obs = env.reset(idx)

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                total_rew += rewards

        print("Agent {} Reward:{}".format(i,total_rew))
        allRewards.append(total_rew)
    np.savetxt("gan_experimental_rewards.txt", allRewards, delimiter=",")
    
    allRewards = []
    for i in range(1):
        env = ExperimentalTestEnv(map_type="test", visualization=False, agent_idx=i)
        model = A2C.load("./weights/a2c_random_{}".format(i+1), env = env)

        total_rew = 0
    
        for idx in range(30):
            done = False
            obs = env.reset(idx)

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                total_rew += rewards

        print("Agent {} Reward:{}".format(i,total_rew))
        allRewards.append(total_rew)
    np.savetxt("gan_random_rewards.txt", allRewards, delimiter=",")

if __name__ == '__main__':
    main()
