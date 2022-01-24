from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from environment.singleAgentTestEnv import TestQuadrotorFormation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np

def main():

    #vecenv = make_vec_env(lambda: TestQuadrotorFormation(map_type="test", visualization=True), n_envs=1, vec_env_cls=SubprocVecEnv)
    env = TestQuadrotorFormation(map_type="test", visualization=False)
    allRewards = []
    for i in range(10):
        model = DQN.load("./weights/dqn_experimantal_gan_0_{}".format(i), env = env)
        #model = A2C.load("./weights/a2c_random_curr2", env = env)

        total_rew = 0
    
        for idx in range(30):

            done = False
            obs = env.reset(idx)

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                total_rew += rewards
                #input("Test")

            #time.sleep(0.1)
        print("Agent {} Reward:{}".format(i,total_rew))
        allRewards.append(total_rew)
    np.savetxt("gan_experimental_rewards.txt", allRewards, delimiter=",")
if __name__ == '__main__':
    main()
