from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from environment.singleAgentTestEnv import TestQuadrotorFormation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
import time

def main():
    all_rewards=[]
    #vecenv = make_vec_env(lambda: TestQuadrotorFormation(map_type="test", visualization=True), n_envs=1, vec_env_cls=SubprocVecEnv)
    env = TestQuadrotorFormation(map_type="test", visualization=False)
    solved_count=np.zeros((1,5))
    for k in range(1):
        epp_reward=[]
        print("Episode {} is running ...".format(k))
        for i in range(5):
            model = DQN.load("./weights/dqn_gan5_{}_{}".format(k,i), env = env)
            #model = A2C.load("./weights/a2c_random_curr2", env = env)
            solved_sum=0
            for idx in range(1000):
                total_rew = 0
                done = False
                obs = env.reset(idx)

                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, solved = env.step(action)
                    total_rew += rewards
                solved_sum+=solved    
                epp_reward.append(total_rew)
            solved_count[k][i]=solved_sum/1000
            all_rewards.append(epp_reward)
    np.savetxt("Performance_gan5_3.txt",solved_count)

            #time.sleep(0.1)
        #print("Agent {} Reward:{}".format(i+1,total_rew))
if __name__ == '__main__':
    main()
