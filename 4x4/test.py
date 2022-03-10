from stable_baselines3 import A2C, DQN
from environment.base_env import CurriculumEnv

def main():

    env = CurriculumEnv(map_type="test", visualization=False)
    model = DQN.load("./weights/dqn_random_2", env = env)
    total_rew = 0
    for i in range(30):
        curr_rew = 0
        done = False
        obs = env.reset()

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            total_rew += rewards
            curr_rew += rewards
        print(f"map {i+1} score: ", curr_rew)
    print(total_rew)
    
if __name__ == '__main__':
    main()
