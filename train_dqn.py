from pickle import FALSE
import numpy as np
from agent.dqn import DQN,BATCH_SIZE
from environment.env import QuadrotorFormation


data_percents=[10,20,30,40,50,60,70,80,90,100]

agent=DQN()

for data_percent in data_percents:
    env=QuadrotorFormation(map_type="random", visualization=True, data_percent=data_percent)
    agent=DQN()
    episode=int((data_percent/10)*200)#1e6
    print("Training for {} data percent started".format(data_percent))
    for epp in range(episode):
        state=env.reset()
        done=FALSE
        while done:
            action=agent.choose_action(state)
            next_state,reward,done,info=env.step(action)
            agent.store_transition(state,action,reward,next_state)
            if len(agent.memory)>BATCH_SIZE:
                agent.learn()
    agent.save_model("dqn-model-1-{}".format(data_percent))

        









