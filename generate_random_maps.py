import numpy as np
import _pickle as cPickle
import random


random_maps=[]


map_size=4

map1=np.zeros((2,map_size,map_size))

loc_arr=[]

for i in range(4):
    for j in range(4):
        loc_arr.append([i,j])

loc_arr=np.array(loc_arr)

total_map_num=1000

for i in range(total_map_num):
    reward_num=random.randint(1,16)
    map1=np.zeros((2,map_size,map_size))
    random_reward_index=random.sample(range(0, len(loc_arr)), reward_num)
    
    for j in range(reward_num):
        x=loc_arr[random_reward_index[j],0]
        y=loc_arr[random_reward_index[j],1]
        map1[1,x,y]=1
    random_maps.append(map1)

f = open('./random_{}_maps.pkl'.format(total_map_num), 'wb')


pickler = cPickle.Pickler(f)

pickler.dump(np.array(random_maps))

f.close()

