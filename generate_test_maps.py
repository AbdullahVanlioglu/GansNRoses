import numpy as np
import _pickle as cPickle


test_maps=[]
max_reward_list=[]

map_size=4

#1
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0]])
max_reward1=-3
test_maps.append(map1)
max_reward_list.append(max_reward1)

#2
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,1]])
max_reward1=-3
test_maps.append(map1)
max_reward_list.append(max_reward1)


#3
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,0],
                      [1,0,0,0]])
max_reward1=-3
test_maps.append(map1)
max_reward_list.append(max_reward1)

#4
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,1,0,0],
                      [0,0,0,0],
                      [0,0,0,0]])
max_reward1=-1
test_maps.append(map1)
max_reward_list.append(max_reward1)

#5
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1],
                      [0,0,0,0]])
max_reward1=-3
test_maps.append(map1)
max_reward_list.append(max_reward1)

#6
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,0,1],
                      [0,0,0,0],
                      [0,1,0,0]])
max_reward1=-5
test_maps.append(map1)
max_reward_list.append(max_reward1)

#7
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0],
                      [1,0,0,0]])
max_reward1=-6
test_maps.append(map1)
max_reward_list.append(max_reward1)


#8
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,1,0],
                      [0,0,0,0],
                      [0,0,0,0],
                      [0,0,1,0]])
max_reward1=-5
test_maps.append(map1)
max_reward_list.append(max_reward1)

#9
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,1,0,0],
                      [0,0,0,0],
                      [0,0,1,0]])
max_reward1=-6
test_maps.append(map1)
max_reward_list.append(max_reward1)

#10
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0],
                      [1,0,0,1]])
max_reward1=-9
test_maps.append(map1)
max_reward_list.append(max_reward1)

#11
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,0,1],
                      [0,0,1,0],
                      [0,1,0,0]])
max_reward1=-5
test_maps.append(map1)
max_reward_list.append(max_reward1)

#12
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,0,0],
                      [0,0,0,0],
                      [0,0,0,1],
                      [0,1,0,0]])
max_reward1=-5
test_maps.append(map1)
max_reward_list.append(max_reward1)

#13
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,1,0,0],
                      [0,0,0,0],
                      [1,0,0,1]])
max_reward1=-9
test_maps.append(map1)
max_reward_list.append(max_reward1)

#14
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,1,0],
                      [1,0,0,0],
                      [0,0,0,1],
                      [0,1,0,0]])
max_reward1=-7
test_maps.append(map1)
max_reward_list.append(max_reward1)

#15
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,0,1,0],
                      [0,0,1,1],
                      [1,0,0,0]])
max_reward1=-6
test_maps.append(map1)
max_reward_list.append(max_reward1)


#16
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,1,0,0],
                      [0,0,0,1],
                      [1,0,1,0]])
max_reward1=-8
test_maps.append(map1)
max_reward_list.append(max_reward1)

#17
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,0,0,0],
                      [0,1,0,1],
                      [1,0,1,0]])
max_reward1=-8
test_maps.append(map1)
max_reward_list.append(max_reward1)

#18
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,1,0],
                      [0,1,0,1],
                      [0,0,1,0],
                      [1,0,0,1]])
max_reward1=-8
test_maps.append(map1)
max_reward_list.append(max_reward1)

#19
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,1,0,1],
                      [0,0,0,1],
                      [0,1,1,1]])
max_reward1=-7
test_maps.append(map1)
max_reward_list.append(max_reward1)

#20
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [0,1,1,0],
                      [1,0,1,0],
                      [0,1,0,1]])
max_reward1=-8
test_maps.append(map1)
max_reward_list.append(max_reward1)

#21
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,0,1],
                      [1,1,0,1],
                      [0,1,0,0],
                      [1,0,0,1]])
max_reward1=-11
test_maps.append(map1)
max_reward_list.append(max_reward1)

#22
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,1,1,0],
                      [0,1,1,1],
                      [0,1,1,1]])
max_reward1=-8
test_maps.append(map1)
max_reward_list.append(max_reward1)

#23
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,1],
                      [1,1,0,1],
                      [0,1,1,0],
                      [1,0,1,1]])
max_reward1=-10
test_maps.append(map1)
max_reward_list.append(max_reward1)

#24
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,1,0],
                      [1,1,0,1],
                      [0,1,1,1],
                      [1,0,1,1]])
max_reward1=-10
test_maps.append(map1)
max_reward_list.append(max_reward1)

#25
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,0,1],
                      [1,1,1,0],
                      [0,1,1,1],
                      [1,0,1,1]])
max_reward1=-12
test_maps.append(map1)
max_reward_list.append(max_reward1)

#26
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,1,0],
                      [1,1,1,1],
                      [1,1,1,1],
                      [1,0,1,1]])
max_reward1=-12
test_maps.append(map1)
max_reward_list.append(max_reward1)

#27
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,0,1],
                      [1,1,1,1],
                      [0,1,1,1],
                      [1,1,1,1]])
max_reward1=-13
test_maps.append(map1)
max_reward_list.append(max_reward1)

#28
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,0,1],
                      [1,1,0,0],
                      [0,0,1,0],
                      [1,0,0,1]])
max_reward1=-11
test_maps.append(map1)
max_reward_list.append(max_reward1)

#29
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,1,1,1],
                      [1,1,1,1],
                      [1,1,1,1],
                      [1,1,1,1]])
max_reward1=-15
test_maps.append(map1)
max_reward_list.append(max_reward1)

#30
map1=np.zeros((2,map_size,map_size))
map1[1,:,:]=np.array([[0,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
max_reward1=-3
test_maps.append(map1)
max_reward_list.append(max_reward1)


max_reward=np.sum(max_reward_list)
print("Max_Reward:{}".format(max_reward+161))
print("Min_Reward:{}".format(30*-100))
"""f = open('./all_test_maps.pkl', 'wb')


pickler = cPickle.Pickler(f)

pickler.dump(np.array(test_maps))

f.close()
"""
