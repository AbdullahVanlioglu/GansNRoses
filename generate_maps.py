from itertools import product
import numpy as np
import pickle as cPickle
from config import get_arguments, post_config
import random

opt = get_arguments().parse_args()
opt = post_config(opt)

#Map size
map_size = opt.full_map_size
#Generate all possible map configuration
#0:Ground, 1:Prize
harita = list(product(range(2), repeat=map_size**2))
random.shuffle(harita)
#Map list to save each map
map_list = []

f = open('./library/all_possible_maps.pkl', 'wb')
pickler = cPickle.Pickler(f)

#loop over harita to convert proper form
for h in harita:
    #Reset current map
    matrix_map = np.zeros((2,map_size, map_size))
    for i in range(map_size**2):
        if h[i] == 0:
            matrix_map[0, i//map_size, i%map_size] = 1
        elif h[i] == 1:
            matrix_map[1, i//map_size, i%map_size] = 1
    
    map_list.append(matrix_map)

pickler.dump(map_list)

f.close()
