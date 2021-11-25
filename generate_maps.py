from itertools import product
import numpy as np
import _pickle as cPickle

#Map size
map_size = 4
#Generate all possible map configuration
#0:Ground, 1:Prize
harita = product(range(2), repeat=map_size**2)

#Map list to save each map
map_list = []

f = open('./all_possible_maps.pkl', 'wb')
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
