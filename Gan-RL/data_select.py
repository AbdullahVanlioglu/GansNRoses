import pickle
import _pickle as cPickle
import numpy as np
def get_data(index):
    with open('library_maps.pkl', 'rb') as f:
        map_dataset = pickle.load(f)
    return map_dataset[index][0].copy() 



indexs=[0,1,3,5,6]
test_maps=[]
for index in indexs:
    cur_map=get_data(index)
    test_maps.append(cur_map)


f = open('nice_maps.pkl', 'wb')


pickler = cPickle.Pickler(f)

pickler.dump(np.array(test_maps))

f.close()