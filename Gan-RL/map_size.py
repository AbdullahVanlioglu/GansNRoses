import pickle 


with open('all_possible_maps.pkl', 'rb') as f:
    map_dataset = pickle.load(f) 


print(len(map_dataset))