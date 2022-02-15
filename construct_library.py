import numpy as np
import pickle

class Library():
    #Initialize library
    def __init__(self,library_size=180):
        self.library_size = library_size
        self.train_library = []
        #Load test maps and add it to test_library

    def add(self, map):
        self.train_library.append(np.array(map).reshape((1,2,4,4)))

        print("Library size increased:", len(self.train_library))
        #Save training library maps
        #self.save_maps()
    
    def get(self):
        rindex = np.random.randint(0,len(self.train_library))
        return self.train_library[rindex]
    
    def save_maps(self):
        with open(f'library/library_map.pkl', 'wb') as f:
            pickle.dump(self.train_library, f)

