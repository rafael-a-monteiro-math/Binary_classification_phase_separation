import itertools as it
import numpy as np
import pickle
import os
import sys


grid_indexes =  [(variable_0, variable_1) for variable_0, variable_1 in it.combinations (np.arange (10), 2)]
    
print ("Creating grid for indexes")


try: os.mkdir ("Grids")
except: pass

with open ("Grids/digits_index.p", 'wb') as save:
    pickle.dump (grid_indexes, save, protocol = pickle.HIGHEST_PROTOCOL)
