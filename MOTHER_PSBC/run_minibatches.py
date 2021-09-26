#import  matplotlib.pyplot as plt
import scipy.sparse as sc
import itertools as it
import pandas as pd
import numpy as np
import warnings
import shutil 
import copy
import glob
import sys
import os
import time
import tensorflow as tf
from tensorflow import keras
try: ## In order to open and save dictionaries, "dt": self.dt, "kind" : "Viscosity"
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import multiprocess as mp
warnings.filterwarnings (action = "ignore", message = "internal issue")

############################################################################
from  tfversion_binary_phase_separation import *
from  tf_PSBC_extra_libs_for_training_and_grid_search import *

############################################################################
print ("\nGrid Search")
### RETRIEVE PARAMETERS

### READ VARIABLES AND RETRIEVE TRAINING DATA (BOTH VARIABLES COMBINED)
Neumann_str = sys.argv [1]
Nt_str = sys.argv [2]
index = int (sys.argv [3])
with_PCA = sys.argv [4]  == "True"   ### Funny thing:if you do "bool("False")" you get True 
cpu = int (sys.argv [5])
parallel = sys.argv [6] == "True"
grid_type = sys.argv [7]

filename = Neumann_str+"_"+str (Nt_str)+"_"+grid_type+".p"

with open ("../../Grids/"+filename, 'rb') as pickled_dic:
    grid_range  = pickle.load (pickled_dic)

with open ("../../Grids/digits_index.p", 'rb') as pickled_dic:
    grid_indexes  = pickle.load (pickled_dic)

cv = grid_range ["cv"]
print (cv)
Neumann = grid_range ["Neumann"]
EPOCHS = grid_range ["EPOCHS"]
patience = grid_range ["patience"]
Nt = grid_range ["Nt"]

print ("Test Nt")
assert (Nt == int(Nt_str))
test_Neumann = Neumann_str == "True"
print ("Test Neumann")
assert (Neumann == test_Neumann)

train_dt_U = grid_range ["train_dt_U"]
train_dt_P = grid_range ["train_dt_P"]

###-----------------------------------------------------------------------------
variable_0, variable_1 = grid_indexes [index]  # Recall that 
                                               # variable_0 < variable_1

print ("Variables given: \n\tvariable_0 :",\
    variable_0,"\n\tvariable_1 :", variable_1,\
        "\n\twith_PCA :", with_PCA,\
            "\n\tcpu :", cpu,
            "\n\t parallel processing : ", parallel,\
                "\ngrid_type:", grid_type)
###-----------------------------------------------------------------------------

S = select_split_pickle (level= 2)
## retrieve non-shuffled data
X_all, Y_all, _ = S.select_variables_from_pickle (variable_0, variable_1)

###  RETRIEVE TRAIN-TEST INDEXES and 
file_name = "../../Pickled_datasets/generate_k_fold_"+ str (variable_0)+ "_" + str (variable_1)+".p"

with open (file_name, 'rb') as pickled_dic:
    generate_k_fold = pickle.load (pickled_dic)

results = []        ### FOR PARALLEL CASE


##### CREATING GRIDS!!!
parameters_model = {}

Nx = 784 
if Nt == 1: 
    parameters_model ["layer_share_range"] = [1]
else: 
    parameters_model ["layer_share_range"] = [1, Nt]
parameters_model ["lr_U_range"] = np.asarray(
    [1e-3, 1e-2, 1e-1], dtype = np.float32)
parameters_model ["lr_P_range"] = np.asarray(
    [1e-3, 1e-2, 1e-1], dtype = np.float32)
parameters_model ["dt_range"] = np.asarray(
    [.2], dtype = np.float32)

if grid_type == "all":
    parameters_model ["eps_range"] = np.asarray(
        np.r_[[0], np.power (.5, np.arange (10, -2, -1))], dtype = np.float32
        )
    parameters_model ["ptt_range"] = [int (Nx/k) for k in range(1,11)]
elif grid_type == "vary_eps":
    parameters_model ["eps_range"] = np.asarray(
        np.r_[[0], np.power (.5,np.power (.5, [4,3,2,1,0])) ], dtype = np.float32
        )
    parameters_model ["ptt_range"] = [int (Nx/4)]
elif grid_type == "vary_Nt":
    parameters_model ["eps_range"] = np.asarray([0], dtype = np.float32)
    parameters_model ["ptt_range"] = [int (Nx/k) for k in [1, 2, 4, 8, 16]]

#####################################################################
### BEGIN PARALLEL PROCESSING
if parallel:
    print ("\nRUNNING THE MODEL IN PARALLEL")
    a = time.time ()
    pool = mp.Pool(cpu)
    for i in range(cv):
        ### Normalized and centralized (mean zero)
        train_index, test_index,\
            mean_train_grid, Vstar, var_0_pickled, var_1_pickled = generate_k_fold [str (i)]

        assert (variable_0 == var_0_pickled)
        assert (variable_1 == var_1_pickled)

        ### Split
        X_train_grid, Y_train_grid = X_all[train_index], Y_all[train_index]
        ### Centralization 
        #mean_train_grid = np.mean (X_train_grid, axis = 0)
        X_train_grid = X_train_grid - mean_train_grid
        X_test_grid, Y_test_grid = X_all [test_index] - mean_train_grid, Y_all [test_index]

        ### Now run grid search IN PARALLEL
        print ("\nUsing", cpu, "processors")
        # Step 3: Use loop to parallelize
        args_now = (i, X_train_grid, Y_train_grid, X_test_grid, Y_test_grid, parameters_model,\
            784, Neumann, EPOCHS , patience, Nt, train_dt_U, train_dt_P, with_PCA, Vstar)
        results.append(
            pool.apply_async(
                my_gridSearch_with_index,
                args = args_now )
                )
    # results is a list of pool.ApplyResult objects
    all_results = [r.get() for r in results]
    pool.close()
    pool.join()
    #pool.clear()
    print ("\n It took", time.time () -a, "to run the model in parallel")
else:
    print ("\nRUNNING THE MODEL IN SERIALLY")
    a = time.time ()
    for i in range(cv):
        ### Normalized and centralized (mean zero)
        train_index, test_index,\
            mean_train_grid, Vstar, var_0_pickled, var_1_pickled = generate_k_fold [str (i)]

        assert (variable_0 == var_0_pickled)
        assert (variable_1 == var_1_pickled)

        ### Split
        X_train_grid, Y_train_grid = X_all [train_index], Y_all [train_index]
        ### Centralization 
        #mean_train_grid = np.mean (X_train_grid, axis = 0)
        X_train_grid = X_train_grid - mean_train_grid
        X_test_grid, Y_test_grid = X_all [test_index] - mean_train_grid, Y_all [test_index]

        ### Now run grid search IN PARALLEL
        print ("\nUsing", cpu, "processors")
        # Step 3: Use loop to parallelize
        results.append(
            my_gridSearch_with_index (i, X_train_grid, Y_train_grid,\
            X_test_grid, Y_test_grid, parameters_model, 784, Neumann,\
            EPOCHS , patience, Nt, train_dt_U, train_dt_P,\
                with_PCA, Vstar)
        )
    # results is a list of pool.ApplyResult objects
    all_results = results
    print ("\n It took", time.time () -a, "to run the model serially")

#return results
for j, a, b in all_results:
    if  j == 0:
        Accuracies, Parameters = a, b
    else:
        Accuracies_tmp, Parameters_tmp = a, b
        assert (Parameters_tmp == Parameters)
        Parameters = Parameters_tmp
        Accuracies = np.vstack ([Accuracies, Accuracies_tmp]) 

print ("Creating Accuracies and parameter pickled file")
print ("end",with_PCA)
if with_PCA:
    print ("It's here")
    file_name = "PCA_all_grid_search_results_"\
        + str (variable_0)+ "_" + str (variable_1)+".p"
    with open (file_name, 'wb') as save:
        pickle.dump ( (Accuracies, Parameters), save, protocol = pickle.HIGHEST_PROTOCOL)        
    print ("Statistics pickled to ", file_name)
else:
    file_name = "Normal_all_grid_search_results"\
        + str (variable_0)+ "_" + str (variable_1)+".p"
    with open (file_name, 'wb') as save:
        pickle.dump ( (Accuracies, Parameters), save, protocol = pickle.HIGHEST_PROTOCOL)        
    print ("Statistics pickled to ", file_name)
        
