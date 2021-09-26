################################################################################
#### HOW TO USE
### This program runs from folder X
###
### where X is either Neumann, Periodic, or Neumann_non_subordinate
###
### Then it does to the next layer.
###
### It is called by typing
### python** train_model.py Nt index with_PCA cpu parallel
###
### where 
###
### i) Neumann_str : bool,
###     Boundary Conditions are of Neumann type it True, otherwise they are
###     Periodic.
### ii) Nt : int,
###     number of layers.
### iii) index : int,
###    indicates a pair of distinct variables in {0,...9}.
### iv) with_PCA : bool,
###     indicates whether or not the model uses PCA.
### v) cpu : int,
###     Number of cpus used in case of parallel processing.
### vi) parallel : bool,
###     Whether the model is parallel or not.
### vii) subordinate : bool,
###     Whether the model is subordinate or not.
###
################################################################################
### WHAT IT DOES
################################################################################
###
### Trains the PSBC model.
###
### It relies on the folder structure of the datafiles, 
### for it accesses hyperparameters selected during model selection step.
### 
################################################################################

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

################################################################################
### Import libraries from another folder. Read
### https://stackoverflow.com/questions/58990164/sys-path-insert-inserts-path-to-module-but-imports-are-not-working

import sys
sys.path.insert (0, "../MOTHER_PSBC/")
from  tfversion_binary_phase_separation import *
from tf_PSBC_extra_libs_for_training_and_grid_search import *

###--------- INPUT -------------------------------------------------------------

Neumann_str = sys.argv [1]
Nt = int(sys.argv [2])

filename = "training_" + Neumann_str + "_"+str (Nt)+".p"
print ("Parameters passed to the program : ", sys.argv)
os.chdir (str (Nt))   #Change directory

### RETRIEVE PARAMETERS
with open ("../../Grids/"+filename, 'rb') as pickled_dic:
    grid_range  = pickle.load (pickled_dic)
with open ("../../Grids/digits_index.p", 'rb') as pickled_dic:
    grid_indexes  = pickle.load (pickled_dic)

# Setting up parameters for the model
cv = grid_range ["cv"]   ## In this case denotes the number of repetitions
print ("Each model will be trained", cv, "times")
Neumann = grid_range ["Neumann"]
EPOCHS = grid_range ["EPOCHS"]
patience = grid_range ["patience"]

############################################
print ("Asserting Nt")
assert (grid_range["Nt"] == Nt)
print (grid_range)
############################################

train_dt_U = grid_range ["train_dt_U"]
train_dt_P = grid_range ["train_dt_P"]
save_history = True
Nx = 784

### READ VARIABLES AND RETRIEVE TRAINING DATA (BOTH VARIABLES COMBINED)
index = int (sys.argv [3])
variable_0, variable_1 = grid_indexes [index]  # Recall that 
                                               # variable_0 < variable_1
with_PCA = sys.argv [4] == "True"    ### Funny thing:if you do "bool("False")" you get True 
cpu = int (sys.argv [5])
parallel = sys.argv [6] == "True"    
subordinate = sys.argv [7] == "True"    
###-----------------------------------------------------------------------------

print (parallel)

print ("Variables given: \n\n* variable_0 :",\
    variable_0,", variable_1 :", variable_1)
print ("\n* Number of cross valications :", cv)
print ("\n* Parallel is", parallel, ". (If parallel is True, then use ", cpu," cores.)")
print ("\n* Nx :", Nx, ", Neumann :", Neumann, ", Epochs : ", EPOCHS, ", Patience : ", patience)
print ("\n* Nt :",  Nt, ", train_dt_U :", train_dt_U, ", train_dt_P :", train_dt_P)
print ("\n* Subordinate :", subordinate, ", with_PCA :", with_PCA)

###-----------------------------------------------------------------------------

X_train, Y_train, X_test, Y_test,_ = \
    prepare_train_test_set (variable_0, variable_1, level = 2)
results = []

print ("Constructing GRIDS with best hyperparameters!!!")

try:
    retrieve_best_par = BestPararameters_ptt_card_weights_k_shared_fixed (
        Nt, variable_0, variable_1)

    parameters_model_1, parameters_model_Nt = fill_parameters_dict (
        Nt,  retrieve_best_par, weight_sharing_split = True)

    all_parameters = {**parameters_model_1, **parameters_model_Nt}
except:
    print ("\nRetrieving best parameters using pickled files\
        in the folder 'Collection'" )
    with open ("../../Statistics/Best_parameters.p", 'rb') as filename:
        Best_parameters = pickle.load (filename)
    
    if Neumann:
        if subordinate:
            folder_name = "Neumann"
        else:
            folder_name = "Neumann_non_subordinate"
    else:
        folder_name = "Periodic"
    all_parameters = Best_parameters [folder_name][Nt][(variable_0, variable_1)]


all_keys = list(all_parameters.keys())

#####################################################################
# TRAINING

for key in all_parameters.keys():
    X_train, Y_train, _, _, _ =\
    prepare_train_test_set (variable_0, variable_1, level = 2)
    
    parameters_now = all_parameters [key]
    
    assert (key [0] == parameters_now ["layer_share_range"])
    assert (key [1] == parameters_now ["ptt_range"])
    
    print (
        "Training the model for weight_k_share : ", parameters_now ["layer_share_range"],\
         "and partition cardinality ", parameters_now ["ptt_range"]
    )
    
    append_to_saved_file_name = "_Index_"+ str (key[0]) + "_" + str (key[1]) + "_" + str (Nt)
    
    print ("\n Parameters in use :", parameters_now)
    
    all_results = fitting_several_models(
    cv, parallel, cpu, X_train, Y_train, X_train, Y_train, parameters_now,\
        Nx, Neumann, EPOCHS, patience, Nt, train_dt_U, train_dt_P,\
            with_PCA, None, True, append_to_saved_file_name,\
                save_history = save_history, subordinate = subordinate)
    
    #return results
    for j, a, b in all_results:
        if  j == 0:
            Accuracies, Parameters = a[np.newaxis,:], b
        else:
            Accuracies_tmp, Parameters_tmp = a[np.newaxis,:], b
            assert (Parameters_tmp == Parameters)
            Accuracies = np.vstack ([Accuracies, Accuracies_tmp]) 
    
    try: os.mkdir ("training")
    except: pass

    print ("Creating Accuracies and parameter pickled file")
    file_name = "Training_accuracies_"+ str (key[0])+ "_" + str (key[1])+ "_" +\
        str (Nt)+"_vary_eps_" + str (variable_0)+ "_" + str (variable_1)+".p"
    if with_PCA:
        file_name = "PCA_" + file_name

    file_name = "training/"+file_name
    with open (file_name, 'wb') as save:
        pickle.dump ( (Accuracies, Parameters), save, protocol = pickle.HIGHEST_PROTOCOL)        
        print ("Statistics pickled to ", file_name)
            
    evaluate_model (
        *key,  Nt, variable_0, variable_1, all_parameters,
        Neumann = Neumann, subordinate = subordinate)
