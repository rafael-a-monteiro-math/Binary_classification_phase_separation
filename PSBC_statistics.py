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

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

warnings.filterwarnings (action = "ignore", message = "internal issue")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
## Things necessary to do nice plots
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from  matplotlib.transforms import Affine2D


sys.path.insert (0, "MOTHER_PSBC/")

from  tfversion_binary_phase_separation import *
from tf_PSBC_extra_libs_for_training_and_grid_search import *
from tf_PSBC_extra_libs_for_classifier import *

IMAGES ="images/"

from matplotlib import rcParams
plt.rc ('axes', labelsize = 12)
plt.rc ('xtick', labelsize = 12)
plt.rc ('ytick', labelsize = 12)
plt.rc ('font', size = 12)
plt.rc ('grid', alpha = 0.6)
plt.rc ('legend', fontsize = 12)
rcParams ['font.family'] =  "Times New Roman"
rcParams ['mathtext.fontset'] = 'custom' 
rcParams ['lines.linewidth'] = 2
rcParams ['lines.markersize'] = 12
rcParams ['lines.markeredgewidth'] = 2

######################################################################
def create_statistics ():
    All_accuracies = {}  ### All accuracies in training and test sets

    Best_parameters = {}  ### Best hyperparameters obtained 
                        ### through grid search
    All_filepaths_best_weights = {}

    All_histories = {}

    with open ("Grids/digits_index.p", 'rb') as pickled_dic:
        grid_indexes  = pickle.load (pickled_dic)

    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate",
                        "Classifier_196", "PCA_196"]:

        All_accuracies [folder_name] = {}
        Best_parameters [folder_name] = {}
        All_filepaths_best_weights [folder_name] = {}
        All_histories [folder_name] = {}
        

        if folder_name in ["Classifier_196", "PCA_196"]:
            Nt_range = [2]
            k_range = [4] 
            lks_range = {2 : [1]} 
            if folder_name == "PCA_196":
                index_range = [0,1]
                pairs_of_digits = [(4,9), (3, 5)]    
            else: 
                index_range = np.arange(45)
            classifier = True
            with_PCA = folder_name == "PCA_196"
            eps_range = [0]
        else:
            Nt_range = [1, 2, 4]
            k_range = [1,2,4,8,16] 
            index_range = [0]
            classifier = False
            with_PCA = False
            lks_range = {1 : [1], 2: [1,2], 4: [1,4]}
            eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]

        for Nt in Nt_range:
            
            Best_parameters [folder_name][Nt] = {}
            All_accuracies [folder_name][Nt] ={}
            All_filepaths_best_weights [folder_name][Nt] = {}    
            All_histories [folder_name][Nt] ={}
        
            for index in index_range:
                if folder_name == "PCA_196":
                    variable_0, variable_1 = pairs_of_digits [index]
                else:
                    variable_0, variable_1 = grid_indexes [index]    

                All_accuracies [folder_name][Nt]\
                    [(variable_0, variable_1)] = {}
                Best_parameters [folder_name][Nt]\
                    [(variable_0, variable_1)] = {}
                All_filepaths_best_weights [folder_name][Nt]\
                    [(variable_0, variable_1)] = {}
                All_histories [folder_name][Nt]\
                    [(variable_0, variable_1)] = {}
                ##############################################
                #### ONLY FOR BEST PARAMETERS
                filepath = folder_name + "/" + str (Nt)+ "/"
                os.chdir (filepath)
                retrieve_best_par = BestPararameters_ptt_card_weights_k_shared_fixed (
                    Nt, variable_0, variable_1,
                    classifier = classifier,
                    with_PCA = with_PCA)
                parameters_model_1, parameters_model_Nt = fill_parameters_dict (
                    Nt,  retrieve_best_par, weight_sharing_split = True,
                    classifier = classifier)
                all_parameters = {**parameters_model_1, **parameters_model_Nt}
                os.chdir ("../../")
                ################################################

                for lks in lks_range [Nt]:
                    All_accuracies [folder_name][Nt]\
                        [(variable_0, variable_1)][lks] = {}
                    Best_parameters [folder_name][Nt]\
                        [(variable_0, variable_1)][lks] = {}
                    All_filepaths_best_weights [folder_name][Nt]\
                        [(variable_0, variable_1)][lks] = {}
                    All_histories [folder_name][Nt]\
                        [(variable_0, variable_1)][lks] = {}

                    for enum_pt_cdn, k in enumerate (k_range):
                        pt_cardnlty = int (784 / k) 

                        #### FOR BEST PARAMETERS    
                        Best_parameters [folder_name][Nt][(variable_0, variable_1)]\
                        [lks][pt_cardnlty] =  copy.deepcopy(all_parameters [lks, pt_cardnlty])

                        #### FOR ALL_FILEPATHS
                        All_filepaths_best_weights [folder_name][Nt][(variable_0, variable_1)]\
                        [lks][pt_cardnlty] = {}

                        All_filepaths_best_weights [folder_name][Nt][(variable_0, variable_1)]\
                        [lks][pt_cardnlty]["info"] =\
                        {"In case of folders Neumann, Periodic, and Neuman_non_subordinate, " +\
                        "indexes vary in [0,1, 2, 3, 4, 5], corresponding to " +\
                        "eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]. " +\
                        "In other cases the index is 0, corresponing to eps_range = [0]."}

                        #### FOR ALL_HISTORIES
                        All_histories [folder_name][Nt][(variable_0, variable_1)]\
                        [lks][pt_cardnlty] = {}

                        All_histories [folder_name][Nt][(variable_0, variable_1)]\
                        [lks][pt_cardnlty]["info"] =\
                        {"In case of folders Neumann, Periodic, and Neuman_non_subordinate, " +\
                        "indexes vary in [0,1, 2, 3, 4, 5], corresponding to " +\
                        "eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]. " +\
                        "In other cases the index is 0, corresponing to eps_range = [0]."}
                        

                        for enum_eps, eps in enumerate (eps_range):
                            All_filepaths_best_weights [folder_name][Nt]\
                            [(variable_0, variable_1)][lks]\
                            [pt_cardnlty][enum_eps] = {}
                        
                            list_best_weights = []
                            list_val_acc = []

                            name = str (enum_eps) + "_0_"\
                                    + str (pt_cardnlty) + "_" +\
                                        str (lks) +"_0_0"
                            
                            if folder_name in ['PCA_196','Classifier_196']:
                                
                                append_to_saved_file_name = "_var0_" +\
                                    str (variable_0)+ "_var1_" + str (variable_1)
                                
                                if with_PCA:
                                        append_to_saved_file_name = "_PCA_" +\
                                            append_to_saved_file_name
                            else:
                                append_to_saved_file_name = "_Index_" + str (lks) +\
                                    "_" + str (pt_cardnlty) + "_" + str (Nt)
                                
                            all_hist_aux = []

                            for fold in range (5):

                                ### FOR ALL_FILEPATHS_BEST WEIGHT
                                filepath = folder_name +\
                                "/" + str (Nt) + "/weights/" + str (enum_eps)+\
                                    "_0_" + str (pt_cardnlty) +\
                                        "_" + str (lks)+ "_0_0_fold_" + str (fold)
                                
                                valpath = filepath +\
                                    append_to_saved_file_name+ "_val_accuracy.p"
                                filepath = filepath +\
                                    append_to_saved_file_name+ "_best_weights.h5"
                                
                                list_best_weights.append (filepath)        
                                list_val_acc.append (valpath)    

                                ### FOR ALL_HISTORY    
                                dict_aux = {}
                                filepath_history = folder_name + "/" + str (Nt) + "/"+ \
                                "history/" + name + "_fold_" + str (fold) +\
                                    append_to_saved_file_name +".p"

                                with open (filepath_history, 'rb') as pickled_dic:
                                    hist_now = pickle.load (pickled_dic)

                                for new_key , key in zip(
                                    ["dt_P", "dt_U","w_p_inf", "w_u_inf"],\
                                    ["dt_P ", "dt_U ","||W_P||_{infty} ", "||W_U||_{infty} "]):
                                    dict_aux [new_key] = hist_now [key]
                                all_hist_aux.append (copy.deepcopy(dict_aux))
                            
                            ### FOR ALL_FILEPATHS_BEST WEIGHT
                            All_filepaths_best_weights [folder_name][Nt]\
                                [(variable_0, variable_1)][lks]\
                                    [pt_cardnlty][enum_eps]\
                                        ["best_weights"] = list_best_weights
                            All_filepaths_best_weights [folder_name][Nt]\
                                [(variable_0, variable_1)][lks]\
                                    [pt_cardnlty][enum_eps]\
                                        ["val_acc"]= list_best_weights
                            
                            ### FOR ALL_HISTORY    
                            All_histories [folder_name][Nt]\
                            [(variable_0, variable_1)][lks]\
                            [pt_cardnlty][enum_eps] = all_hist_aux
                        
                    ### FOR ALL_ACCURACIES
                    for a in zip(["train_set", "test_set"],["Training_accuracies_","Test_set_accuracies_"]):
                        
                        key, prefix  = a
                        
                        for enum_pt_cdn, k in enumerate (k_range):
                            pt_cardnlty = int (784 / k) 
                
                            if folder_name in ["Classifier_196", "PCA_196"]:
                                filename =prefix + str (lks)+ "_" + str (pt_cardnlty)+\
                                    "_" + str (Nt)+ "_classifier_"\
                                        + str (variable_0)+ "_" + str (variable_1)+ ".p"
                                if folder_name == "PCA_196":
                                    filename = "PCA_" +filename
                            else:
                                filename =\
                                prefix + str (lks)+ "_" + str (pt_cardnlty)+ "_" +\
                                    str (Nt)+ "_vary_eps_"\
                                        + str (variable_0)+ "_" + str (variable_1)+ ".p"

                            with open (os.path.join (
                                folder_name, str (Nt)+\
                                    "/training/" +filename), 'rb') as pickled_dic:
                                training  = pickle.load (pickled_dic)

                            Matrix = np.squeeze(training[0]) 
                            if training [0].shape[1] ==1:
                                Matrix = Matrix[:,np.newaxis]
                            
                            if enum_pt_cdn == 0: 
                                if len (k_range) ==1:
                                    All_accuracies [folder_name][Nt]\
                                        [(variable_0, variable_1)][lks][key] =\
                                    np.zeros ((5, 1), dtype=np.float32)
                                else:
                                    All_accuracies [folder_name][Nt]\
                                        [(variable_0, variable_1)][lks][key] =\
                                    np.zeros ((5, 5, 6), dtype=np.float32)
                        
                            for enum_eps, eps  in  enumerate (eps_range[:training[0].shape[1]]):
                                ###################################
                                print ("\nAsserting viscosity rate")
                                assert (training[1][(enum_eps,0,0,0,0,0)][0] == eps)
                                print ("Viscosities match!")
                                print ("\nAsserting pt_cardnlty")
                                print (training[1][(enum_eps,0,0,0,0,0)][2], pt_cardnlty)
                                print ("Parameterization cardinalities match!")
                                print ("\nAsserting layers_K_shared")
                                assert (training[1][(enum_eps,0,0,0,0,0)][3] == lks)
                                print ("layers_K_shared match!")
                                if len (k_range) ==1:
                                    All_accuracies [folder_name][Nt]\
                                    [(variable_0, variable_1)][lks][key][:] = Matrix
                                else:
                                    All_accuracies [folder_name][Nt]\
                                    [(variable_0, variable_1)][lks][key][:,enum_pt_cdn,:] = Matrix

                                All_accuracies [folder_name][Nt]\
                                [(variable_0, variable_1)][lks][key + "_mean"] =\
                                np.mean(
                                    All_accuracies [folder_name][Nt]\
                                    [(variable_0, variable_1)][lks][key], axis =0)
                                All_accuracies [folder_name][Nt]\
                                [(variable_0, variable_1)][lks][key + "_std"] =\
                                np.std(
                                    All_accuracies [folder_name][Nt]\
                                    [(variable_0, variable_1)][lks][key], axis =0)


    with open ("Statistics/All_accuracies.p", 'wb') as save:
        pickle.dump (All_accuracies, save, protocol = pickle.HIGHEST_PROTOCOL)
                            
    with open ("Statistics/Best_parameters.p", 'wb') as save:
        pickle.dump (Best_parameters, save, protocol = pickle.HIGHEST_PROTOCOL)

    with open ("Statistics/All_filepaths_best_weights.p", 'wb') as save:
        pickle.dump (All_filepaths_best_weights, save, protocol = pickle.HIGHEST_PROTOCOL)


    with open ("Statistics/All_histories.p", 'wb') as save:
        pickle.dump (All_histories, save, protocol = pickle.HIGHEST_PROTOCOL)


###############################################################################


def create_snapshot_best_weights ():

    with open ("Statistics/Best_parameters.p", 'rb') as pickled_dic:
        Best_parameters  = pickle.load (pickled_dic)

    with open ("Statistics/All_filepaths_best_weights.p", 'rb') as pickled_dic:
        All_filepaths_best_weights  = pickle.load (pickled_dic)

    with open ("Grids/digits_index.p", 'rb') as pickled_dic:
        grid_indexes  = pickle.load (pickled_dic)

    Snapshot_best_weights = {}
    I = Initialize_parameters ()
    L = load_PSBC_model()

    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate",\
                        "Classifier_196","PCA_196"]:

        Snapshot_best_weights [folder_name] = {}
        
        Neumann = not (folder_name == "Periodic")        
        Nx = 784
        train_dt_U = True
        train_dt_P = True
        subordinate = not (folder_name == "Neumann_non_subordinate")        
        with_phase = True
        
        if folder_name in ["Classifier_196", "PCA_196"]:
            Nt_range = [2]
            k_range = [4]
            lks_range = {2 : [1]} 
            if folder_name == "PCA_196":
                index_range = [0,1]
                pairs_of_digits = [(4,9), (3, 5)]
            else: 
                index_range = np.arange(45)
            classifier = True
            with_PCA = folder_name == "PCA_196"
            eps_range = [0]
        else:
            Nt_range = [1,2,4]
            k_range = [1,2, 4, 8, 16]
            index_range = [0]
            classifier = False
            with_PCA = False
            lks_range = {1 : [1], 2: [1,2], 4: [1,4]}
            eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]

        for Nt in Nt_range:
            Snapshot_best_weights [folder_name][Nt] = {}

            for index in index_range:
                
                if folder_name == "PCA_196":
                    variable_0, variable_1 = pairs_of_digits [index]
                else:
                    variable_0, variable_1 = grid_indexes [index]    
        
                Snapshot_best_weights [folder_name][Nt][(variable_0, variable_1)] = {}
                
                best_parameters_now = Best_parameters[folder_name][Nt]\
                [(variable_0, variable_1)]
                
                for lks in lks_range [Nt]:
                    
                    Snapshot_best_weights [folder_name][Nt]\
                        [(variable_0,variable_1)][lks] = {}

                    for enum_pt_cdn, k in enumerate (k_range):
                        pt_cardnlty = int (784 / k) 
                        Snapshot_best_weights [folder_name][Nt]\
                            [(variable_0,variable_1)][lks]\
                                [pt_cardnlty] = {}
                        parameters = Best_parameters[folder_name][Nt]\
                            [(variable_0,variable_1)][lks]\
                                [pt_cardnlty]

                        dt = parameters ["dt_range"][0]
                        lr_U = parameters ["lr_U_range"][0]
                        lr_P = parameters ["lr_P_range"][0]   

                        assert (pt_cardnlty == parameters ["ptt_range"][0])
                        assert (lks == parameters ["layer_share_range"][0])
                        
                        for enum_eps , eps in enumerate (parameters ["eps_range"]):

                            ## Now we fit the model 5 times
                            Snapshot_best_weights [folder_name][Nt]\
                                [(variable_0,variable_1)][lks]\
                                    [pt_cardnlty][enum_eps] = {}
                            
                            print (folder_name, Nt, variable_0, variable_1, lks)
                            list_best_weights =\
                                All_filepaths_best_weights [folder_name][Nt]\
                                    [(variable_0, variable_1)][lks]\
                                        [pt_cardnlty][enum_eps]["best_weights"]
                        
                            ### Set up the model
                            parameters = I.dictionary (
                                Nx, eps, Nt, pt_cardnlty,
                                lks, dt = dt,
                                Neumann = Neumann)

                            lr_schedule_U =\
                                keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate = lr_U,
                                decay_steps = 5,
                                decay_rate = 0.99,
                                staircase = True)

                            lr_schedule_P =\
                                keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate = lr_P,
                                decay_steps = 5,
                                decay_rate = 0.99,
                                staircase = True)

                            model = PSBC_build_model (
                                parameters,
                                train_dt_U = train_dt_U,
                                train_dt_P = train_dt_P,
                                lr_U = lr_schedule_U,
                                lr_P= lr_schedule_P, 
                                subordinate = subordinate,
                                with_phase = with_phase,
                                loss = keras.losses.mean_squared_error,
                                metrics = [classify_zero_one],
                                print_summary = enum_eps == 0)

                            P_now, U_now  = [], []

                            ### Retrieving all folds
                            for filepath in list_best_weights:

                                #print ("filepath", filepath, key)
                                L.load_model_with_layers (model, filepath)
                                P_now.append(
                                    tf.stack(
                                        model.psbc_UP.P_network.trainable_variables,
                                        axis = 0))
                                U_now.append(
                                    tf.stack(
                                        model.psbc_UP.U_network.trainable_variables,
                                        axis = 0))
                                
                            Snapshot_best_weights [folder_name][Nt]\
                                [(variable_0,variable_1)][lks]\
                                    [pt_cardnlty][enum_eps]["P"] = \
                                        np.copy(P_now)
                            
                            Snapshot_best_weights [folder_name][Nt]\
                                [(variable_0,variable_1)][lks]\
                                    [pt_cardnlty][enum_eps]["U"] = \
                                        np.copy(U_now)

    with open ("Statistics/Snapshot_best_weights.p", 'wb') as save:
        pickle.dump (Snapshot_best_weights, save, protocol = pickle.HIGHEST_PROTOCOL)



###############################################################################

def create_snapshot_best_weights_stats ():

    with open ("Statistics/Snapshot_best_weights.p", 'rb') as pickled_dic:
        Snapshot_best_weights  = pickle.load (pickled_dic)

    with open ("Grids/digits_index.p", 'rb') as pickled_dic:
        grid_indexes  = pickle.load (pickled_dic)

    Snapshot_best_weights_stats = {}
    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate",\
                        "Classifier_196", "PCA_196"]:

        
        if folder_name in ["Classifier_196", "PCA_196"]:
            Nt_range = [2]
            k_range = [4]
            lks_range = {2 : [1]} 
            if folder_name == "PCA_196":
                index_range = [0,1]
                pairs_of_digits = [(4,9), (3, 5)]
            else: 
                index_range = np.arange(45)
            classifier = True
            with_PCA = folder_name == "PCA_196"
            eps_range = [0]
        else:
            Nt_range = [1,2,4]
            k_range = [1,2, 4, 8, 16]
            index_range = [0]
            classifier = False
            with_PCA = False
            lks_range = {1 : [1], 2: [1,2], 4: [1,4]}
            eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]
        
        Snapshot_best_weights_stats [folder_name] = {}
        
        for Nt in Nt_range:
            Snapshot_best_weights_stats [folder_name][Nt] = {}

            for index in index_range:

                if folder_name == "PCA_196":
                    variable_0, variable_1 = pairs_of_digits [index]
                else:
                    variable_0, variable_1 = grid_indexes [index]    
            
                Snapshot_best_weights_stats [folder_name][Nt]\
                [(variable_0,variable_1)] = {}


                for lks in lks_range [Nt]:
                    
                    Snapshot_best_weights_stats [folder_name][Nt]\
                    [(variable_0,variable_1)][lks]= {}

                    for matrix_type in ["max", "min", "diameter"]:
                        
                        for variables in ["P", "U"]:
                        
                            for enum_pt_cdn, k in enumerate (k_range):
                                pt_cardnlty = int (784 / k) 

                                for enum_eps , eps in enumerate (eps_range):
                                
                                    if enum_pt_cdn == 0  and enum_eps == 0: 
                                        
                                        Snapshot_best_weights_stats [folder_name][Nt]\
                                        [(variable_0,variable_1)][lks]\
                                        [matrix_type + "_" + variables] =\
                                        np.zeros (
                                            (5, len (k_range), len (eps_range)),
                                            dtype=np.float32
                                        )

                                    if matrix_type == "max":
                                        aux = np.maximum([
                                            np.max(A) for A in\
                                            Snapshot_best_weights[folder_name][Nt]\
                                                [(variable_0,variable_1)][lks]\
                                                    [pt_cardnlty][enum_eps]\
                                                        [variables]
                                                        ],1)
                                    elif matrix_type == "min":
                                        aux = np.minimum([
                                            np.min(A) for A in\
                                            Snapshot_best_weights[folder_name][Nt]\
                                                [(variable_0,variable_1)][lks]\
                                                    [pt_cardnlty][enum_eps]\
                                                        [variables]
                                                        ], 0)
                                    else:
                                        aux =\
                                            Snapshot_best_weights_stats[folder_name][Nt]\
                                                [(variable_0,variable_1)][lks]\
                                                    ["max_" +variables][:,enum_pt_cdn,enum_eps]\
                                                        - Snapshot_best_weights_stats[folder_name][Nt]\
                                                            [(variable_0,variable_1)][lks]\
                                                                ["min_" +variables][:,enum_pt_cdn,enum_eps]
                                        
                                    assert (len (aux)== 5)
                                    
                                    Snapshot_best_weights_stats [folder_name][Nt]\
                                        [(variable_0,variable_1)][lks]\
                                            [matrix_type + "_" + variables]\
                                                [:,enum_pt_cdn,enum_eps] = np.copy(aux)
                                    
                        for func, func_name in zip ([np.mean, np.std],["mean", "std"]):
                            for variables in ["P", "U"]:
                                aux = func(Snapshot_best_weights_stats [folder_name][Nt]\
                                    [(variable_0,variable_1)][lks]\
                                        [matrix_type+ "_" +variables], axis = 0)

                                Snapshot_best_weights_stats [folder_name][Nt]\
                                    [(variable_0,variable_1)][lks]\
                                        [func_name + "_" +matrix_type + "_" + variables] = aux

    with open ("Statistics/Snapshot_best_weights_stats.p", 'wb') as save:
        pickle.dump (Snapshot_best_weights_stats, save, protocol = pickle.HIGHEST_PROTOCOL)          


################################################################################

def create_confusion_matrices ():

    All_confusion_matrices = {}

    with open ("Grids/digits_index.p", 'rb') as pickled_dic:
        grid_indexes  = pickle.load (pickled_dic)

    for folder_name in ["Classifier_196", "PCA_196"]:

        os.chdir (folder_name)
        
        how_many = 5
        
        All_confusion_matrices [folder_name] = {}
        
        if folder_name in ["Classifier_196", "PCA_196"]:
            Nt_range = [2]
            k_range = [4]
            lks_range = {2 : [1]} 
            if folder_name == "PCA_196":
                index_range = [0,1]
                pairs_of_digits = [(4,9), (3, 5)]
            else: 
                index_range = np.arange(45)
            classifier = True
            with_PCA = folder_name == "PCA_196"
            eps_range = [0]
        else:
            Nt_range = [1,2,4]
            k_range = [1,2, 4, 8, 16]
            index_range = [0]
            classifier = False
            with_PCA = False
            lks_range = {1 : [1], 2: [1,2], 4: [1,4]}
            eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]

        for Nt in Nt_range:
            
            os.chdir (str (Nt))
        
            All_confusion_matrices [folder_name][Nt] = {}

            for index in index_range:

                if folder_name == "PCA_196":
                    variable_0, variable_1 = pairs_of_digits [index]
                else:
                    variable_0, variable_1 = grid_indexes [index]    


                All_confusion_matrices [folder_name][Nt]\
                [(variable_0,variable_1)] = {}
                
                model = PSC_ensemble (variable_0 = variable_0, variable_1 = variable_1,
                            how_many = how_many, with_PCA = with_PCA)
                ### Prepare dataset

                S = select_split_pickle (level = 2)

                X_test, Y_test, _ = S.select_variables_from_pickle (
                        variable_0, variable_1,type_of = "test", averaged = False
                    )

                y_pred = model.hard_vote(X_test)
        
                print ("\n\nVariables", variable_0, variable_1)
                Conf_matrix_raw = confusion_matrix (Y_test, y_pred)
                Conf_matrix_total = np.sum (Conf_matrix_raw, axis = 1,keepdims = True)
                Conf_matrix_relative  =  np.asarray (Conf_matrix_raw / Conf_matrix_total, dtype=np.float32)
                
                All_confusion_matrices [folder_name][Nt]\
                [(variable_0,variable_1)]\
                ["relative"] = Conf_matrix_relative
                All_confusion_matrices [folder_name][Nt]\
                [(variable_0,variable_1)]\
                ["raw"] = Conf_matrix_raw
                All_confusion_matrices [folder_name][Nt]\
                [(variable_0,variable_1)]\
                ["accuracy"] = accuracy_score (Y_test, y_pred)
                All_confusion_matrices [folder_name][Nt]\
                [(variable_0,variable_1)]\
                ["f1_score"] = f1_score (Y_test, y_pred)

                print ("\n\nConfusion matrix\n ", Conf_matrix_relative)
                
            
            os.chdir ("../")

        os.chdir ("../")

    with open ("Statistics/All_confusion_matrices.p", 'wb') as save:
        pickle.dump (All_confusion_matrices, save, protocol = pickle.HIGHEST_PROTOCOL)          


################################################################################

def create_multi_class_classifier_confusion_matrix ():

    mnist = fetch_openml('mnist_784', version = 1)

    X, Y = mnist ["data"], mnist ["target"]

    M = MinMaxScaler(feature_range = (0,1))
    M.fit (X)
    X_norm = M.transform (X)
    _, _, X_norm_test, Y_test = X_norm [:60000,:], Y [:60000], X_norm [60000:,:], Y [60000:]

    Y_test = Y_test.astype ("uint8")

    All_confusion_matrices_multiclass = {}

    with open ("Grids/digits_index.p", 'rb') as pickled_dic:
        grid_indexes  = pickle.load (pickled_dic)

    for folder_name in ["Classifier_196"]:

        os.chdir (folder_name)
        
        how_many = 5
        
        All_confusion_matrices_multiclass [folder_name] = {}
        
        if folder_name in ["Classifier_196", "PCA_196"]:
            Nt_range = [2]
            k_range = [4]
            lks_range = {2 : [1]} 
            if folder_name == "PCA_196":
                index_range = [0,1]
                pairs_of_digits = [(4,9), (3, 5)]
            else: 
                index_range = np.arange (45)
            classifier = True
            with_PCA = folder_name == "PCA_196"
            eps_range = [0]
        else:
            Nt_range = [1,2,4]
            k_range = [1,2, 4, 8, 16]
            index_range = [0]
            classifier = False
            with_PCA = False
            lks_range = {1 : [1], 2: [1,2], 4: [1,4]}
            eps_range =  np.r_[[0], np.power (.5, [4,3,2,1,0])]

        for Nt in Nt_range:

            All_confusion_matrices_multiclass [folder_name][Nt] = {}

            os.chdir (str (Nt))

            psc = PSBC_multi_class ()

            score ={
                "tournament": [],
                "pred_tournament" : [],
                "hard_vote": [],
                "pred_hard_vote" : []
            }
            
            for classification_method in ["hard_vote", "tournament"]:

                All_confusion_matrices_multiclass [folder_name][Nt]\
                [classification_method] = {}
            
                T = tournament(hardvote_or_tournament= classification_method)
                
                for i in range(10):
                    print (i)
                    y_pred = psc (np.asarray (X_norm_test [1000 * i:1000 * (i + 1)], dtype = np.float32))
                    M = y_pred
                    class_pred = T.aux_tournament(M)
                    score ["pred_" + classification_method].append (np.copy (class_pred))
                    score [classification_method].append(np.sum(class_pred == Y_test [1000 * i:1000 * (i + 1)])/1000)
                
                y_pred = np.concatenate (score ["pred_" + classification_method])    
                Conf_matrix_raw = confusion_matrix (Y_test, y_pred)
                Conf_matrix_total = np.sum (
                    Conf_matrix_raw, axis = 1, keepdims = True)
                Conf_matrix_relative  =  np.asarray (
                    Conf_matrix_raw / Conf_matrix_total, dtype = np.float32)
            
                All_confusion_matrices_multiclass [folder_name][Nt]\
                    [classification_method]["relative"] = Conf_matrix_relative
                All_confusion_matrices_multiclass [folder_name][Nt]\
                    [classification_method]["raw"] = Conf_matrix_raw
                All_confusion_matrices_multiclass [folder_name][Nt]\
                    [classification_method]["accuracy"] = accuracy_score (Y_test, y_pred)
                
                print (accuracy_score (Y_test, y_pred), np.sum(y_pred == Y_test)/len (Y_test))
            
        os.chdir ("../")

    os.chdir ("../")

    with open ('Statistics/All_confusion_matrices_multiclass.p', 'wb') as handle:
        pickle.dump (All_confusion_matrices_multiclass, handle, protocol = pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    print ("Creating independent pickled files with statistics")
    create_statistics ()
    
    print ("Creating pickled files with Snapshots of W_u and W_p weights")
    create_snapshot_best_weights ()
    
    print ("Creating pickled files with statitics of Snapshots W_u and W_p")
    create_snapshot_best_weights_stats ()
    
    print ("Creating confusion matrices for Classifier and PCA")
    create_confusion_matrices ()
    
    print ("Creating confusion matrices for multiclass classifier")
    create_multi_class_classifier_confusion_matrix ()

    print ("Done!")
    