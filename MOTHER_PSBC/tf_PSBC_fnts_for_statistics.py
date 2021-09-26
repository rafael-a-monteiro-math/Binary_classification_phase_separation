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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
## Things necessary to do nice plots
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from  matplotlib.transforms import Affine2D

sys.path.insert (0, "MOTHER_PSBC/")

from  tfversion_binary_phase_separation import *
from tf_PSBC_extra_libs_for_training_and_grid_search import *

IMAGES ="images/"



def figure_save(
    figure_name, tight_layout = True, figure_extension = "pdf", resolution = 600
):
    """
    This function saves plots as a file.
    
    Parameters
    ----------
    figure_namet : string
        Name of saved figure
    tight_layout : {bool, True}, optional
        Saved figure specifications
    figure_extension : {string, "pdf"}, optional
        Saved figure extension
    resolution : {int,600}, optional
        Saved figure resolution

    Returns
    -------
    No return
      
    References
    -------
    This function is based on a function used in one of the notebooks of the book
    
        'Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow:
        Concepts, Tools, and Techniques to Build Intelligent Systems', 
        2nd edition, by Aurelien Geron. 
    
    """

    path = os.path.join(IMAGES, figure_name.replace(" ", "_") + "." + figure_extension)
    print ("Saving figure as ", figure_name.replace(" ", "_"))
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = figure_extension, dpi = resolution)



def collect_all_scores (folder = "Neumann", training=True, layers_k_shared = True):
    """
    Returns
    -------
    """
    all_scores = {}
    
    prefix = "Training_accuracies_" if training else "Test_set_accuracies_"
    
   
    for Nt in [1,2,4]:
        
        if layers_k_shared:
            wks = Nt
        else: wks = 1
        
        for k in [1,2,4,8,16]:
            pt_cardnlity = int(784/k) 
            filename = prefix + str (wks)+"_"+str (pt_cardnlity)+"_"+ str (Nt)+"_vary_eps_"\
                +str (0)+"_"+str (1)+".p"

            with open (os.path.join(folder, "vary_eps/"+str (Nt)+"/training/"+filename), 'rb') as pickled_dic:
                training  = pickle.load (pickled_dic)

            Matrix = np.squeeze(training[0]) #np.vstack(np.array_split(np.squeeze(training[0]), 5))


            all_scores [(Nt,pt_cardnlity, wks)] = Matrix
    return all_scores


def collect_all_in_dict (folder = "Neumann"):
    """
    Returns
    -------
    """    
    dic_with_all = {}
    for Nt in [1,2,4]:

        pt_feastible = [1]
        if Nt > 1:
            pt_feastible.append(Nt)
        for i in pt_feastible:
            dic_with_all.update({**collect_all_scores(folder = folder, training=False, layers_k_shared= (i==1)) })
    
    return dic_with_all


def create_stas_matrix (dict_data):
    """
    Returns
    -------
    """
    stats_matrix_mean = {}
    stats_matrix_std = {}
    for Nt in [1,2,4]:

        pt_feastible = [1]
        if Nt > 1:
            pt_feastible.append(Nt)
    
        for layers_k_shared in pt_feastible:
            stats_matrix_mean [(Nt, layers_k_shared)] = np.zeros ((5,6))
            stats_matrix_std [(Nt, layers_k_shared)] = np.zeros ((5,6))


            for exp, k in enumerate ([1,2,4,8,16]):
                pt_cardnlity = int(784/k) 
                stats_matrix_mean [(Nt, layers_k_shared)][exp, :] = np.mean(dict_data [(Nt, pt_cardnlity,layers_k_shared )], axis =0)
                stats_matrix_std [(Nt, layers_k_shared)][exp, :] = np.std(dict_data [(Nt, pt_cardnlity,layers_k_shared )], axis =0)
    
    return {"mean" : stats_matrix_mean, "std" : stats_matrix_std}


##### CREATING GRIDS!!!
def PSBC_dict_of_best_hyperparameters (variable_0 = 0,  variable_1 = 1):
    """
    Returns
    -------
    """

    Best_parameters = {}

    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate"]:

        Best_parameters [folder_name] = {}
        
        for Nt in [1,2,4]:
        
            filepath = folder_name + "/vary_eps/"+str (Nt)+"/"

            os.chdir (filepath)
            retrieve_best_par = BestPararameters_ptt_card_weights_k_shared_fixed (
                Nt, variable_0, variable_1)

            parameters_model_1, parameters_model_Nt = fill_parameters_dict (
                Nt,  retrieve_best_par, weight_sharing_split = True)

            all_parameters = {**parameters_model_1, **parameters_model_Nt}
            os.chdir ("../../../")
        
            Best_parameters [folder_name][Nt] = copy.deepcopy(all_parameters)
    
    return Best_parameters
#####################################################################

def PSBC_dict_with_best_weights_location (variable_0 = 0,  variable_1 = 1):
    """
    Returns
    -------
    """

    All_filepaths_best_weights = {}

    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate"]:

        All_filepaths_best_weights [folder_name] = {}
        
        for Nt in [1,2,4]:
            All_filepaths_best_weights [folder_name][Nt] ={}
            for key in Best_parameters[folder_name][Nt].keys():
                
                
                All_filepaths_best_weights [folder_name][Nt][key] = {}
                
            
                pt_cardnlty = key[1]
                append_to_saved_file_name = "_Index_"+ str (key[0]) + "_" + str (key[1]) + "_" + str (Nt)
                
                
                for enum_eps in range(6):
                    All_filepaths_best_weights [folder_name][Nt][key][enum_eps] =[]
                    
                    
                    for fold in range (5):
                        filepath = folder_name +\
                        "/vary_eps/"+str (Nt)+"/weights/"+str (enum_eps)+"_0_"+str (key[1]) +\
                        "_" + str (key[0])+"_0_0_fold_"+str (fold)+\
                        append_to_saved_file_name+"_best_weights.h5"

                        All_filepaths_best_weights [folder_name][Nt][key][enum_eps].append(filepath)        
            
    return All_filepaths_best_weights
#####################################################################



def PSBC_dict_with_best_weights_statistics ():
    """
    Returns
    -------
    """

    All_best_weights_stats = {}

    for folder_name in ["Neumann", "Periodic", "Neumann_non_subordinate"]:

        All_best_weights_stats [folder_name] = {}
        
        for Nt in [1, 2, 4]:
            
            All_best_weights_stats [folder_name][Nt] ={}
            layers_K_shared = [1] if Nt ==1 else [1, Nt]
            
            for lks in layers_K_shared:
                All_best_weights_stats [folder_name][Nt][lks] = {}
                
                "MAX"
                All_best_weights_stats [folder_name][Nt][lks]["max_mean_P"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["max_std_P"] = np.zeros((5,6))
                All_best_weights_stats [folder_name][Nt][lks]["max_mean_U"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["max_std_U"] = np.zeros((5,6))
                "MIN"
                All_best_weights_stats [folder_name][Nt][lks]["min_mean_P"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["min_std_P"] = np.zeros((5,6))
                All_best_weights_stats [folder_name][Nt][lks]["min_mean_U"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["min_std_U"] = np.zeros((5,6))
                "DIAMETER"
                All_best_weights_stats [folder_name][Nt][lks]["dmt_mean_P"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["dmt_std_P"] = np.zeros((5,6))
                All_best_weights_stats [folder_name][Nt][lks]["dmt_mean_U"] = np.zeros((5,6)) 
                All_best_weights_stats [folder_name][Nt][lks]["dmt_std_U"] = np.zeros((5,6))
            
            
                for enum_pt_cdn in range (5):
                    k = int(np.power(2,enum_pt_cdn))
                    pt_cardnlty = int(784/k)

                    for eps in range (6):
            
                        if eps == 0:
                            max_P_aux = [np.max(A) for A in\
                                All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["P"]]
                            max_U_aux = [np.max(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["U"]]
                            min_P_aux = [np.min(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["P"]]
                            min_U_aux = [np.min(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["U"]]
                            dmt_P_aux = np.maximum(max_P_aux, 1) - np.minimum(min_P_aux, 0)
                            dmt_U_aux = np.maximum(max_U_aux, 1) - np.minimum(min_U_aux, 0)
                        else:
                            max_P_aux = [np.max(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["P"]]
                            max_U_aux = [np.max(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["U"][0]]
                            min_P_aux = [np.min(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["P"]]
                            min_U_aux = [np.min(A) for A in\
                                        All_best_weights[folder_name][Nt][(lks,pt_cardnlty)][eps]["U"][0]]
                            dmt_P_aux = np.maximum(max_P_aux, 1) - np.minimum(min_P_aux, 0)
                            dmt_U_aux = np.maximum(max_U_aux, 1) - np.minimum(min_U_aux, 0)
                        
                        ## MAX
                        All_best_weights_stats [folder_name][Nt][lks]["max_mean_P"][enum_pt_cdn,eps] = np.mean (max_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["max_std_P"][enum_pt_cdn,eps]  = np.std (max_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["max_mean_U"][enum_pt_cdn,eps] = np.mean (max_U_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["max_std_U"][enum_pt_cdn,eps]  = np.std (max_U_aux)
                        
                        ## MIN
                        All_best_weights_stats [folder_name][Nt][lks]["min_mean_P"][enum_pt_cdn,eps] = np.mean (max_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["min_std_P"][enum_pt_cdn,eps]  = np.std (max_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["min_mean_U"][enum_pt_cdn,eps]  = np.mean (max_U_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["min_std_U"][enum_pt_cdn,eps]  = np.std (max_U_aux)

                        ## DMT
                        All_best_weights_stats [folder_name][Nt][lks]["dmt_mean_P"][enum_pt_cdn,eps] = np.mean (dmt_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["dmt_std_P"][enum_pt_cdn,eps]  = np.std (dmt_P_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["dmt_mean_U"][enum_pt_cdn,eps]  = np.mean (dmt_U_aux)
                        All_best_weights_stats [folder_name][Nt][lks]["dmt_std_U"][enum_pt_cdn,eps]  = np.std (dmt_U_aux)(self, parameter_list)
    
    return All_best_weights_stats   