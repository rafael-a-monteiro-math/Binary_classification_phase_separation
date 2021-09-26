#!/usr/bin/env python
""" Python library for the paper 
'Binary Classification as a Phase Separation process'.
This module contains an implementation of the PSBC method, besides auxiliary 
functions. 
 """
  
__author__ = "Rafael de Araujo Monteiro"
__affiliation__ =\
    "Mathematics for Advanced Materials - Open Innovation Lab,"\
        +"\n (Matham-OIL, AIST),"\
            +"\n Sendai, Japan"
__copyright__ = "None"
__credits__ = ["Rafael Monteiro"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "Rafael Monteiro"
__email__ = "rafael.a.monteiro.math@gmail.com"
__github__ = "https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation"
__date__ =  "September, 2021"

#####################################################################
###    This code is part of the simulations done for the paper 
###    "Binary Classification as a Phase Separation Process",
###    by Rafael Monteiro.
###    
###    author : Rafael Monteiro
###    affiliation : Mathematics for Advanced Materials - 
###                  Open Innovation Lab (MathAM-OIL, AIST)
###                  Sendai, Japan
###    email : rafael.a.monteiro.math@gmail.com
###    date : September 2021
###
#####################################################################
### IMPORT LIBRARIES
#####################################################################

#import  matplotlib.pyplot as plt
import scipy.sparse as sc
import itertools as it
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
import shutil 
import copy
import time
import glob
import sys
import os
try: ## In order to open and save dictionaries
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import multiprocess as mp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

warnings.filterwarnings (action = "ignore", message = "internal issue")

############################################################################
from  tfversion_binary_phase_separation import *
############################################################################
##https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath

def BestPararameters_ptt_card_weights_k_shared_fixed(
    Nt, var0, var1, with_accuracy = False, classifier = False,
    with_PCA = False, ptt_value = int (784 / 4)
    ):
    """
    'BestPararameters_ptt_card_weights_k_shared_fixed'
    relies on the folder structure of the datafiles, 
    and must be called either from 
    vary_eps or,
    in the case of classifier, from
    the classifier folder.
    It returns a dictionary with the best hyperparameters associated with 
    a certain architecture (chosen through grid search beforehand).


    Parameters
    ----------
    Nt : int, 
        Number of layers in the PSBC.
    var0 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    var1 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset. 
        It is assumed that var0 is different from var1.

    with_accuracy : {bool, False}, optional,

    classifier : {bool, False}, optional,
        If True, then the function retrieves best parameters for the multiclass
        classifier, otherwise it retrieves best parameters used in grid search
        in variables "0" and "1".
    with_PCA : {bool, False}, optional,
        If True, then the model searches for a basis matrix
        that is non-canonical, usin PCA. This matrix is provided.
    ptt_value : {int, int(784 / 4)}, optional,
        Used for the PCA, to fix the parameterization cardinality used
        on the grid.
    Returns
    -------

    """
    other_folder = "grid_search/"
    
    if with_PCA:
        file_now = other_folder + "PCA_all_grid_search_results_" + \
        str (min (var0, var1)) + "_" + str (max (var0, var1)) + ".p"
    else:     
        file_now = other_folder + "Normal_all_grid_search_results" + \
        str (min (var0, var1)) + "_" + str (max (var0, var1)) + ".p"
    
    with open (file_now, 'rb') as pickled_dic:
        grid  = pickle.load (pickled_dic)
    Accuracies = grid [0]
    Parameters = grid [1]
    averages = np.mean (Accuracies, axis = 0, keepdims = True)
    
    if Nt == 1:
        layers_k_shared = [Nt]
    else: 
        layers_k_shared = [1, Nt]
    
    Nx = 784
    
    if classifier:
        ptt_range = [ptt_value]
        layers_k_shared = [1]
    else:
        ptt_range = [int (Nx / k) for k in [1, 2, 4, 8, 16]]
    
    max_per_k_sharing = {
        tuple (-np.ones (len (Accuracies.shape))) :\
            Parameters [tuple (-np.ones (len (Accuracies.shape)))]
    }
    #names = Parameters [tuple (-np.ones (len (Accuracies.shape)))]

    for wks_enum, wks in enumerate (layers_k_shared):
        for ptt_card_enum, ptt_card in enumerate (ptt_range):
            average_constrained = averages [:, :, ptt_card_enum, wks_enum,:,:]
            max_average_constrained = np.max (average_constrained)
            aux = np.concatenate (
                np.where (average_constrained == max_average_constrained)
                )
            index = tuple (
                np.concatenate (
                    [aux [:2], [ptt_card_enum], [wks_enum], aux [2:]]
                    )
                )
            best_parameter = Parameters [index]    
            assert (best_parameter [2] == ptt_card)
            assert (best_parameter [3] == wks)
            if with_accuracy:
                max_per_k_sharing [(wks,ptt_card)] =\
                    [best_parameter, max_average_constrained]
            else: 
                max_per_k_sharing [(wks,ptt_card)] =\
                    best_parameter
    return max_per_k_sharing

def fill_parameters_dict (
    Nt,  Best_parameters_dict, weight_sharing_split = False,
    classifier = False):
    """

    'fill_parameters_dict'



    Parameters
    ----------
    Nt : int, 
        Number of layers in the PSBC.
    Best_parameters_dict : dict,
        Dictionary with best parameters.
    weight_sharing_split : {bool, False}, optional,
        If True, then parameters for training are extended and stored in 
        two dictionaries, one for  layers-1-sharing, other for layers-Nt-sharing.
        If False, then all parameters, regardless of layers-k-shared architecture, 
        are all in the same dictionary.
        (This is not fully necessary, but was first implemented to make
        batch sizes on jupyter notebooks smaller.)
    classifier : {bool, False}, optional,
        If True, then the function creates a grid with parameters to be used 
        during training, extending the hyperparameters at eps = 0 to other values
        of eps.   

    Returns
    -------

    """
    #name_parameters = Best_parameters_dict [tuple( -np.ones (6))]
    parameter_filled_dict_1 = {}
    parameter_filled_dict_Nt = {}
    parameters ={}
    for key in Best_parameters_dict.keys ():
        if len (key) !=2:
            pass
        else:
            par_aux = Best_parameters_dict [key]
            parameters ["eps_range"] = np.asarray (
                np.r_[[0], np.power (.5, [4,3,2,1,0])], dtype = np.float32)
            if classifier:
                parameters ["eps_range"] = np.asarray ([0], dtype = np.float32)
            parameters ["dt_range"] = np.asarray (
                [par_aux [1]], dtype = np.float32)
            parameters ["ptt_range"] = np.asarray (
                [par_aux [2]], dtype = np.uint16)
            parameters ["layer_share_range"] = np.asarray (
                [par_aux [3]], dtype = np.uint16)
            parameters ["lr_U_range"] = np.asarray (
                [par_aux [4]],dtype = np.float32)
            parameters ["lr_P_range"] = np.asarray (
                [par_aux [5]], dtype = np.float32)
    
            if weight_sharing_split:
                if key[0] == 1:
                    parameter_filled_dict_1 [key] = copy.deepcopy (parameters)
                else:
                    parameter_filled_dict_Nt [key] = copy.deepcopy (parameters)
            else:
                parameter_filled_dict_1 [key] = copy.deepcopy (parameters)        
        
    return parameter_filled_dict_1, parameter_filled_dict_Nt

def prepare_train_test_set (variable_0, variable_1, level = 2):
    """
    The function 'prepare_train_test_set' 
    relies on the folder structure of the datafiles, 
    and must be called from vary_Nt or Vary_eps.

    Parameters
    ----------
    variable_0 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    variable_1: int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    level : {int, 2}, optional, 
            This is used to signal which part of the folder structure
            one is. It indicates the number of layers below 
            that of folder MOTHER_PSBC (considered as layer 0).
    Returns
    -------
    
    """
    S = select_split_pickle (level = level)
    ## retrieve non-shuffled train data
    X_train, Y_train, average = S.select_variables_from_pickle (
        variable_0, variable_1, averaged = True)
    # Shuffle data
    p = X_train.shape[0]
    shuffle_train = np.arange (p)
    np.random.shuffle(shuffle_train)
    X_train, Y_train = X_train [shuffle_train], Y_train [shuffle_train]

    X_test, Y_test, _ = S.select_variables_from_pickle (
        variable_0, variable_1,type_of = "test", averaged = False
    )
    X_test = X_test - average
    
    return X_train, Y_train, X_test, Y_test, average

class select_split_pickle ():
    """
    The class 'select_split_pickle' is used for data retrieval: 
    given a pair of indexes, it retrieves either train or test subsets
    associated with such labels.
    It also offers the possibility of averaging the data and balancing it or not.

    It relies on the folder structure of the datafiles, 
    for it accesses pickled datasets.

    """
    
    def __init__ (self,level = 2):
        """
        Initializer.

        Parameters
        ----------
        level : {int, 2}, optional, 
            This is used to signal which part of the folder structure
            one is. It indicates the number of layers below 
            that of folder MOTHER_PSBC (considered as layer 0).
        """
    
        self.level = level
        
    def select_variables_from_pickle (
        self, variable_0, variable_1, averaged = False, type_of = "train"):
        
        self.variable_0 = variable_0
        self.variable_1 = variable_1
        append = "../" * self.level
        
        file_name_0 = append + "Pickled_datasets/X_" + \
            type_of + "_"+ str (variable_0) + ".p"
        file_name_1 = append + "Pickled_datasets/X_" + \
            type_of + "_"+ str (variable_1) + ".p"
        

        with open (file_name_0, 'rb') as pickled_dic:
            X_0  = pickle.load (pickled_dic)
        with open (file_name_1, 'rb') as pickled_dic:
            X_1  = pickle.load (pickled_dic)

        X_all = np.r_[X_0, X_1]

        average = None
        if averaged:
            average = np.mean (X_all, axis = 0)
            X_all = X_all - average

        Y_all = np.r_[np.zeros (X_0.shape [0]), np.ones (X_1.shape [0])]
        
        return X_all , Y_all, average

def fitting_several_models(
    cv, parallel, cpu, X_train,
    Y_train, X_test, Y_test, parameters_model,
    Nx, Neumann, EPOCHS , patience, Nt,
    train_dt_U, train_dt_P,with_PCA, V_star,
    save_best_only, append_to_saved_file_name = "", save_history = False,
    subordinate = True, with_phase = True, 
    batch_size = 32, normalization = True
    ):
    """
    'fitting_several_models' is responsible for
    creating and fitting several PSBC models with certain parameters, 
    where in each of them the model is fitted cv times.
    It is useful for assessment of statistical properties of the model.

    Parameters
    ----------
    cv : int, 
        Number of cross validations.
    parallel : bool, 
        If True then molde is trained in parallel, or serially otherwise.
    cpu : int, 
        Number of cores to be used in case of parallel computing.
    X_train : numpy.array or tensor of size N_data_train X N_features,
        Matrix with features. 
    Y_train : numpy.array
        Array with labels.
    X_test : numpy.array or tensor of size N_data_test X N_features,
        Matrix with features. 
    Y_test : numpy.array
        Array with labels.
    parameters_model : dict,
        Dictionary with parameters for PSBC's construction.
    Nx : int, 
        Number of features.
    Neumann : bool,
        If True, boundary conditions are of Neumann type, otherwise they are
        of Periodic type.
    EPOCHS : int, 
        Number of epochs.
    patience:  int, 
        Patience used in Early Stopping.
    Nt : int, 
        Number of layers in the PSBC.
    train_dt_U : bool,
        If True, then dtu is a trainable weight, non-trainable otherwise.
    train_dt_P : bool,
        If True, then dtp is a trainable weight, non-trainable otherwise.
    with_PCA : bool,
        If True, then the model searches for a basis matrix
        that is non-canonical, usin PCA. This matrix is provided.
    V_star : None or a numpy matrix,
        This is the basis matrix, only accessed if with_PCA is True.
    save_best_only : bool,
        If True, then the model saves the best parameter's weight at the 
        epoch with highest accuracy; no weight is saved if False.
    append_to_saved_file_name  : {str, ""}, optional,
        Extension to be added to saved file name. 
        Useful for statistical purposes, like standardization of outputs.
    save_history : {bool, False}, optional,
        If True, then history during training is saved, non-saved otherwise.
    subordinate : {bool, True}, optional,
            If True, then the model is subordinate, non-subordinate otherwise.
    with_phase : {bool, True}, optional,
        If True, then the model has a coupled phase equation,
        otherwise it does not have one.
    batch_size : {int,32}, optional,
        Batch_size, for minibatch gradient descent purposes.
    normalization : {bool, True}, optional.
        If True, then normalization in the for 
           X - > 0.5  + 0.5 * X
        is applied (in which case X is assumed to have been centralized),
        otherwise no normalization is performed.
        
    
    Returns
    -------

    """
    #####################################################################
    ### BEGIN PARALLEL PROCESSING
    results = []
    if parallel:
        print ("\nRUNNING THE MODEL IN PARALLEL")
        pool = mp.Pool (cpu)
        for i in range (cv):
            args_now = (
                i, X_train, Y_train, X_test, Y_test,\
                    parameters_model, Nx, Neumann, EPOCHS , patience,\
                        Nt, train_dt_U, train_dt_P, with_PCA,\
                            V_star, save_best_only, append_to_saved_file_name,\
                                save_history, subordinate, with_phase,
                                batch_size, normalization
                            )
            results.append(
                pool.apply_async(
                    my_gridSearch_with_index,
                    args = args_now))
        # results is a list of pool.ApplyResult objects

        print (results)
        all_results = [r.get () for r in results]
        pool.close ()
        pool.join ()
        #pool.clear ()
    else:
        print ("\nRUNNING THE MODEL SERIALLY")
        for i in range (cv):
            
            results.append(
                my_gridSearch_with_index (
                    i, X_train, Y_train,\
                    X_test, Y_test, parameters_model, Nx, Neumann,\
                        EPOCHS , patience, Nt, train_dt_U, train_dt_P,\
                            with_PCA, V_star, save_best_only,
                            append_to_saved_file_name,
                            save_history, subordinate, with_phase,
                            batch_size, normalization
                            )
            )
        # results is a list of pool.ApplyResult objects
        all_results = results     
    return all_results

#####################################################################
### GRID SEARCH
#####################################################################

##### CREATING GRIDS!!!

def create_grid_for_search (Nt, grid_type):

    """
    'create_grid_for_search' creates a dictionary with grids where
     PSBC's hyperparameters lie. Parameters in these dictionaries are:
    
        * viscosity ("eps_range")
        *  parameterization cardinality ("ptt_range")
        * layers-k-share ("layer_share_range")
        * learning rates for U ("lr_U_range")
        * learning rates for P  ("lr_P_range")
        * dt parameters ("dt_range"; we remark that both
                        dtu and dtp are initialized with same value
    
    Different grids are given, accordind to the grid_type
    passed to the function.

    Parameters
    ----------
    Nt : int, 
        Number of layers in the PSBC.
    grid_type : str, in {"all", "vary_eps", "vary_Nt", "classifier"},
        Indicates which type of grid search should be returned.

    Returns
    -------
    Dictionary with grid.
    """
    
    Nx = 784 
    parameters_model = {}
    if Nt == 1: 
        parameters_model ["layer_share_range"] = [1]
    else: 
        parameters_model ["layer_share_range"] = [1, Nt]
    parameters_model ["lr_U_range"] = np.asarray (
        [1e-3, 1e-2, 1e-1], dtype = np.float32)
    parameters_model ["lr_P_range"] = np.asarray (
        [1e-3, 1e-2, 1e-1], dtype = np.float32)
    parameters_model ["dt_range"] = np.asarray (
        [.2], dtype = np.float32)

    if grid_type == "all":
        parameters_model ["eps_range"] = np.asarray (
            np.r_[[0], np.power (.5, np.arange (10, -2, -1))],\
                dtype = np.float32)
        parameters_model ["ptt_range"] = [int (Nx/k) for k in range (1,11)]
    elif grid_type == "vary_eps":
        parameters_model ["eps_range"] = np.asarray (
            np.r_[[0], np.power (.5,np.power (.5, [4,3,2,1,0])) ],\
                dtype = np.float32
            )
        parameters_model ["ptt_range"] = [int (Nx / 4)]
    elif grid_type == "vary_Nt":
        parameters_model ["eps_range"] = np.asarray ([0], dtype = np.float32)
        parameters_model ["ptt_range"] = [int (Nx / k) for k in [1, 2, 4, 8, 16]]
    elif grid_type == "classifier":
        parameters_model ["eps_range"] = np.asarray ([0], dtype = np.float32)
        parameters_model ["ptt_range"] = [int (Nx / 4)]
        parameters_model ["layer_share_range"] = [1]
    
    return parameters_model


def my_gridSearch_with_index (
    i, X_train_grid, Y_train_grid, X_test_grid, Y_test_grid, parameters,
    Nx = 784, Neumann = True, EPOCHS = 1, patience = 5, Nt = 1,
    train_dt_U = True, train_dt_P = True, 
    with_PCA = False, V_star = None, save_best_only = False,
    append_to_saved_file_name = "", save_history = False,
    subordinate = True, with_phase = True,
    batch_size = 32, normalization = True
):
    """

    'my_gridSearch_with_index' is used for grid search purposes. 
    It is convenient in the sense that the model is trained on 
    (X_train_grid, Y_train_grid) and tested on (X_test_grid, Y_test_grid).
    It also supports the case of new basis matrices, which can be passed to 
    the function throught the parameter V_star.
    Several options for saving and name of outrput files are given.

    Parameters
    ----------
    i : int,
        Index, used for book keeping purposes during cross validation.
    X_train_grid : numpy.array of size N_data X N_features
        numpy array with features for train set.
    Y_train_grid : numpy.ndarray of size N_data X 1
        Array with labels for train set.
    X_test_grid : numpy.array of size N_data X N_features
        numpy array with features for test set.
    Y_test_grid : numpy.ndarray of size N_data X 1
        Array with labels for test set. 
    parameters : dict,
        Dictionary with parameters for PSBC's construction.
    Nx : {int, 784}, optional,
        Number of features
    Neumann : {bool, True}, optional,
        If True, boundary conditions are of Neumann type, otherwise they are
        of Periodic type.
    EPOCHS : {int, 1}, optional,
        Number of epochs.
    patience:  int, 
        Patience used in Early Stopping.
    Nt : {int, 1}, optional,
        Number of layers in the PSBC.
    patience : {int, 5}, optional,
        Patience parameter for Early stopping purposes.
    train_dt_U : {bool, True}, optional,
        If True, then dtu is a trainable weight, non-trainable otherwise.
    train_dt_P : {bool, True}, optional,
        If True, then dtp is a trainable weight, non-trainable otherwise.
    with_PCA : bool,
        If True, then the model searches for a basis matrix
        that is non-canonical, usin PCA. This matrix is provided.
    V_star : None or a numpy matrix,
        This is the basis matrix, only accessed if with_PCA is True.
    save_best_only : {bool, False}, optional,
        If True, then the model saves the best parameter's weight at the 
        epoch with highest accuracy; no weight is saved if False.
    append_to_saved_file_name  : {str, ""}, optional,
        Extension to be added to saved file name. 
        Useful for statistical purposes, like standardization of outputs.
    save_history : {bool, False}, optional,
        If True, then history during training is saved, non-saved otherwise.
    subordinate : {bool, True}, optional,
        If True, then the model is subordinate, non-subordinate otherwise.
    with_phase : {bool, True}, optional,
        If True, then the model has a coupled phase equation,
            otherwise it does not have one.
    batch_size : {int,32}, optional,
        Batch_size, for minibatch gradient descent purposes.
    normalization : {bool, True}, optional,
        If True, then normalization in the for 
           X - > 0.5  + 0.5 * X
        is applied (in which case X is assumed to have been centralized),
        otherwise no normalization is performed.
        
    Returns
    -------

    """
    print ("\nFixed hyperparameters\n")
    print ("Nx :", Nx, "\tNt :", Nt,"\tNeumann :", Neumann,
         "\tpatience :", patience,
         "\ttrain_dt_U :", train_dt_U, "\ttrain_dt_P :", train_dt_P,
         "\n\twith_PCA :", with_PCA)
                                
    I = Initialize_parameters ()
    
    Param_values = {
        (-1,-1,-1,-1, -1,-1):\
            ("eps", "dt", "pt_cardnlty", "layers_K_shared", "lr_U", "lr_P")
    }
    
    ### Grid ranges
    eps_range = parameters ["eps_range"]
    dt_range = parameters ["dt_range"]
    ptt_range = parameters ["ptt_range"]
    layer_share_range = parameters ["layer_share_range"]
    lr_U_range = parameters ["lr_U_range"]
    lr_P_range = parameters ["lr_P_range"]   

    Accuracies = np.zeros ((
        len (eps_range),
        len (dt_range),
        len (ptt_range),
        len (layer_share_range),
        len (lr_U_range), 
        len (lr_P_range)        
    ))
    print (
        "\n\n\tWe will fit the model", np.product(Accuracies.shape),"times\n\n"
        )
    print_flag = True

    if normalization:
        val_data = (.5 + .5*X_test_grid, Y_test_grid)
    else: 
        val_data = (X_test_grid, Y_test_grid)

    now_fitting = 0
    for enum_eps, eps in enumerate (eps_range):
        for enum_dt, dt in enumerate (dt_range):
            for enum_ptt, pt_cardnlty in enumerate (ptt_range):
                for enum_wk_sh, layers_K_shared in enumerate (layer_share_range):
                    for enum_lr_U, lr_U in enumerate (lr_U_range):
                        for enum_lr_P, lr_P in enumerate (lr_P_range):

                            print ("\nVarying parameters time \n", now_fitting)
                            now_fitting +=1
                            print (
                                "eps :", eps, "\tdt :", dt, "\tptt_cardnlty :",\
                                    pt_cardnlty, "\tlayers_K_shared :",\
                                        layers_K_shared, "\tlr_U :", lr_U,\
                                            "\tlr_P :", lr_P
                                            )
                            
                            Param_values [(
                                enum_eps, enum_dt,
                                enum_ptt, enum_wk_sh, enum_lr_U, enum_lr_P
                                )] = (eps, dt, pt_cardnlty,\
                                    layers_K_shared, lr_U, lr_P)
                        
                            if with_PCA:
                                basis = V_star[: pt_cardnlty,:]
                                parameters = I.dictionary (
                                    Nx, eps, Nt, pt_cardnlty,
                                    layers_K_shared, dt = dt,
                                    Neumann = Neumann, 
                                    basis = basis, retrieve_basis = True
                                    )
                            else: 
                                parameters = I.dictionary (
                                    Nx, eps, Nt, pt_cardnlty,
                                    layers_K_shared, dt = dt,
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
                                parameters, train_dt_U = train_dt_U,
                                train_dt_P = train_dt_P,
                                lr_U = lr_schedule_U,
                                lr_P= lr_schedule_P, 
                                subordinate = subordinate,
                                with_phase = with_phase,
                                with_PCA = with_PCA,
                                loss = keras.losses.mean_squared_error,
                                metrics = [classify_zero_one],
                                print_summary = print_flag,
                                validation_data = val_data
                                )
                            
                            #### If saving weights 
                            name = str (enum_eps) + "_" + str (enum_dt) + "_"\
                                + str (pt_cardnlty) + "_" + str (layers_K_shared) + "_"\
                                    + str (enum_lr_U) + "_" + str (enum_lr_P)
                            filepath = "weights/" + name + "_fold_" + str (i) +\
                                append_to_saved_file_name 
                            
                            if save_best_only:
                                ##------- FOR MODEL CHECKPOINT ---------------------
                                print ("Save best model as", filepath + ".h5")
                                try: os.mkdir ("weights")
                                except: pass
                            else:
                                print ("Not saving best model")
                                
                            print ("\nfilepath", filepath)
                            callback_early = EarlyStoppingAtMax_acc_and_save(
                                metric = classify_zero_one,
                                filename = filepath,
                                save_best = save_best_only, 
                                patience = patience
                                )

                            if normalization:
                                history = model.fit(
                                    .5 + .5*X_train_grid, Y_train_grid,
                                    epochs = EPOCHS,
                                    batch_size = batch_size,
                                    callbacks = [callback_early]
                                )
                            else:
                                history = model.fit(
                                    X_train_grid, Y_train_grid,
                                    epochs = EPOCHS,
                                    batch_size = batch_size,
                                    callbacks = [callback_early]
                                )

                            if save_history:
                                print ("Saving the model's history as", filepath + "p")
                                try: os.mkdir ("history")
                                except: pass

                                filepath_history = "history/" + name + "_fold_" + str (i) +\
                                append_to_saved_file_name +".p"
                                
                                with open (filepath_history, 'wb') as save:
                                    pickle.dump (history.history, save,\
                                        protocol = pickle.HIGHEST_PROTOCOL)        
                                
                                print ("History pickled to ", filepath_history)
                            
                            max_acc = np.max(model.validation_accuracy)
                            Accuracies [enum_eps,enum_dt, enum_ptt,
                                        enum_wk_sh, enum_lr_U, enum_lr_P] = max_acc
                            print ("Maximal accuracy was", max_acc)
                            print_flag = False

    return i, Accuracies, Param_values


def mini_batches(X, Y, batch_size = 32):
    """
    'mini_batches' function.

    This function splits the data (X, Y) in 'number_mini_batches' parts.

    Parameters
    ----------
    X : numpy.array of size N_data X N_features
        numpy array with features. 
    Y : numpy.ndarray of size N_data X 1
        Array with labels. 
    batch_size : {int, 1}, optional
        Number of elements in which (X,Y) will be split.

    Returns
    -------
    A list 'minibatches' of tuples with 'number_mini_batches' elements, 
    where the elements makeup a partition of the dataset (X,Y).

    minibatches : list.
    """
    
    full_batch_size = X.shape [1]
    number_mini_batches = int (full_batch_size / batch_size)
    splitting = np.array_split (
        np.random.permutation (full_batch_size), number_mini_batches)
    
    mini_batches = []
    
    for i in range (number_mini_batches):
        X_mini_batch_now = X [:, splitting [i]]
        Y_mini_batch_now = Y [:, splitting [i]]
        ## Create list
        mini_batches.append ((X_mini_batch_now, Y_mini_batch_now))
    
    return mini_batches


def select_variables (X, Y, variable_1, variable_2, averaged = False):
    """
    'select_variables' is a function used to retrieve the subset of sample data
    (X,Y) with labels variable_1, variable_2. 
    The user can also choose whether the output file is averaged. 

    Parameters
    ----------
    X : numpy.array of size N_data X N_features
        numpy array with features. 
    Y : numpy.ndarray of size N_data X 1
        Array with labels. 
    variable_1 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    variable_2 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    averaged : {bool, False}, optional,
        If True, then the output data is averaged and centered 
        (that is, the output has average zero). 
        Otherwise, no centering is performed and the algorithm 
        returns None for this variable.
    Returns
    -------

    """
    Y_int = Y.astype ("uint8")
    select_var_1 = np.where (Y_int == variable_1) [0]
    select_var_2 = np.where (Y_int == variable_2) [0]
    X_all = np.r_[X [np.ravel (select_var_1), :], X [np.ravel (select_var_2), :]]
    
    average = None
    if averaged:
        average = np.mean (X_all, axis = 0)
        X_all = X_all - average
        
    Y_all = np.r_[0 * Y_int [select_var_1], np.ones(len (Y_int [select_var_2]))]
    
    p = np.arange (len (Y_all))
    np.random.shuffle(p)
    
    return X_all [p, :].astype (np.float32), Y_all [p], average
    

def evaluate_model (
    layers_k_shared, pt_cardnlty, Nt, variable_0, variable_1,
    best_parameters,  with_PCA = False, Neumann = True, 
    train_dt_U = True, train_dt_P = True,
    classifier = False, subordinate = True, with_phase = True
):
    """
    'evaluate_model' was designed to evaluate trained models on the test set.
    It relies on the folder structure of the datafiles, 
    for it accesses trained weights.

    Parameters
    ----------
    layers_k_shared,
    pt_cardnlty : {int, None}, optional    
            Parameterization cardinality.
    Nt : int, 
        Number of layers in the PSBC.
    variable_0 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    variable_1 : int, 
        Integer in the range {0,...,9}, corresponding to the label
        of a digit in the MNIST dataset.
    best_parameters :

    with_PCA : {bool, False}, optional,
        If True, then the model searches for a basis matrix
        that is non-canonical, usin PCA. This matrix is provided.
    Neumann : {bool, True}, optional,
        If True, boundary conditions are of Neumann type, otherwise they are
        of Periodic type.
    train_dt_U : {bool, True}, optional,
        If True, then dtu is a trainable weight, non-trainable otherwise.
    train_dt_P : {bool, True}, optional,
        If True, then dtp is a trainable weight, non-trainable otherwise.
    classifier : {bool, False}, optional,
        If True, then a substring 'classifier' is added to the saved weights,
        and other saved data, indicating the digits that are being trained.
    subordinate : {bool, True}, optional,
        If True, then the model is subordinate, non-subordinate otherwise.        
    with_phase : {bool, True}, optional,
            If True, then the model has a coupled phase equation,
            otherwise it does not have one.
        
    Returns
    -------

    """
    

    ### Opening weights' training results
    if classifier:
        append_to_saved_file_name = "_var0_" + str (variable_0) + "_var1_" + str (variable_1)

        if with_PCA:
            append_to_saved_file_name = "_PCA_" + append_to_saved_file_name
    else: 
        append_to_saved_file_name= "_Index_" + str (layers_k_shared) +\
        "_"+ str (pt_cardnlty) + "_" + str (Nt)
    
    ### Setting up filepath for pickled file where data will be stored
    if classifier:
        file_name = "Training_accuracies_" + str (layers_k_shared) +\
            "_"+ str (pt_cardnlty) + "_" + str (Nt) +"_classifier_"\
                + str (variable_0) + "_" + str (variable_1) +".p"
    else:
        file_name = "Training_accuracies_" + str (layers_k_shared) +\
            "_"+ str (pt_cardnlty) + "_" + str (Nt) +"_vary_eps_"\
                + str (variable_0) + "_" + str (variable_1) +".p"
            
    if with_PCA:
        file_name = "PCA_" + file_name
    
    file_name = "training/" + file_name

    with open (file_name, 'rb') as pickled_dic:
        Training_data  = pickle.load (pickled_dic)
    

    # Training data is a list with Accuracies and Parameters.
    # Accuracies repeat the paramters due to cross validations or repetitions

    Accuracies_test, Parameters_test = Training_data
    cv_range, eps_range, dt_range,\
    ptt_range, layer_share_range, lr_U_range, lr_P_range = Accuracies_test.shape
    Nx = 784
    
    ### X_test is retrieved. The average of X_train has been subtracted of X_test
    _, _, X_test, Y_test, _ =\
        prepare_train_test_set (variable_0, variable_1, level = 2)

    
    I = Initialize_parameters ()
    PSBC_loader = load_PSBC_model ()
    
    for enum_cv_range in range (cv_range):
        for enum_eps in range (eps_range):
            for enum_dt in range (dt_range):
                for enum_ptt in range (ptt_range):
                    for enum_wk_sh in range (layer_share_range):
                        for enum_lr_U in range (lr_U_range):
                            for enum_lr_P in range (lr_P_range):
                                eps, dt,  pt_cardnlty,\
                                    layers_K_shared,lr_U, lr_P =\
                                        Parameters_test [(
                                            enum_eps, enum_dt,
                                            enum_ptt, enum_wk_sh, enum_lr_U, enum_lr_P
                                            )]
                                
                                parameters_now = I.dictionary (
                                    Nx, eps, Nt, pt_cardnlty,
                                    layers_K_shared, dt = dt,
                                    Neumann = Neumann)
                                
                                model = PSBC_build_model (
                                    parameters_now, train_dt_U = train_dt_U,
                                    train_dt_P = train_dt_P,
                                    lr_U = lr_U,
                                    lr_P = lr_P,
                                    subordinate = subordinate,
                                    with_phase = with_phase,
                                    with_PCA = with_PCA,
                                    loss = keras.losses.mean_squared_error,
                                    metrics = [classify_zero_one])
                                
                                filepath = str (enum_eps) + "_" + str (enum_dt) +"_"\
                                + str (pt_cardnlty) + "_" + str (layers_K_shared) +"_"\
                                    + str (enum_lr_U) + "_" + str (enum_lr_P)
                                filepath = "weights/" + filepath + "_fold_"\
                                    + str (enum_cv_range) +\
                                append_to_saved_file_name +"_best_weights.h5"
                                
                                PSBC_loader.load_model_with_layers (model, filepath )
                                
                                ### DON"T FORGET TO SCALE THE DATA!!!
                                y_pred = model.predict (.5 + .5 * X_test) 
                                
                                accuracy_now = classify_zero_one (y_pred, Y_test).numpy ()

                                print ("\n\nAccuracy of this model on training set",\
                                    Accuracies_test [enum_cv_range,enum_eps,enum_dt, enum_ptt,\
                                        enum_wk_sh,enum_lr_U, enum_lr_P],\
                                            "\nAccuracy of this model on test set",\
                                                accuracy_now, "\n")
                                print ("------------------------------------------\n\n")
                                Accuracies_test [\
                                    enum_cv_range,enum_eps,enum_dt, enum_ptt,\
                                        enum_wk_sh,enum_lr_U, enum_lr_P] =\
                                            accuracy_now

    ### Saving testing results
    if classifier:
        save_name = "Test_set_accuracies_" + str (layers_k_shared) + \
            "_" + str (pt_cardnlty) + "_" + str (Nt) + \
                "_classifier_" + str (variable_0) + "_" + str (variable_1) + ".p"
    else:
        save_name = "Test_set_accuracies_" + str (layers_k_shared) + \
        "_" + str (pt_cardnlty) + "_" + str (Nt) + \
            "_vary_eps_" + str (variable_0) + "_" + str (variable_1) + ".p"

    if with_PCA:
        save_name = "PCA_" + save_name

    save_name = "training/"+save_name
    
    with open (save_name, 'wb') as save:
        pickle.dump ((Accuracies_test, Parameters_test), save,\
            protocol = pickle.HIGHEST_PROTOCOL)        
        print ("Statistics pickled to ", save_name)
