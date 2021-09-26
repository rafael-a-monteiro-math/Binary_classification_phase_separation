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
import scipy as scp
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
import multiprocess as mp
warnings.filterwarnings (action = "ignore", message = "internal issue")

############################################################################
from tfversion_binary_phase_separation import *
from tf_PSBC_extra_libs_for_training_and_grid_search import *
############################################################################

class PSC_ensemble (keras.Model):
    """
    'PSC_ensemble' is a keras.Model that uses ensemble learning techniques to
    implements a classifier associated with  a pair (a,b) of
    distinct digits in  {0,...,9}.
    It relies on the folder structure of the datafiles, 
    for it accesses trained weights.
    """
    
    def __init__ (
        self, variable_0, variable_1, how_many = 5, 
        ptt_value = 196, with_PCA = False,
        append_to_saved_file_name = "",
        **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        variable_0 : int, 
            Integer in the range {0,...,9}, corresponding to the label
            of a digit in the MNIST dataset.
        variable_1 : int, 
            Integer in the range {0,...,9}, corresponding to the label
            of a digit in the MNIST dataset.    
        how_many : {int, 5}, optional, 
            Number of times that the models are fitted (for statistics).
        ptt_value : {int, 196}, optional,
            parameterization cardinality. In this case it is fixed.
        with_PCA : {bool, False}, optional,
            If True, then the model searches for a basis matrix
            that is non-canonical, usin PCA. This matrix is provided.
        append_to_saved_file_name  : {str, ""}, optional,
            Extension to be added to saved file name. 
            Useful for statistical purposes, like standardization of outputs.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.I = Initialize_parameters ()
        self.load = load_PSBC_model ()
        self.S = select_split_pickle (level = 2)
        self.L = load_PSBC_model ()
        self.all_models = []
        self.all_averages = []
        self.variable_0 = variable_0
        self.variable_1 = variable_1
        self.ptt_value = ptt_value
        
        self.with_PCA = with_PCA
        self.append_to_saved_file_name = append_to_saved_file_name
        Neumann = True
        Nt = 2
        filename = "grid_search_"+str (Neumann)+"_"+str (Nt)+".p"
        ### RETRIEVE PARAMETERS
        with open ("../../Grids/"+filename, 'rb') as pickled_dic:
            self.grid_range  = pickle.load (pickled_dic)
            
        self.cv = min(how_many, self.grid_range ["cv"])
        print ("Number of grids", self.cv)
        self.load_all_models ()
            
    def load_all_models (self):
        """
        Loads weights associated to a pair of digits.

        Returns
        -------
        Nothing is returned.
        """
        
        print ("Loading models")
        Neumann = self.grid_range ["Neumann"]
        patience = self.grid_range ["patience"]
        Nt = self.grid_range ["Nt"]
        Nx = 784
        train_dt_U = self.grid_range ["train_dt_U"]
        train_dt_P = self.grid_range ["train_dt_P"]
        
        for i in range(self.cv):
            
            retrieve_best_par = BestPararameters_ptt_card_weights_k_shared_fixed (
                    Nt, self.variable_0,  self.variable_1,
                    classifier = True, with_PCA = self.with_PCA,
                    ptt_value=self.ptt_value)

            eps, dt, pt_cardnlty, layers_K_shared, lr_U, lr_P =\
                retrieve_best_par [(1,self.ptt_value)]

            parameters = self.I.dictionary (
                            Nx, eps, Nt, pt_cardnlty,
                            layers_K_shared, dt = dt,
                            Neumann = Neumann)

            lr_schedule_U = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr_U, decay_steps = 5,
                decay_rate = 0.99, staircase = True)

            lr_schedule_P = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr_P, decay_steps = 5,
                decay_rate = 0.99, staircase = True)

            self.all_models.append(PSBC_build_model (
                parameters, train_dt_U = True, train_dt_P = True,
                lr_U = lr_schedule_U, lr_P = lr_schedule_P, 
                with_PCA= self.with_PCA,
                loss = keras.losses.mean_squared_error,
                metrics = [classify_zero_one], print_summary = (i == 1) ))
            
            
            filepath = "weights/0_0_" + str (self.ptt_value) + "_1_0_0_fold_" + str (i) 
            if self.with_PCA:
                filepath = filepath + "_PCA_"
            
            filepath = filepath + "_var0_"+str (self.variable_0)+"_var1_"+\
            str (self.variable_1)+"_best_weights.h5"

            filepath = self.append_to_saved_file_name + filepath
            
            self.L.load_model_with_layers (self.all_models [-1], filepath)

            _,_, average = self.S.select_variables_from_pickle (
                self.variable_0, self.variable_1, averaged = True)

            self.all_averages = average
            
    def call (self, X):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.

        """
        output = tf.Variable(
            tf.zeros((X.shape [0], self.cv)),\
                shape= (X.shape [0], self.cv), dtype=tf.float32
                )

        for i in range (self.cv):
            model_now = self.all_models [i]
            average_now = self.all_averages
            aux = model_now.predict_zero_one (.5 + .5 * (X - average_now))
            
            ### The next assignments are due tot he fact that variable_0 < variable_1
            output [:,i].assign (aux)
            
        return  output
    
    def hard_vote (self,X):
        """
        Given the array with committes' votes (for classes 0, or 1),
        decides wchich class as individual is assgned to based on majority vote.

        Returns
        -------
        An array of sze (batch size, ) with values in {0,1}.
        """
        output = self.call (X)
        return scp.stats.mode (output.numpy (), axis = 1) [0]


class PSBC_multi_class (keras.Model):
    """
    'PSBC_multi_class' is a keras.Model that implements a multiclass classifier
     based on the PSBC model. It relies on the folder structure of the datafiles, 
     for it accesses trained weights for each pair (a,b) of distinnct digits in 
     {0,...,9}.
    """

    def __init__ (
        self, ensemble = True,
        with_PCA = False, append_to_saved_file_name = "", **kwargs):        
        """
        Initializer.

        Parameters
        ----------    
        ensemble : {bool,True}, optional,

        with_PCA : {bool, False}, optional,
            If True, then the model searches for a basis matrix
            that is non-canonical, usin PCA. This matrix is provided.
        append_to_saved_file_name  : {str, ""}, optional,
            Extension to be added to saved file name. 
            Useful for statistical purposes, like standardization of outputs.
        
        **kwargs
        """

        super ().__init__ (**kwargs)
        self.I = Initialize_parameters ()
        self.load = load_PSBC_model ()
        self.S = select_split_pickle (level = 2)
        self.L = load_PSBC_model ()
        self.all_models = []
        self.ensemble = ensemble
        self.with_PCA = with_PCA
        self.append_to_saved_file_name = append_to_saved_file_name
        
        Neumann = True
        Nt = 2

        filename = "grid_search_" + str (Neumann) + "_" + str (Nt) + ".p"
        ### RETRIEVE PARAMETERS
        with open ("../../Grids/" + filename, 'rb') as pickled_dic:
            self.grid_range  = pickle.load (pickled_dic)
        with open ("../../Grids/digits_index.p", 'rb') as pickled_dic:
            self.grid_indexes  = pickle.load (pickled_dic)

        print ("Loading models")
        self.load_all_models ()
        
    def load_all_models (self):
        """
        Loads weights associated to all pairs of digits.

        Returns
        -------
        Nothing is returned.
        """
        
        how_many = 5 if self.ensemble else 1
        
        for index in range (45):
            variable_0, variable_1 =  self.grid_indexes [index]
            self.all_models.append(
                PSC_ensemble (
                    variable_0 = variable_0, variable_1 = variable_1,
                    how_many = how_many, with_PCA = self.with_PCA,
                    append_to_saved_file_name = self.append_to_saved_file_name)
            )
            
    def call (self, X):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        output = tf.Variable(
            tf.zeros ((X.shape [0], 10, 10)),\
                shape= (X.shape [0], 10, 10), dtype=tf.float32
                )

        for index in range(45):
            variable_0, variable_1  = self.grid_indexes [index]
            model_now = self.all_models [index]
            aux = np.ravel(model_now.hard_vote (X)) #.5 + .5 *(X - average_now))
            ### The next assignments are due tot he fact that variable_0 < variable_1
            output [:,variable_0, variable_1].assign (1 - aux)
            output [:,variable_1, variable_0].assign (aux)

        return  output

class tournament ():
    """
    The 'class_tournament' class is use for computation of votes. 
    It can  use two methods: hard voting or a sequence of tournaments, where
    the worst participants are excluded of the next round until a single set 
    of individuals with same number of votes is obtained. The label is chosen by
    uniform sampling over the latter set of individuals.

    (There are many references for hard_voting, but I couldn't
    find any for tournaments. I'm not aware if this is new. Nevertheless,
    the results are almost the same, with hard voting more accurate 
    by about 0.2% points.) 
    """
    
    def __init__ (self, hardvote_or_tournament = "hard_vote"):
        """
        Initializer.

        Parameters
        ----------    
        hardvote_or_tournament : {str, "hard_vote"}, optional,
            Type of voting, {"hard_vote","tournament"}.
        """
        self.index = np.arange (10)
        self.tags = tf.constant(np.arange (10), shape= (10,))
        self.hardvote_or_tournament = hardvote_or_tournament
    
    def aux_tournament (self, M):
        """
        Assign an individual to a class among the digits {0,...9} based on a
        voting matrix.

        Returns
        -------
        Array with the class an individual is assigned to.
        """
        ### M has size (batch_size, 10, 10)
        score = tf.reduce_sum (M, axis = -1)
        
        self.chosen =  - np.ones (M.get_shape () [0])
        min_score = tf.reduce_min (score, axis = -1, keepdims = True)
        max_score = tf.reduce_max (score, axis = -1, keepdims = True)
        
        if self.hardvote_or_tournament == "hard_vote":
            eliminated_ix = tf.where (score == max_score)        
            for i in range(M.get_shape() [0]):
                
                self.chosen [i] = np.random.choice (self.tags [score [i] == max_score [i]])
            return self.chosen
            
        eliminated_ix = tf.where (score == min_score)
        
        how_many_elim = tf.unique_with_counts (eliminated_ix [:,0]).count
        
        how_many_still_playing = tf.reduce_sum (tf.cast (score < 10, tf.int32), axis= -1)
        
        for a in eliminated_ix:
            if how_many_still_playing [a [0]] == how_many_elim [a [0]] and (self.chosen [a [0]] <0):
                self.chosen[a[0]] = np.random.choice(self.tags[score[a[0]] <10])
            else:
                M [a [0],:, a [1]].assign (tf.zeros (shape = (10,)))  ### remove points from lowest individual
                M [a [0], a[1], a [1]].assign (10)  ## remove it from tournment

        if any(self.chosen <0):
            return self.aux_tournament (M)
        else:
            return self.chosen        