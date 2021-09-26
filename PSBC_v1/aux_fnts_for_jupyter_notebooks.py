#!/usr/bin/env python
""" Python library for the paper 
'Binary Classification as a Phase Separation process'.
This module contains an implementation of the auxiliar functions used in the jupyter-notebooks for the paper. 
 """
 
__author__ = "Rafael de Araujo Monteiro"
__affiliation__ =\
    "Mathematics for Advanced Materials - Open Innovation Lab,"\
        +"\n (Matham-OIL, AIST),"\
            +"\n Sendai, Japan"
__copyright__ = "None"
__credits__ = ["Rafael Monteiro"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Rafael Monteiro"
__email__ = "rafael.a.monteiro.math@gmail.com"
__date__ =  "September, 2020"

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
###    data : September, 2020
###
#####################################################################
### IMPORT LIBRARIES
#####################################################################
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
import  matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import scipy.sparse as sc
import sympy
import itertools as it

###-----------------------------------
## The modules for this paper are here
from  binary_phase_separation import *
###-----------------------------------


### In order to open ans save dictionaries
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import pandas as pd
import warnings
warnings.filterwarnings(action = "ignore", message = "internal issue")

## Things necessary to do nice plots
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from  matplotlib.transforms import Affine2D

# where we are saving stuff
import os, glob
PROJECT_DIR =  os.getcwd()#os.path.dirname(os.path.realpath(__file__))
IMAGES = os.path.join(PROJECT_DIR, "figures/")
PROJECT_DIR = os.path.join(PROJECT_DIR)

################################################################################
### AUXILIAR FUNCTIONS
################################################################################

## A save figures function.
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
    print("Saving figure as ", figure_name.replace(" ", "_"))
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = figure_extension, dpi = resolution)

def accuracies (
    parameters, name, accuracy_type,\
        number_folders = 10, number_simulations = 10):
    """
    This function is only used in the jupyter notebook for the MNIST dataset
    
    
    Parameters
    ----------
    parameters : dictionary
        Dictionary containing summary of data for some PSBC experiments.
    name : string
        Name of the keys of the dictionary "parameters" that we are studying, corresponding to a PSBC configuration.
    accuracy_type : string
        Either "best_accuracy_train" or "best_accuracy_test".
    number_folders : {int, 10}, optional
        Number of folders, where each folder corresponds of one value of the parameter being valued.
    number_simulations : {int, 10}, optional
        Number of simulations that were run with the same parameter, for statistical purposes.
    
    Returns
    -------
    A : matrix
        Matrix with all the accuracies of type accuracy_type, 
        where each row has all the simulations for a certain parameter
    value_of_parameter_varying : list 
        list with parameters values.
    """

    param_now = parameters[name]
    value_of_parameter_varying = []
    A = np.zeros([number_folders, number_simulations])
    for i in range(1, number_folders + 1):
        
        ## read results for all "number_simulations" simulations with that parameter
        v = np.reshape(param_now[str(i)][accuracy_type], (1, number_simulations))
        
        ## Save values of parameters
        value_of_parameter_varying.append(param_now[str(i)]["value_of_parameter_varying"])
        A[i-1,:] = v
            
    return A, value_of_parameter_varying