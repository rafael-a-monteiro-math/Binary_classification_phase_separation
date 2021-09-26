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
## Things necessary to do nice plots
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from  matplotlib.transforms import Affine2D

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
