#!/usr/bin/env python
""" Python library for the paper 
'Binary Classification as a Phase Separation process'.
This module contains an new implementation of the PSBC method, besides auxiliary 
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
import glob
import sys
import os
try: ## In order to open and save dictionaries
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

warnings.filterwarnings (action = "ignore", message = "internal issue")

### Worthwhile reading
### https://keras.io/getting_started/intro_to_keras_for_researchers/
#####################################################################
### DIFFUSION CLASS
#####################################################################

class Diffusion:
    """
    The 'Diffusion' class creates finite differences matrices of 
    finite differences, either of Neumann or Periodic type.
    """
    def __init__ (self):
        """
        Class initializer. No returned value. 
        """
        pass
    
    def matrix(
            self, N, Neumann = True):
        """ 
        'matrix' method.
        
        This method returns a finite difference matrix of the 1D Laplacian, 
        as a matrix with size  N x N.
        Boundary Condition is either Neumann or of Periodic type.

        Parameters
        ----------
        N : int, 
            Size of finite difference matrix (N x N).
        Neumann : {bool, True}, optional
            Boundary condition.  If 'Neumann' is True, returns the 1D Laplacian
            discretized for Neumann Boundary conditions. 
            If 'Neumann' is False, return Periodic Boundary counditions.
        
        Returns
        -------
        A : numpy.ndarray
            Matrix of size N x N
        
        Examples
        --------
        """
        A = sc.diags ([1, -2, 1], [-1, 0, 1], shape = (N, N))
        A = np.asarray (A.todense ())
        if Neumann:
            A [0, 1], A [N-1, N-2] = 2,2
        else: #Periodic
            A [0, N-1], A [N-1, 0] = 1, 1

        return tf.transpose (tf.constant (A, dtype = tf.float32))

    def id_minus_matrix(
                self, Nx, eps, dt, Neumann = True):

            """ 
            'id_minus_matrix' method.
            
            This method returns a matrix 
            M = Id_N - (eps^2) * dt / (dx^2) * diffusion_matrix,
                where dx := sqrt(dt)  (default).
            where diffusion_matrix is the finite difference of the 1D Laplacian
            with size N x N matrix, and Boundary Conditions of either
            Neumann or Periodic type.

            Parameters
            ----------
            Nx : int, 
                Set paramters for the size of returned matrix (Nx x Nx)
            eps : float
                Diffiusion / Viscosity parameter. 
            dt : float
                Mesh grid size of time discretization 
            Neumann : {bool, True}, optional
                Boundary condition.  If 'Neumann' is True, returns the 1D Laplacian
                discretized for Neumann Boundary conditions. 
                If 'Neumann' is False, return Periodic Boundary counditions.

            Returns
            -------
            'M, inv_M, diffusion_matrix', where
            M : numpy.ndarray 
                Matrix of size N x N
            inv(M) : numpy.ndarray
                Matrix of size N x N
            diffusion_matrix : numpy.ndarray
                Matrix of size N x N
            
            Examples
            --------
            """
            M = tf.linalg.eye (Nx, dtype = tf.float32)
            if eps  ==  0 or Nx == 1:
                return M, M, tf.zeros ([Nx, Nx])
            
            dif_matrix = self.matrix (Nx, Neumann)
            M = M - tf.pow (eps, 2) * dif_matrix 
            Minv = tf.linalg.inv (M)

            return M, Minv, dif_matrix

#####################################################################
### INITIALIZE PARAMETERS CLASS
#####################################################################

class Initialize_parameters:
    """
        This class initialize parameters of the PSBC.
    """
    def __init__ (self):
        """
        Class initializer.
        
        Returns
        -------
        Nothing is returned. 
        """
        return None
    
    def dictionary (
        self, Nx, eps, Nt, pt_cardnlty,
        layers_K_shared, Neumann = True, dt = .2,
        learnable_eps = False, basis = None,
        retrieve_basis = False
    ):
        """
        'dictionary' method.
        
        This method initializes a dictionary of parameters containing all
        the information necessary to start the model training.

        Parameters
        ----------
        Nx : int, 
            Set paramters for the size of returned matrix (Nx x Nx)
        dt : float
            Mesh grid size of time discretization 
        ptt_cardnly : int
            Size of support of vectors in the basis matrix.
            See Section 2.2 in the paper.
        layers_K_shared : int
            Number of successive layers that are sharing their weights.
        Neumann : {bool, True}, optional
            Boundary condition.  If 'Neumann' is True, returns the 1D Laplacian
            discretized for Neumann Boundary conditions. 
            If 'Neumann' is False, return Periodic Boundary counditions.
        learnable_eps: {bool, False}, optional
            If True, then epsilon with be a learnable parameter, optimized
            during training. In such case, eps is randomly assigned as
            an Uniform in the interval [0,0.5]
            If False, then eps takes the value initially assigned to it.
        eps : float
            Diffiusion / Viscosity parameter. Only assigned by user
            if learnable_eps == True.
        
        Returns
        -------
        Dictionary containing initialization parameters.
        
        """
        parameters = {
        'Neumann' : Neumann,
        'Nx' : Nx,
        'Nt' : Nt, # Sets the maximum number of iterations
        'dt' : dt, # Time step
        'pt_cardnlty' : min (Nx, max (pt_cardnlty, 1)), # Dimens. of alpha_x_t^{\fl{n}}. 
                                                        # It has constraints
        'layers_K_shared' : min (Nt, max (layers_K_shared, 1))
        }
        parameters.update ({'eps' : eps})
        D = Diffusion ()
        parameters ['M'], parameters ['M_inv'], parameters ['dif_matrix'] = \
            D.id_minus_matrix (Nx, eps, dt, Neumann = Neumann)
        if retrieve_basis:
            parameters ['basis'] = tf.constant (basis, dtype=  tf.float32)
        else:
            parameters ['basis'] = self.pt_of_unity(parameters)
        
        return parameters

    def pt_of_unity(self, parameters, Nx = None, pt_cardnlty = None):
        """
        'pt_of_unity' method.
        
        This method initializes a basis matrix from parameters
        of a partition of unity.
        
        Parameters
        ----------
        parameters : dictionary
            Dictionary containing initialization parameters.

        Nx : {int, None}, optional    
            Number of features, or number of spatial grid points.
        pt_cardnlty : {int, None}, optional    
            Parameterization cardinality.
        
        Returns
        -------
        Basis matrix 'basis', where
        
        basis : numpy.ndarray of size Nx X pt_cardnlty.
        """
        ##-------------------------------------------
        ## READING PARAMETERS
        if Nx == None:
            Nx, pt_cardnlty = \
            parameters ['Nx'], parameters ['pt_cardnlty']  # 1 <= ptt_cardnlt <= Nx
        ##-------------------------------------------
        where_is_one = np.array_split (np.arange (Nx), pt_cardnlty)
        basis = np.zeros ([Nx, pt_cardnlty], dtype = np.float32)
        for a in zip (where_is_one, np.arange (pt_cardnlty)):
            basis [a [0], a [1]] = 1

        return tf.transpose (tf.constant(basis))

### A NEW METRIC

## https://keras.io/api/metrics/
def classify_zero_one (y_pred, y):
    """
    'classify_zero_one' function.

    This function computes the accuracy of the predicted vector
    y_pred when compared to the vector y.

    Parameters
    ----------
    y_pred : {np.array or tensor of type np.float32)}
        Vector with predicted labels.
    y : {np.array or tensor of type np.float32)}
        Vector with true labels.

    Returns
    -------
    The percentage of values in both vectors that agree.
    """
        
    classification = tf.cast (y_pred >.5, tf.float32)
    return tf.reduce_mean (tf.cast ((classification == y), tf.float32))

class classify_zero_one_v2 (tf.keras.metrics.Metric):
    """
    'classify_zero_one_v2' class.

    This class is essentially the same as the function classify_zero_one,
    but uses the Keras API. It computes the accuracy of the predicted vector
    y_pred when compared to the vector y.
    """
           
    ### See also https://datascience.stackexchange.com/questions/40886/how-to-change-the-names-of-the-layers-of-deep-learning-in-keras
    def __init__ (self, name = 'classify_zero_one_v2', **kwargs):
        """
        Initializer.

        Parameters
        ----------
        name : {str, 'classify_zero_one_v2'}, optional, 
            Model name recorded on history.
        **kwargs
        """
        super ().__init__ (name = name, **kwargs)
        self.correct_ones = self.add_weight (name = 'correct', initializer = 'zeros')
        self.how_many = self.add_weight (name = 'how_many', initializer = 'zeros')
        
    def update_state(self, y_pred, y):
        """
        Parameters
        ----------
        y_pred : {np.array or tensor of type np.float32)}
            Vector with predicted labels.
        y : {np.array or tensor of type np.float32)}
            Vector with true labels.
        
        Returns
        -------
        The percentage of values in both vectors that agree.
        """
        classification = tf.cast (y_pred >.5, tf.float32)
        self.correct_ones.assign_add (
            tf.reduce_sum (tf.cast ((classification == y), tf.float32))
            )
        self.how_many.assign_add (tf.cast (tf.shape (y)[0], tf.float32))

    def result (self):
        return self.correct_ones / self.how_many
    
    def reset_states(self):
        self.correct_ones.assign(0)
        self.how_many.assign(0)
        
    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        return super ().get_config ()

#####################################################################
### SAVE AND LOAD MODEL FUNCTIONS
#####################################################################
## See https://www.tensorflow.org/guide/keras/save_and_serialize

def save_PSBC (
    model, parameters, weight_name = "PSBC_model",
    parameters_name = "parameters.p"):
    """
    'save_PSBC' function.

    This function saves weights of a PSBC model as a file with extension ".h5".
    It can also save parameters as a pickled file (extension ".p").

    Parameters
    ----------
    model : {keras.Model},
        Model whose weights will be saved.
    parameters : {dictionary},
        Parameters of the model, to be pickled.
    weight_name : {str, "PSBC_model"}, optional,
        Name given to the h5 file containing the weights.
    parameters_name : {str, "parameters.p"}, optional,
        Name given to pickled files that contains the pickled parameters.
    
    Returns
    -------
    Nothing is returned.
    """
    
    number_layers = len (model.layers)
    Frozen_layers = {}
    
    if model != None:
        print ("Model's weights saved in pickled file as ", weight_name)
        model.save_weights (weight_name + ".h5")
    if parameters != None:
        print ("Model's parameters saved in pickled file as ", parameters_name)
        with open (parameters_name, 'wb') as save:
            pickle.dump (parameters, save, protocol = pickle.HIGHEST_PROTOCOL)

class load_PSBC_model ():
    """
    'load_PSBC_model' class.

    This class loads weights into a PSBC model
    """
    
    def __init__ (self):
        """
        Initializer.
        """
        pass

    def retrieve_parameters (self, filepath):
        """
        'retrieve_parameters' method.

        This method returns the pickled files at filepath,
        
        Parameters
        ----------
        filepath : {str},
            File path to the pickled files to be retrieved.

        Returns
        -------
        Unpickled  files stored at filepath.
        """
        with open (filepath, 'rb') as pickled_dic:
            Unfrozen_layers  = pickle.load (pickled_dic)
        return Unfrozen_layers

    def load_model_with_layers (self, model, filepath):
        """
        'load_model_with_layers' method.

        This method loads a model with weights stored at filepath,
        
        Parameters
        ----------
        model : keras.Model 
            PSBC model.
        filepath : {str},
            File path to the pickled files to be retrieved.

        Returns
        -------
        Nothing is returned.
        """
        model.load_weights (filepath)

#####################################################################
### LEARNING SCHEDULES AND CALLBACKS
#####################################################################
###  Validation data using custom models is a pain. See for instance
### https://github.com/keras-team/keras/issues/10472

### https://stackoverflow.com/questions/61939790/keras-custom-metrics-self-validation-data-is-none-when-using-data-generators#61971137
### https://stackoverflow.com/questions/47676248/accessing-validation-data-within-a-custom-callback
### https://stackoverflow.com/questions/50127527/how-to-save-training-history-on-every-epoch-in-keras
### https://datascience.stackexchange.com/questions/85409/getting-nn-weights-for-every-batch-epoch-from-keras-model

### https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods#63495028

### https://www.tensorflow.org/guide/checkpoint
#https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
#https://keras.io/api/callbacks/model_checkpoint/
#https://keras.io/guides/writing_your_own_callbacks/
#https://keras.io/api/callbacks/early_stopping/
#https://keras.io/guides/serialization_and_saving/

#https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
### https://keras.io/api/layers/base_layer/#add_loss-method
#https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer

"""
These two learning schedules are created to be default learning schedules.
"""
lr_schedule_U = keras.optimizers.schedules.ExponentialDecay (
    initial_learning_rate = 1e-3,
    decay_steps = 5,
    decay_rate = 0.99,
    staircase = True)

lr_schedule_P = keras.optimizers.schedules.ExponentialDecay (
    initial_learning_rate = 5e-1,
    decay_steps = 10,
    decay_rate = 0.95,
    staircase = True)

class EarlyStoppingAtMax_acc_and_save (keras.callbacks.Callback):
    #Mostly taken from https://keras.io/guides/writing_your_own_callbacks/
    """

    'EarlyStoppingAtMax_acc_and_save' is a keras callback.

    It is based on an early stop method, but has other features to help with
    storage of parameters, that can be iused for statistical purposes.
    It is also used for validation, manipulating information 
    related to the validation set at the end of each epoch.
    """
    def __init__ (
        self,  metric, filename = None,
        save_best = False, patience = np.inf,
        **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        filename :{str, None}, optional,
            If save_best is True, then model's weights are saved at filename.
        save_best : {bool, False}, optional,
            If True, then the weights of the best model are saved.
        patience: {int, np.inf}, optional,
            Threshold number. If the model does not improve its accuracy after 
            a patience number of epochs, training stops.
        **kwargs
        """
        
        super (EarlyStoppingAtMax_acc_and_save, self).__init__ (**kwargs)
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.metric = metric
        self.save_best = save_best
        self.filename = filename
        
    def on_train_begin (self, logs = None):
        """
        Define what is done at the beginning of each epoch.

        Returns
        -------
        Nothings is returned.
        """
        ### SEE https://keras.io/guides/training_with_built_in_methods/
        ### and https://keras.io/guides/customizing_what_happens_in_fit/
        self.wait = 0   # The number of epoch it has waited 
                        # when loss is no longer minimum.
        self.stopped_epoch = 0 # The epoch the training stops at.
        self.best = -np.Inf    # Initialize the best as minus infinity.
        self.callback_log = []
        
        if len (self.model.validation_data) != 0:
            #raise RuntimeError('Requires validation_data.')
            y_pred = self.model (
                self.model.validation_data [0], training = False)  
            accuracy_now = self.metric (
                y_pred, self.model.validation_data [1]).numpy ()
            self.callback_log.append (accuracy_now)
        
    def set_model(self, model):
        """
        Used to assert whether the model has a validation data or not, and also
        what shoudl be the filename of saved files.

        Returns
        -------
        Nothings is returned.
        """
        self.model = model
        if self.filename == None:
            self.filename = self.model.name + ".p"

        if len (model.validation_data) == 0:
            print ("Validation data not available")
        else:
            print ("Validation data available")

    def on_epoch_end (self, epoch, logs = None):
        """
        Define what is done at the end of each epoch.

        Returns
        -------
        Nothings is returned.
        """
        
        if len (self.model.validation_data) != 0: 
            y_pred = self.model (
                self.model.validation_data [0], training = False)  
            accuracy_now = self.metric (
                y_pred, self.model.validation_data [1]).numpy ()
            self.callback_log.append (accuracy_now)
            print ("\n\nAccuracy on the validation data", accuracy_now, "\n")
            if np.greater (accuracy_now,self.best):
                self.best = accuracy_now
                self.wait = 0
                # Record the best weights if current results is better (greater).
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    
    def on_train_end(self, epoch, logs = None):
        """
        Define what is done at the end of training.

        Returns
        -------
        Nothings is returned.
        """
        print ("\nEnd of training")
        if self.stopped_epoch > 0:
            print ("\nEpoch %05d: early stopping" % (self.stopped_epoch + 1))
        
        if len (self.model.validation_data) != 0:
            print ("\nSaving validation data accuracy data")

            print (self.callback_log)
            self.model.validation_accuracy = self.callback_log
            
            if self.save_best:
                print ("\nSaving validation data accuracy weights\
                    and data as pickled file")

                print (self.filename)
                callback_pickled = self.filename +"_val_accuracy.p"

                print (self.filename, callback_pickled)
                with open (callback_pickled, 'wb') as save:
                    pickle.dump (self.callback_log, save,
                                 protocol = pickle.HIGHEST_PROTOCOL)
            
                print ("\nRestoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

                print ("\nSaving the model")
                weights_pickled =  self.filename + "_best_weights"
                
                save_PSBC (self.model, None, weight_name = weights_pickled)
                print ("\nAccuracy on the validation data per epoch",
                      self.callback_log)
        else:
            print ("Copying history to validation_accuracy")
            self.model.validation_accuracy =\
            self.model.history.history ['classify_zero_one']


#####################################################################
### ALL BASIC LAYERS
#####################################################################

class PSBC_basic_layer (keras.layers.Layer):
    """
    'PSBC_basic_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    It consists of 1 iteration step of Allen-Cahn's equation, withough diffusion.
    """
    
    def __init__ (self, parameters, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.pt_cardnlty = parameters ["pt_cardnlty"]
        self.Nx = parameters ["Nx"]
        self.basis = parameters ["basis"]
        
    def call (self, X, dt):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
    
        #print (Z.shape)
        X =\
            X + tf.multiply(
                dt, tf.multiply (
                    tf.multiply (X, 1 - X), (X - self.w [tf.newaxis,:] @ self.basis)
                    ))
        return X, dt

    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        self.w = self.add_weight (
            shape = [self.pt_cardnlty],
            initializer = keras.initializers.RandomNormal (mean = .5, stddev = .1)
        )
        super ().build (batch_input_shape)

    def compute_output_shape (self, batch_input_shape):
        """
        Returns
        -------
        """
        return tf.TensorShape(batch_input_shape)[:-1] 
        
    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config ()
        return {**base_config, "pt_cardnlty": self.pt_cardnlty, "kind" : "Basic"}

class PSBC_viscosity_layer (keras.layers.Layer):
    """
    'PSBC_viscosity_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    Since the model is implicit, this step consists of the multiplication by 
    the inverse of the matrix L_N. (See the paper, Equations (2.15) and (2.16).)
    """
    
    def __init__ (self, parameters, trainable = False, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        trainable : {bool, False}, optional,
            Parameter used in case the diffusion parameter also becomes a
            trainable weight. In practice, we overwrite this term as False,
            because keras' autodiff struggles to compute the inverse matrix's
            derivative.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.dt = parameters ["dt"]
        self.Nx = parameters ["Nx"]
        self.eps = parameters ["eps"]
        self.dif_mat = parameters ["dif_matrix"]
        self.trainable = False   ##overwrite trainable
        
    def call(self, X, dt):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        X =  X @ tf.linalg.inv (
            tf.linalg.eye (self.Nx) - tf.pow (self.epsilon[0,0], 2) * self.dif_mat
            ) 
        return X, dt

    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        if self.trainable:
            self.epsilon = self.add_weight(
                initializer = keras.initializers.RandomUniform (
                    minval = 0, maxval = .3),
                shape = [1,1], trainable = self.trainable)
        else:
            self.epsilon = self.add_weight (
                initializer = tf.constant_initializer (self.eps),
                shape = [1,1], trainable = self.trainable
                )
        super ().build (batch_input_shape)

    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config ()
        return {**base_config, "dt": self.dt, "kind" : "Viscosity"}

class Zero_layer (keras.layers.Layer):
    """
    'Zero_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    It is the initial layer in the model, augmenting the input data by 
    a parameter dt that will be used in the PSBC model.

    """

    def __init__ (self, parameters, trainable = True, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        trainable : {bool, True}, optional,
            Defines whether dt (time discretization) is a learnable 
            weight or not.
        **kwargs
        """
        super ().__init__ (**kwargs)
        self.dt = parameters ["dt"]
        self.trainable = trainable
        
    def call(self, X):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        self.outputs = X, self.dt
        return X, self.dt

    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        self.dt = self.add_weight (
            initializer = tf.constant_initializer (np.squeeze (self.dt)),
            shape = [1,1], trainable = self.trainable)
        super ().build (batch_input_shape)

    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config()
        return {**base_config}

class Augment_layer (keras.layers.Layer):
    """
    'Augment_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    This function is responsible for coupling in the phase equation into 
    the model.
    """

    def __init__ (
        self, parameters, trainable = True, 
        with_phase = True,  subordinate = True, **kwargs
        ):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        trainable : {bool, True}, optional,
            Defines whether dtp (time discretization) is a learnable 
            weight or not.
        subordinate : {bool, True}, optional,
            If True, then the model is subordinate, non-subordinate otherwise.
        with_phase : {bool, True}, optional,
            If True, then the model has a coupled phase equation,
            otherwise it does not have one.
        **kwargs
        """
        
        
        super ().__init__ (**kwargs)
        if subordinate:
            self.Phase = 0.5 * tf.ones ([1, parameters ["pt_cardnlty"]])
        else:
            if with_phase:
                self.Phase = 0.5 * tf.ones ([1, 1])
            else: 
                # No phase, the model is scalar. Bad results ahead, even in simple
                # cases. Recall that pt_cardnlty  ==  1.
                self.Phase = tf.zeros ([1, 1]) #This is a 0. No training necessary
            
        self.Zero_layer_P = Zero_layer (parameters, trainable = with_phase)
        self.Zero_layer_P.build ((784,))
        self.train_var_dt_P = self.Zero_layer_P.trainable_variables

    def call(self, U, dt):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        P, dt_P = self.Zero_layer_P (self.Phase)
        return U, dt, P, dt_P
    
    def build (self, batch_input_shape):    
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        super ().build (batch_input_shape)
    
    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        base_config = super ().get_config()
        return {**base_config}

### CONCATENATING

class PSBC_concatenated (keras.layers.Layer):
    """
    'PSBC_concatenated' is a keras.layer.Layer used in the 
    PSBC's construction.

    This function is responsible for putting layers together, respecting
    the layers-k-shared architecture.
    the model.
    """

    def __init__ (self, parameters, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.Nt = parameters ["Nt"]
        self.eps = parameters ["eps"]
        self.pt_cardnlty = parameters ["pt_cardnlty"]
        self.parameters = parameters
        self.basis = parameters ["basis"] 
        
        ### Setting up the necessary  layers
        self.layers_K_shared = parameters ["layers_K_shared"]
        aux1 = np.repeat(
            self.layers_K_shared,\
                repeats= int (self.Nt / self.layers_K_shared))
        aux2 = self.Nt % self.layers_K_shared
        if aux2 != 0:
            aux = np.concatenate((aux1, [aux2]))
        else: aux = aux1
        self.sequence_layers = aux
        self.shared_layers =\
            [PSBC_basic_layer (parameters) for _ in self.sequence_layers]
        self.diffusion_layer = PSBC_viscosity_layer (parameters)
                               
    def call (self, X, dt):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        Z = X, dt
        for en_layer, layer in enumerate (self.shared_layers):
            for _ in range (self.sequence_layers [en_layer]):
                Z =  layer (*Z)
                if self.eps != 0:
                    Z = self.diffusion_layer (*Z)
        return Z
    
    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        super ().build (batch_input_shape)

    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config ()
        return {**base_config}

class PSBC_U_and_P (keras.layers.Layer):
    """
    'PSBC_U_and_P' is a keras.layer.Layer used in the 
    PSBC's construction.

    This function is responsible for putting layers together, respecting
    the layers-k-shared architecture.
    the model.
    """
    
    def __init__ (
        self, parameters, with_phase = True,
        subordinate = True, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        subordinate : {bool, True}, optional,
            If True, then the model is subordinate, non-subordinate otherwise.
        with_phase : {bool, True}, optional,
            If True, then the model has a coupled phase equation,
            otherwise it does not have one.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        initializer = Initialize_parameters ()
        self.par_U = parameters
        
        ### Setting up the parameters for the P model
        if subordinate:
            self.par_P = initializer.dictionary(
                Nx = parameters ["pt_cardnlty"], eps = 0,
                Nt = parameters ["Nt"], 
                pt_cardnlty = parameters ["pt_cardnlty"],
                layers_K_shared = parameters ["layers_K_shared"],
                dt = 1., learnable_eps = False
            )                   
        else:
            if with_phase:
                self.par_P = initializer.dictionary(
                    Nx = 1, eps = 0,
                    Nt = parameters ["Nt"], 
                    pt_cardnlty = 1,
                    layers_K_shared = parameters ["layers_K_shared"],
                    dt = 1., learnable_eps = False
                ) 
            else: 
                '''
                No phase, the model is scalar. Bad results ahead, even
                in simple cases. Recall that pt_cardnlty  ==  1.
                '''
                self.par_P = initializer.dictionary(
                    Nx = 1, eps = 0,
                    Nt = parameters ["Nt"], 
                    pt_cardnlty = 1,
                    layers_K_shared = parameters ["layers_K_shared"],
                    dt = 1., learnable_eps = False)                   
            
        self.U_network = PSBC_concatenated (self.par_U)
        self.P_network = PSBC_concatenated (self.par_P)
                               
    def call (self, U, dt_U, P, dt_P):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        U,dt_U = self.U_network (U, dt_U)
        P, dt_P = self.P_network (P, dt_P)
        
        return  U, dt_U, P, dt_P
    
    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        super ().build (batch_input_shape)        

    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config ()
        return {**base_config
        }

class Final_layer (keras.layers.Layer):
    """
    'Final_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    This layers combines both equations of the PSBC, in U and in P,
    combining them into the function S^P(U),
    as defined in Section 2 of the paper.
    """
    
    def __init__ (self,parameters, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.basis_P = parameters ["auxiliar_basis"]

    def call (self, U, dt_U, P, dt_P):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        return convexify (P, U, self.basis_P)

    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        super ().build (batch_input_shape)

    def get_config (self):
        """
        Configuration of the model. 
        
        Returns
        -------

        Dictionary with the model's configuration.
        """
        
        base_config = super ().get_config ()
        return {**base_config, "kind" : "Weight_sharing"}

### SOME AUXILIARY FUNCTIONS
def convexify (P, U, basis_P):
    """
    Equation (2.10) in the paper, that brings together the variables U and P.
    
    Returns
    -------
    A tensor of size (batch size, Nt)
    """
    P_tilde = tf.matmul(P, basis_P)
    return tf.add (tf.multiply (U, 1 - P_tilde),  tf.multiply (1 - U, P_tilde))

#@tf.autograph.experimental.do_not_convert
def orthodox_dt (weights):
    """    
    'orthodox_dt' function.

    This function is related to the Invariant Region Enforcing Condition..
    It calculates the maximum value of 
    time step based on the l^infty norm of training parameters.
    
    It assumes that the values passed are those in the range
    of the parameterization (2.7). See Section 3 in the paper.

    Parameters
    ----------
    weights: {tensor or numpy array, dtype = np.float32},
        Although it has this name, these are the weights mapped 
        throught the basis matrix.

    Returns
    -------
    Maximum value 'new_dt' of time discretization 
    (according to Enforced Invariant Region Constraint),
    diameter of the weights set 'max_diameter'.

    new_dt : float,
    max_diameter : float.  
    """
    Min_weight = tf.minimum (tf.reduce_min (weights), 0)
    Max_weight = tf.maximum (tf.reduce_max (weights), 1) 
    max_diameter = Max_weight - Min_weight
    new_dt = 0.57 / tf.pow (tf.maximum (max_diameter, 1), 2)
    return new_dt, max_diameter

#####################################################################
### BINARY_PHASE_SEPARATION MODEL
#####################################################################

class PSBC (keras.Model):
    """
    'PSBC' is a keras.Model that implements the PSBC model.
    """
    
    def __init__ (self, parameters,
                 lr_U = lr_schedule_U,lr_P = lr_schedule_P,
                 train_dt_U = True, train_dt_P = True, 
                 subordinate = True,
                 with_phase = True,
                 validation_data = (),
                 with_PCA = False,
                 **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        trainable : {bool, True}, optional,
            Defines whether dtp (time discretization) is a learnable 
            weight or not.
        with_phase : {bool, True}, optional,
            The model has phase if True, or does not if False.
        subordinate : {bool, True}, optional,
            The model is subordinate if True, or non-subordinate if False.
        lr_U : {learning schedule, lr_schedule_U}, optional, 

        lr_P = {learning schedule, lr_schedule_U}, optional, 

        train_dt_U : {bool, True}, optional,
            If True, then dtu is a trainable weight, non-trainable otherwise.
        train_dt_P : {bool, True}, optional,
            If True, then dtp is a trainable weight, non-trainable otherwise.
        subordinate : {bool, True}, optional,
            If True, then the model is subordinate, non-subordinate otherwise.
        with_phase : {bool, True}, optional,
            If True, then the model has a coupled phase equation,
            otherwise it does not have one.
        validation_data : {tuple, ()}, optional,

        with_PCA : {bool, False}, optional,

        **kwargs
        """
        
        super ().__init__ (**kwargs)
        self.Nt = parameters ["Nt"]
        self.eps = parameters ["eps"]
        self.pt_cardnlty = parameters ["pt_cardnlty"]
        self.layers_K_shared = parameters ["layers_K_shared"]
        self.par_U = parameters
        self.with_phase = with_phase
        self.with_PCA = with_PCA
        initializer = Initialize_parameters ()
        
        
        if parameters ["Neumann"]:
            print ("Setting up a U layer with Neumann B.C.s.")
        else:
            print ("Setting up a U layer with Periodic B.C.s.")
        
        if subordinate:
            print ("Setting up a subordinate model with phase")
            self.par_P = initializer.dictionary (
                Nx = parameters ["pt_cardnlty"], eps = 0.,
                Nt = parameters ["Nt"], 
                pt_cardnlty = parameters ["pt_cardnlty"],
                layers_K_shared = parameters ["layers_K_shared"],
                dt = 1., learnable_eps = False
            )                   
        else:
            if with_phase:
                print ("Setting up a non-subordinate model with phase")
                self.par_P = initializer.dictionary (
                    Nx = 1, eps = np.asarray ([0], dtype = np.float32),
                    Nt = parameters ["Nt"], 
                    pt_cardnlty = 1,
                    layers_K_shared = parameters ["layers_K_shared"],
                    dt = 1., learnable_eps = False
                )                   
            else: 
                print ("Setting up a non-subordinate model without phase")
                '''
                No phase, the model is scalar. Bad results ahead, even in simple
                cases. Recall that pt_cardnlty  ==  1.
                '''
                self.par_P = initializer.dictionary(
                    Nx = 1, eps = np.asarray ([0], dtype = np.float32),
                    Nt = parameters ["Nt"], 
                    pt_cardnlty = 1,
                    layers_K_shared = parameters ["layers_K_shared"],
                    dt = 1., learnable_eps = False
                )                   
               #This is a 0. No training necessary
        
        self.par_P ["auxiliar_basis"] = initializer.pt_of_unity(
            parameters = None,  Nx = self.par_U ["Nx"],
            pt_cardnlty = self.par_P ["pt_cardnlty"])
            #This is 0. No training necessary
        
        self.zero_layer  = Zero_layer (parameters)
        self.aug_layer =  Augment_layer (
            parameters, subordinate = subordinate, with_phase = with_phase)
        self.psbc_UP  = PSBC_U_and_P (
            parameters, subordinate = subordinate, with_phase = with_phase)
        self.final_layer = Final_layer (self.par_P)
        self.train_dt_U = train_dt_U
        self.train_dt_P = train_dt_P
        self.validation_accuracy = []
        ### Setting up the necessary  layers               
        self.optimizer_U = keras.optimizers.Adam (learning_rate = lr_U)
        self.optimizer_P = keras.optimizers.Adam (learning_rate= lr_P)
        self.validation_data = validation_data
        
    def call (self, X):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        outputs = self.custom_prediction (X)
        return  tf.reduce_mean(outputs, axis = 1)
    
    def custom_prediction (self, X):
        """
        Forward propagates the model, but does not compute the average in the
        last layer.

        Returns
        -------
        Last layer of the dynamical system associated with the PSBC, namely, 
        Equation (2.10) in the paper (see also Equation 2.11)
        """
        inputs = X
        outputs= self.zero_layer (inputs)
        outputs = self.aug_layer (*outputs)   ## The variable outputs is a list
        outputs = self.psbc_UP (*outputs)
        outputs = self.final_layer (*outputs)
        return  outputs
    
    def predict_zero_one (self, X):
        """
        Predicted class for a given element, either 0 or 1.

        Returns
        -------
        An array with size (batch size, ), with values in {0,1}
        """
        return  tf.cast (self.call (X) >.5, tf.float32)
    
    def build (self, batch_input_shape):
        """
        Build method in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        super ().build (batch_input_shape)

    def train_step (self, data):
        """
        Training step in a keras model.

        Returns
        -------
        Nothing is returned.
        """
        x, y = data
        trainable_vars_U = self.psbc_UP.U_network.trainable_variables
        trainable_vars_P = self.psbc_UP.P_network.trainable_variables
        
        ### See https://keras.io/guides/writing_a_training_loop_from_scratch/
        ### and
        ### https://keras.io/guides/training_with_built_in_methods/ 
        ### https://keras.io/guides/functional_api/#extract-and-reuse-nodes-in-the-graph-of-layers
        with tf.GradientTape() as tape:  #with ....Tape(persistent=True) 
            tape.watch([trainable_vars_U, trainable_vars_P])
            y_pred = self (x, training = True)  # Forward pass
            loss = self.compiled_loss (y_pred, y [:,tf.newaxis])
            
        gradients_U, gradients_P = tape.gradient (
            loss, [trainable_vars_U,trainable_vars_P]) # Compute gradients
        
        # Update weights
        self.optimizer_U.apply_gradients (
            zip (gradients_U, self.psbc_UP.U_network.trainable_variables))
        self.optimizer_P.apply_gradients (
            zip (gradients_P, self.psbc_UP.P_network.trainable_variables))
        
        ### UPDATE the dt's
        if self.train_dt_U:
            print ("Training dt_U")
            if self.with_PCA:
                ### Get trainable weights 
                L = tf.stack(self.psbc_UP.U_network.trainable_variables, axis=0)
                R = self.par_U ["basis"]
                print ("Doing computations on the range of the basis matrix\nin order to adjust dtu.")
                Range = tf.matmul(L,R)
                dt_U, max_diam_U = orthodox_dt (Range)
            else:
                dt_U, max_diam_U = orthodox_dt (
                self.psbc_UP.U_network.trainable_variables)

            self.get_layer (index = 0).trainable_variables [0].assign(
                [[tf.minimum (dt_U, self.par_U ["dt"])]]
                )
        if self.train_dt_P:
            print ("Training dt_P")
            dt_P, max_diam_P = orthodox_dt (
                self.psbc_UP.P_network.trainable_variables)
            if self.with_phase:
                self.get_layer (index = 1).trainable_variables [0].assign(
                    [[tf.minimum (dt_P, self.par_P ["dt"])]])
            
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state (y_pred, y)
        # Return a dict mapping metric names to current value
        return_dic = {m.name : m.result() for m in self.metrics }
        return_dic.update({
            "||W_U||_{infty} " : max_diam_U,  
            "||W_P||_{infty} " : max_diam_P,
            "dt_U " : dt_U,
            "dt_P " : dt_P
            })
        return return_dic

def PSBC_build_model (parameters, train_dt_U, train_dt_P, lr_U, lr_P, 
                      loss = keras.losses.mean_squared_error,
                      metrics = classify_zero_one,
                      print_summary = True, 
                      subordinate = True,
                      with_phase = True,
                      with_PCA = False,
                      validation_data = ()
                     ):
    """
    Build the PSBC model with a given architecture.

    Returns
    -------
    A keras model.
    """
    PSBC_model = PSBC (
        parameters,
        lr_P,
        lr_U,
        train_dt_P = train_dt_P,
        train_dt_U = train_dt_U,
        validation_data = validation_data,
        subordinate = subordinate,
        with_phase = with_phase,
        with_PCA = with_PCA)
    PSBC_model.compile(
        loss = loss,
        metrics = [metrics])
    PSBC_model.build ((784,))
    if print_summary:
        print (PSBC_model.summary ())
    return PSBC_model

#####################################################################
### UNFINISHED
#####################################################################


######## TO GET OUTPUTS PER LAYER
class PSBC_concatenated_with_output_per_layer (keras.layers.Layer):
    """
    'PSBC_concatenated_with_output_per_layer' is a keras.layer.Layer used in the 
    PSBC's construction.

    This layers combines both equations of the PSBC, in U and in P,
    combining them into the function S^P(U),
    as defined in Section 2 of the paper.
    """
    
    
    def __init__ (self, parameters, **kwargs):
        """
        Initializer.

        Parameters
        ----------    
        parameters : {dict},
            Dictionary with parameters to initialize the model.
        """
        
        super ().__init__ (**kwargs)
        self.parameters = parameters
        self.Nt = parameters ["Nt"]
        self.eps = parameters ["eps"]
        self.pt_cardnlty = parameters ["pt_cardnlty"]
        self.basis = parameters ["basis"] 
        
        ### Setting up the necessary  layers
        self.layers_K_shared = parameters ["layers_K_shared"]
        aux1 = np.repeat(
            self.layers_K_shared, repeats = int (self.Nt / self.layers_K_shared))
        aux2 = self.Nt % self.layers_K_shared
        if aux2 != 0:
            aux = np.concatenate ((aux1, [aux2]))
        else: aux = aux1
        self.sequence_layers = aux
        self.shared_layers =\
            [PSBC_basic_layer (parameters) for _ in self.sequence_layers]
        self.diffusion_layer = PSBC_viscosity_layer (parameters)
                               
    def call (self, X, dt):
        """
        Forward propagation.

        Returns
        -------
        Output in the final layer.
        
        """
        
        Z = X, dt
        junk = X
        for en_layer, layer in enumerate (self.shared_layers):
            for _ in range (self.sequence_layers[en_layer]):
                Z =  layer (*Z)
                if self.eps != 0:
                    Z = self.diffusion_layer (*Z)
                junk = keras.layers.concatenate ((junk, Z [0]), axis = 0)
        return junk, Z
    
    def get_config (self):
        """
        Configuration of the model. 

        Returns
        -------

        Dictionary with the model's configuration.
        """
        base_config = super ().get_config ()
        return {**base_config}