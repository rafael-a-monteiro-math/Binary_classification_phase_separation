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
__version__ = "0.0.1"
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

import  matplotlib.pyplot as plt
import scipy.sparse as sc
import itertools as it
import pandas as pd
import numpy as np
import warnings
import shutil 
import sympy
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

warnings.filterwarnings(action="ignore",message="internal issue")

#####################################################################
### DIFFUSION CLASS
#####################################################################

class Diffusion:
    """
    The 'Diffusion' class creates finite differences matrices of 
    finite differences, either of Neumann or Periodic type.
    """
    def __init__(self):
        """
        Class initializer. No returned value. 
        """
        return None
    
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
        
        A = sc.diags([1, -2, 1], [-1, 0, 1], shape = (N, N))
        A = np.asarray(A.todense())
        if Neumann:
            A[0, 1], A[N-1, N-2] = 2,2
        else: #Periodic
            A[0, N-1], A[N-1, 0] = 1, 1

        return A

    def id_minus_matrix(
            self, Nx, eps, dt, dx, Neumann = True):

        """ 
        'id_minus_matrix' method.
        
        This method returns a matrix 
        M = Id_N - (eps^2) * dt / (dx^2) * diffusion_matrix
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
        dx : float
            Mesh grid size of spatial discretization.  
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

        M = np.eye(Nx)
        if eps  ==  0 or Nx == 1:
            return M, M, np.zeros([Nx, Nx])
        
        dif_matrix = self.matrix(Nx, Neumann)
        M = M - (np.power(eps, 2)*dt/np.power(dx, 2))*dif_matrix 
        Minv = np.linalg.inv(M)

        return M, Minv, dif_matrix
    
#####################################################################
### INITIALIZE PARAMETERS CLASS
#####################################################################
    
class Initialize_parameters:
    """
        This class initialize parameters of the PSBC.
    """
    def __init__(self):
        """
        Class initializer. No returned value. 
        """
        return None
    
    def dictionary(
            self, Nx, eps, dt, dx, Nt, ptt_cardnlty, weights_K_sharing,
            sigma = .1, Neumann = True):
        """
        'dictionary' method.
        
        This method initializes a dictionary of parameters containing all
        the information necessary to start the model training.

        Parameters
        ----------
        Nx : int, 
            Set paramters for the size of returned matrix (Nx x Nx)
        eps : float
            Diffiusion / Viscosity parameter. 
        dt : float
            Mesh grid size of time discretization 
        dx : float
            Mesh grid size of spatial discretization.  
        ptt_cardnly : int
            Size of support of vectors in the basis matrix.
            See definition TODO in the paper.
        weights_K_sharing : int
            Number of successive layers that are sharing their weights.
        sigma : {float, 0.1}, optional
            Standard deviation of the weights upon initialization. Weights are
            initialized randomly as Normal(0.5, sigma^2).
        Neumann : {bool, True}, optional
            Boundary condition.  If 'Neumann' is True, returns the 1D Laplacian
            discretized for Neumann Boundary conditions. 
            If 'Neumann' is False, return Periodic Boundary counditions.

        Returns
        -------
        Dictionary containing initialization parameters.
        
        """
        
        parameters = {
        'sigma' : sigma,
        'eps' : eps, 
        'Neumann' : Neumann,
        'Nx': Nx,
        'dx': dx,
        'Nt': Nt,   # Sets the maximum number of iterations
        'dt': dt,  # Time step
        'ptt_cardnlty': min(Nx, max(ptt_cardnlty, 1)),  ##dimension of alpha_x_t^{\fl{n}}. It has constraints
        'weights_K_sharing': min(Nt, max(weights_K_sharing, 1))
        }
        parameters['M'], parameters['M_inv'], parameters['dif_matrix'] = \
            Diffusion().id_minus_matrix(Nx, eps, dt, dx, Neumann = Neumann)
        parameters['basis'] = self.ptt_of_unity(parameters)
        self.system_coef(parameters)

        return parameters

    def ptt_of_unity(self, parameters):
        """
        'ptt_of_unity' method.
        
        This method initializes a basis matrix from parameters
        of a partition of unity.
        
        Parameters
        ----------
        parameters : dictionary
            Dictionary containing initialization parameters.
        
        Returns
        -------
        Basis matrix 'basis', where
        
        basis : numpy.ndarray of size Nx X ptt_cardnlty.
        """

        ##-------------------------------------------
        ## READING PARAMETERS
        Nx, ptt_cardnlty = \
            parameters['Nx'], parameters['ptt_cardnlty']  # 1 <= ptt_cardnlt <= Nx
        ##-------------------------------------------

        where_is_one = np.array_split(np.arange(Nx), ptt_cardnlty)
        basis = np.zeros([Nx, ptt_cardnlty], dtype=np.int32)
        for a in zip(where_is_one, np.arange(ptt_cardnlty)):
            basis[a[0], a[1]] = 1

        return basis
    
    def system_coef(self, par):
        """
        'system_coef' method.

        This method initializes the coefficients of the training parameters
        across different layers taking into account weight sharing properties.
        
        Parameters
        ----------
        par : dictionary
            Dictionary containing initialization parameters.
        
        """
        ##-------------------------------------------
        ## READING PARAMETERS
        Nt,weights_K_sharing, ptt_cardnlty, sigma = \
            par['Nt'], par['weights_K_sharing'], par['ptt_cardnlty'], par['sigma']
        number_folds = max(int(Nt / weights_K_sharing), 1)
        ##-------------------------------------------
   	
        ## Getting the location of the folds
        w_index = [
            np.min(a) for a in np.array_split(np.arange(Nt), number_folds)
            ] # [0,fold,2 * fold,...]

        par["alpha_x_t"] = 0.5 + sigma * np.random.randn(ptt_cardnlty, Nt)
        for i in range(Nt):
            if i in w_index:
                continue
            par["alpha_x_t"][:, i] = par["alpha_x_t"][:, i-1]  
            # each one is a column

#####################################################################
### DISCRETIZED MODEL CLASS
#####################################################################
    
class Discretized_model:
    """
        This class auxiliates the PSBC model, computes the derivatives 
        used in its BAckpropagation.
    """
    
    def __init__(self):
        """
        Class initializer. No returned value. 
        """
        return None

    def diff_wrt_U(
            self, U, par, layer):
        """
        'diff_wrt_U' method.
        
        Taking into account the PSBC model, this method computes the derivative 
        of f(U, W) with respect to U.
        
        It returns a matrix in which the kth row corresponds to
        the derivative frac{partial f(U,W) }{partial U_k}
        
        Parameters
        ----------
        U : numpy.ndarray, 
            Set paramters for the size of returned matrix (Nx x Nx).
        par : dictionary
            Dictionary containing initialization parameters.
        layer : int
            Mesh grid size of time discretization.
        
        Returns
        -------
        'diff_F_U_aux.T', the transposed of the matrix 
        'diff_F_U_aux' of derivatives of f(U,W) w.r.t. U.

        'diff_F_U_aux.T': numpy.ndarray of size batch_size X Nx
        """
        
        ##-------------------------------------------
        ## READING PARAMETERS
        w, dt, basis = np.reshape(par["alpha_x_t"][:, layer], (-1, 1)),\
            par['dt'], par['basis']
        ##-------------------------------------------

        alpha_x_t = np.matmul(basis,w)    # alpha_x_t is a column vector

        ## U has shape N_x \times batch_size    
        diff_F_U_aux = 1+ dt * (U * (1-U)+ (U - alpha_x_t) * (1 - 2 * U))
        ## return this value as a matrix, where each column 
        ## denotes the diagonal entries of the derivative matrix

        return diff_F_U_aux.T   # size N_data X Nx

    def diff_wrt_w(self, U, par):
        """
        'diff_wrt_w' method.
        
        Taking into account the PSBC model, this method computes the derivative 
        of f(U, W) with respect to W.
        
        It returns a matrix in which the kth row corresponds to
        the derivative frac{partial f(U,W) }{partial W^{[k]}}
        
        Parameters
        ----------
        U : numpy.ndarray, 
            Set paramters for the size of returned matrix (Nx x Nx)
        par : dictionary
            Dictionary containing initialization parameters.
        layer : int
            Mesh grid size of time discretization. 
        
        Returns
        -------
        'diags_b.T', the transposed of the matrix 
        'diags_b', of derivatives of f(U,W) w.r.t. W.

        'diags_b.T': numpy.ndarray of size batch_size X Nx
        """
        ##-------------------------------------------
        ## READING PARAMETERS
        dt = par['dt']
        ##-------------------------------------------
        
        diags_b = -dt * U * (1-U)

        return diags_b.T   # size N_data X Nx

#####################################################################
### PROPAGATE CLASS
#####################################################################

class Propagate:
    """ 
    This class forward propagates the PSBC model.
    It corresponds to a 1D finite difference discretization of the Allen-Cahn
    model with bistable nonlinearity. 
    """
    def __init__(self):
        """
        Class initializer. No returned value. 
        """
        return None

    def forward(
            self, V_0, parameters,
            waterfall_save = False, Flow_save = True): 
        """
        'forward' method.
        
        From now on this function returns the vector b_i's 

        Parameters
        ----------
        V_0 : numpy.ndarray, 
            Features of an individual to be forward propagated,
            which can be seen as an initial condition in a Cauchy Problem.
        par : dictionary
            Dictionary containing initialization parameters.
        waterfall_save : {bool, False}, optional
            If True, saves a waterfall of the propagation on the initial 
            condition V_0, that can be used for waterfall plotting.

        Flow_save :  {bool, True}, optional
            If True, saves the evolution of the initial condition V_0
            throughout the model.

        Returns
        -------
        If Flow_save == True:
            A matrix 'Flow_matrix' with evolved values of V_0,
            a waterfall matrix 'waterfall',
            a time vector 'time'.

            Flow_matrix : numpy.ndarray of size Nx X batch_size X (Nt +1)
            waterfall :  list 
            time : list 

        If Flow_save == False:
            A matrix 'Flow_matrix' with the final values of V_0.
            a waterfall matrix 'waterfall',
            a time vector 'time'.

            Flow_matrix : numpy.ndarray of size Nx X batch_size X (1)
            empty list,
            empty list.
        """            
        ##-------------------------------------------
        ## READING PARAMETERS
        dt, eps, Nx, Nt, basis = \
            parameters['dt'], parameters['eps'],\
                parameters['Nx'], parameters['Nt'], parameters['basis']
        ##-------------------------------------------

        interval = np.linspace(0, 1, Nx, endpoint = True)
        _, Minv, _ = \
            parameters['M'], parameters['M_inv'], parameters['dif_matrix']

        ## take into account the initial conditions
        v = np.copy(V_0)   
        Flow_matrix = np.copy(v)

        ## We also augment the matrix M to include the boundary conditions
        waterfall = []
        time = []

        for i in range(Nt):
            ## Read polynomial corresponding to that layer
            w = np.reshape(parameters["alpha_x_t"][:, i],(-1, 1))
            alpha_x_t = np.matmul(basis, w)

            ## Semi-discrete iteration
            v = v +  dt * v * (1-v) * (v-alpha_x_t)
            if eps !=0:
                v = np.matmul(Minv,v)

            ## Store the vector to a flow matrix
            ## see also https://stackoverflow.com/questions/1727669/contruct-3d-array-in-numpy-from-existing-2d-array
            if Flow_save:
                Flow_matrix = np.dstack([Flow_matrix,v]) 


            if (i % 2) == 0 and waterfall_save:
                waterfall.append(list(zip(interval, v[:, 0])))
                time.append([i * dt])
        if Flow_save:
            return Flow_matrix, waterfall, time 
        else:
            return v, waterfall, time 

    def backward(
            self, U_flow, P_flow,
            par_U_model, par_P_model, Y,
            subordinate = True):  
        """
        'backward' method.
        
        This method backward propagates the model. 
        Parameters
        ----------
        U_flow : numpy.ndarray, 
            Array containing the state of U in each layer of the PSBC.
        P_flow : numpy.ndarray, 
            Array containing the state of P in each layer of the PSBC.
        par_U_model : dictionary
            Dictionary containing initialization parameters for the U component
            of the PSBC.
        par_P_model : dictionary
            Dictionary containing initialization parameters for the P component
            of the PSBC.
        Y : numpy.ndarray of size 1 x batch_size
            Array with individual's labels.
        subordinate : {bool, True}, optional
            If True the model is subordinate. 

        Returns
        -------
        Cost 'cost' of the model at given parameters, 
        a dictionary 'derivatives' with derivatives of the cost 
        function w.r.t. all parameters.

        cost : float
        derivatives : dictionary
        """    
        ##-------------------------------------------
        ## READING PARAMETERS AND INITIALIZING METHODS
        Discrete_evolution = Discretized_model()
        Nt = par_U_model['Nt']  # Number of layers
        eps = par_U_model['eps']
        basis = par_U_model['basis']
        weights_K_sharing = par_U_model['weights_K_sharing']
        number_folds = max(int(Nt / weights_K_sharing), 1)
        ptt_cardnlty = par_U_model['ptt_cardnlty']
        w_index = [
            np.min(a) for a in np.array_split(np.arange(Nt), number_folds)
            ]   # [0,fold,2 * fold,...]
    
        diffusive = (eps != 0) 	        
        
        ## This dictionary saves derivatives as "layer"+U or
        ##  w for partial cost/partialU^layer
        derivatives = { }
        cost, derivatives["phase" + str(Nt)], derivatives["u" + str(Nt)] = \
        Cost_and_its_derivative(P_flow[:, :, -1], U_flow[:, :, -1], Y,\
            par_U_model,subordinate)
        ##-------------------------------------------
  
        for layer in range(Nt-1, -1, -1): ##{Nt, N-1......,2, 1, 0}
            U_now = U_flow[:, :, layer]
            P_now = P_flow[:, :, layer]
            
            ## derivative w.r.t. U            
            self.update_previous_layer(
                Discrete_evolution, U_now, par_U_model, layer, derivatives,\
                    "u", diffusive, basis)
            ## derivative w.r.t. phase
            if subordinate:
                self.update_previous_layer(
                    Discrete_evolution, P_now, par_P_model, layer,\
                        derivatives, "phase", False, np.eye(ptt_cardnlty)
                        )
            else:
                self.update_previous_layer(Discrete_evolution, P_now,\
                    par_P_model, layer, derivatives, "phase", False, np.eye(1))
            
            ## Implementation weights-N-sharing model
            if layer in w_index:
                j = 1
                while ((layer + j) not in w_index) and ((layer + j) < Nt):
                    derivatives["alpha_x_t" + "u" + str(layer)] += \
                        derivatives["alpha_x_t" + "u"+str(layer + j)]
                    derivatives["alpha_x_t" + "phase"+str(layer)] += \
                        derivatives["alpha_x_t" + "phase"+str(layer + j)]
                    j += 1

        return cost, derivatives
    
    def update_previous_layer(
            self, Discrete_evolution, V,
            parameters, layer, derivatives,
            u_or_phase, diffusive, basis):
        """
        'update_previous_layer' method.
        
        Auxiliary method using in the backward propagation. It constructs the 
        derivatives of the cost with respect to U and W at the layer 'layer'
        using information about the model and cost's derivative at layer 
        'layer + 1'.

        Parameters
        ----------
        Discrete_evolution :
        V :
        parameters :
        
        layer : int
            Layer number.
        derivatives : dictionary
            Dictionary containing cost function derivatives w.r.t. U and W in each layer.
        u_or_phase : string
            Used to tag the key in the dictionary of derivatives, depending if computations concern
            P variables or U variables.
        diffusive : {bool}
            If True the PSBC is diffusive, otherwise it is non-diffusive.
        basis : numpy.ndarry
            Basis matrix.
        """    
        ## derivative w.r.t. U
        Uafter_diff_Ubefore = \
            Discrete_evolution.diff_wrt_U(V, parameters,layer)  # shape N_data X Nx
        Uafter_diff_alpha_x_tbefore = \
            Discrete_evolution.diff_wrt_w(V, parameters)  # shape N_data X Nx; no Basis matrix multiplication yet 
                                             
        
        ## Derivatives of the Cost function
        
        if diffusive:
            #.....w.r.t. to U........
            M_inv = parameters['M_inv']
            aux = np.matmul(derivatives[u_or_phase+str(layer + 1)],M_inv)
            derivatives[u_or_phase + str(layer)] = aux * Uafter_diff_Ubefore
            #.....w.r.t. to w........
            derivatives["alpha_x_t"+u_or_phase + str(layer)] = \
                np.transpose(\
                    np.matmul(np.sum(aux * Uafter_diff_alpha_x_tbefore,\
                        axis = 0, keepdims = True),basis))
        else:
            #.....w.r.t. to U........
            derivatives[u_or_phase + str(layer)] = \
                derivatives[u_or_phase + str(layer + 1)] * Uafter_diff_Ubefore
            #.....w.r.t. to w........
            derivatives["alpha_x_t" + u_or_phase + str(layer)] =np.transpose(\
                np.matmul(np.sum(derivatives[u_or_phase + str(layer + 1)] * \
                    Uafter_diff_alpha_x_tbefore, axis = 0, keepdims = True), basis))

 
    def update_weights(
            self, parameters, par_P_model,
            derivatives, learning_rate): 
        """    
        'update_weights' method.
        
        This method updates the weights doing gradient descent with weight
        sharing.
        
        Parameters
        ----------
        parameters : dictionary
            Dictionary containing initialization parameters for the U component
            of the PSBC.
        par_P_model : dictionary
            Dictionary containing initialization parameters for the P component
            of the PSBC.
        derivatives : dictionary
            Dictionary containing cost function derivatives w.r.t. U and W in each layer.
        learning_rate : float
            Learning rate.
        """        
        ##-------------------------------------------
        ## READING PARAMETERS
        Nt, weights_K_sharing = parameters['Nt'], parameters['weights_K_sharing']
        number_folds = max(int(Nt  /  weights_K_sharing), 1)
        w_index = [\
            np.min(a) for a in np.array_split(np.arange(Nt), number_folds)\
                ]   # [0, fold, 2 * fold,.... ]
        ##-------------------------------------------
        
        for i in range(Nt):
            if i in w_index:
                ## The next two are column vectors
                w_U_model_aux = np.reshape(parameters["alpha_x_t"][:, i],(-1, 1))
                w_Phase_aux = np.reshape(par_P_model["alpha_x_t"][:, i],(-1, 1))

                ## Update parameters
                parameters["alpha_x_t"][:, i] = np.squeeze(
                    w_U_model_aux -learning_rate
                     * derivatives["alpha_x_t" + "u" + str(i)])
                par_P_model["alpha_x_t"][:, i] = np.squeeze(
                    w_Phase_aux -learning_rate * 
                    derivatives["alpha_x_t" + "phase" + str(i)])
                continue
            ## copy the inner layers parameters    
            parameters["alpha_x_t"][:, i] = \
                np.squeeze(parameters["alpha_x_t"][:, i-1])
            par_P_model["alpha_x_t"][:, i] = \
                np.squeeze(par_P_model["alpha_x_t"][:, i-1])
        
    def orthodox_dt(
        self, parameters):
        """    
        'orthodox_dt' method.

        This method is related to the Enforced Invariant Region Condition.
        It calculates the maximum value of 
        time step based on the l^infty norm of training parameters.
        
        Parameters
        ----------
        parameters : dictionary
            Dictionary containing initialization parameters for the U or P component
            of the PSBC.
        
        Returns
        -------
        Maximum value 'new_dt' of time discretization 
        (according to Enforced Invariant Region Constraint),
        diameter of the weights set 'max_diameter'.

        new_dt : float,
        max_diameter : float.  
        """

        max_each_layer = max(np.max(parameters["alpha_x_t"]), 1)  #positive 
        min_each_layer = min(np.min(parameters["alpha_x_t"]), 0)  #non-positive 
        
        max_diameter = max_each_layer - min_each_layer
        new_dt = 0.57 / np.power(max(max_diameter, 1), 2)
        return new_dt, max_diameter

#####################################################################
### BINARY_PHASE_SEPARATION CLASS
#####################################################################
                        
class Binary_Phase_Separation:
    """    
    This is the main class  of the Phase Separation Binary Classifier (PSBC).
    With its methods one can, aong other things, train the model and 
    predict classifications (once the model has been trained).
    """
    def __init__(
            self, cost = None, par_U_model = None,
            par_P_model = None, par_U_wrt_epochs = None, par_P_wrt_epochs = None
        ):
        """
        Class initializer. 
        
        Parameters
        ----------
        cost : {bool, True}, optional
        par_U_model : {dictionary, None}, optional
            Dictionary containing initialization parameters for the U component
            of the PSBC.
        par_P_model : {dictionary, None}, optional
            Dictionary containing initialization parameters for the P component
            of the PSBC.
        par_U_wrt_epochs : {dictionary, None}, optional
            Dictionary containing dictionaries of U parameters throughout the 
            training.
        par_P_wrt_epochs : {dictionary, None}, optional
            Dictionary containing dictionaries of P parameters throughout the 
            training.

        Attributes
        ----------
        cost, par_U_model, par_P_model, par_U_wrt_epochs, par_P_wrt_epochs

        Returns
        -------
        Class initializer. No returned value. 
        Parameters are accessible as instance variables. 
        """        
        
        self.cost = cost 
        self.par_U_model = par_U_model
        self.par_P_model = par_P_model
        self.par_U_wrt_epochs = par_U_wrt_epochs
        self.par_P_wrt_epochs = par_P_wrt_epochs
     
        return None

    def train(
            self, X, Y,
            X_test, Y_test, learning_rate, dt, dx, layers,
            weights_K_sharing, eps = 0, ptt_cardnlty = 1, batch_size = None,
            epochs = 30, subordinate = True, with_phase = True, drop_SGD = 1,
            patience = float("+inf"), sigma = .1, orthodox_dt = True,
            print_every = 30, Neumann = True, save_parameter_hist = False):
        """
        'train' method.

        This method trains the PSBC model with a given set of parameters and 
        data.
        
        Parameters
        ----------
        X : numpy.ndarray of size Nx X N_data
            Matrix with features. 
        Y : numpy.ndarray of size 1 X N_data
            Matrix with labels. 
        X_test : numpy.ndarray of size Nx X N_data_test
            Matrix with features. 
        Y_test : numpy.ndarray of size 1 X N_data_test
            Matrix with labels. 
        learning_rate : float or tuple
            If Tuple with three elements (a,b,c), 
            these numbers  parametrize the learning rate decay.
        dt : float
            Mesh grid size of time discretization 
        dx : float
            Mesh grid size of spatial discretization.  
        layers : int
            Number o f layers. 
        weights_K_sharing : int
            Number of successive layers that are sharing their weights.
        eps : {float, 0}, optional
            Diffiusion / Viscosity parameter. 
        ptt_cardnlty : {int, 1}, optional
            Partition cardinality.
        batch_size : {int, None}, optional
            Size of mini_batches. If None, the method uses full batch size.
        epochs : {int, 30}, optional
            Number of epochs.
        subordinate : {bool, True}, optional
            If True the PSBC is subordinate, otherwise it is not.
        with_phase : {bool, True}, optional
            If True the PSBC uses the phase variable,
            otherwise it does not uses it.
        drop_SGD : {1, float}, optional
            Parameter used to find basin of optimal parameter (stochastically).
            If model reaches this accuracy then SGD is dropped and training
            switches to full batch size.
        patience : {float, float("+inf")}, optional
            Parameter used in Early Stopping.
        sigma : {float, 0.1}, optional
            Standard deviation of the weights upon initialization. Weights are
            initialized randomly as Normal(0.5, sigma^2).
        orthodox_dt = True : {bool, True}, optional
            If True, the Enforced Invariant Region Constrain is applied.
            If False, dt is kept constant throughout training.
        print_every : {int, 30}, optional
            Throughout training some values will be printed every 30 epochs. 
        Neumann : {bool, True}, optional
            Boundary condition.  If 'Neumann' is True, returns the 1D Laplacian
            discretized for Neumann Boundary conditions. 
            If 'Neumann' is False, return Periodic Boundary counditions.
        save_parameter_hist = False : {bool, True}, optional
            Save a dictionary with the model parameters in each epoch. 
            WARNING: memory consuming.

        Returns
        -------
        No returned value. 
        Trainable parameters are accessible as instance variables. 
        
        """     
        ##Initialize methods
        initializer = Initialize_parameters()
        propagator = Propagate()
        
        if with_phase:
        	ptt_cardnlty = max(min(X.shape[0],ptt_cardnlty), 1)   
        else:
        	ptt_cardnlty = 1     ## Forces the partition to have cardinality 1
        	
        Nx = int(X.shape[0])
        par_U_model = initializer.dictionary(
            Nx, eps, dt, dx, layers, ptt_cardnlty,\
                weights_K_sharing, sigma = sigma, Neumann = Neumann)

        par_U_model['M'], par_U_model['M_inv'], par_U_model['dif_matrix'] = \
            Diffusion().id_minus_matrix( Nx, eps, dt, dx, Neumann = Neumann)

        ## par_P_model's layers are subordinate to par_U_model's layers
        if subordinate:
            par_P_model = initializer.dictionary(
                ptt_cardnlty, 0, dt, 0,layers,ptt_cardnlty,\
                    weights_K_sharing, sigma = sigma)
            Phase = 0.5 * np.ones([ptt_cardnlty, 1])
        else:
            par_P_model = initializer.dictionary(
                1, 0, dt, 0,layers, 1, weights_K_sharing, sigma = sigma)
            if with_phase:
                Phase = 0.5 * np.ones([1, 1])
            else: 
                '''
                No phase, the model is scalar. Bad results ahead, even in simple
                cases. Recall that ptt_cardnlty  ==  1.
                '''
                Phase = np.zeros([1, 1])   #This is a 0. No training necessary

        ## ADDING MORE INFORMATION TO par_U_model
        par_U_model.update(
            {"batch_size" : batch_size,  "epochs" : epochs,\
                "subordinate" : subordinate, "with_phase" : with_phase})

        ## INITIALIZE WEIGHTS
        initializer.system_coef(par_U_model)
        initializer.system_coef(par_P_model)

        cost_wrt_epochs, cost_wrt_epochs_non_averaged = [], []  # costs
        accuracies_hist = []  # save the accuracy in each epoch
        diameters_hist = {'P' : [], 'U' : []}  # save hist of diameters
        dt_hist = {'P' : [], 'U' : []}  # save hist of dt

        ## Save given dt
        given_dt = dt
        ## UPDATE dt to satisfy global existence with frozen coefficients
        if orthodox_dt == True:
            #calculate diameters and update U
            dt_U, par_U_model["max_diam"] = \
                propagator.orthodox_dt(par_U_model)
            par_U_model['dt'] = min(given_dt, dt_U)
            par_U_model['M'], par_U_model['M_inv'], _ = \
                Diffusion().id_minus_matrix(\
                    Nx, eps, par_U_model['dt'], dx, Neumann = Neumann)

            #calculate diameters and update P
            dt_P, par_P_model["max_diam"] = \
                propagator.orthodox_dt(par_P_model)
            par_P_model['dt'] = min(given_dt, dt_P)
            
            ## save hist U
            dt_hist["U"].append(np.copy(par_U_model['dt']))
            diameters_hist["U"].append(np.copy(par_U_model["max_diam"]))
            
            ## save hist P
            dt_hist["P"].append(np.copy(par_P_model['dt']))
            diameters_hist["P"].append(np.copy(par_P_model["max_diam"]))
 
        cost_0, _, _ = Cost_and_its_derivative(
            Phase,X,Y, par_U_model, subordinate = subordinate)

        cost_wrt_epochs.append(np.copy(cost_0))    
        cost_wrt_epochs_non_averaged.append(np.copy(cost_0))
        
        ## Create dictionaries
        par_U_wrt_epochs, par_P_wrt_epochs = { }, { }

        ## Calculate mini_batches sizes
        if batch_size != None:
            batch_size = max(1, min(batch_size,Y.shape[1]))
            number_mini_batches = int(np.ceil(Y.shape[1] / batch_size))
        else:
            number_mini_batches = 1
            drop_SGD = 1  #for the model is already deterministic

        ## learning_rate = a + b * np.power(c, epoch_now)
        if isinstance(learning_rate,float) or isinstance(learning_rate, int):
            lng_rate_a = learning_rate
            lng_rate_b = 0
            lng_rate_c = 1
        else:                                    #learning_rate is a tuple!
            lng_rate_a = learning_rate[0]
            lng_rate_b = max(learning_rate[1], 0)
            lng_rate_c = min(.99,max(learning_rate[2], 0))
        
        ## For early stopping
        par_U_model["drop_SGD"] = -1  # work as a flag
        best_accuracy = float("-inf")
        best_epoch = 0
        best_par_U_model, best_par_P_model = { }, { }

        for epoch_now in range(epochs):
            ## For early stopping
            if ((epoch_now - best_epoch) > patience): break
            
            lng_rate = lng_rate_a + lng_rate_b * np.power(lng_rate_c, epoch_now)
            
            if save_parameter_hist: #save parameters
                par_U_wrt_epochs[str(epoch_now)] = copy.deepcopy(par_U_model)
                par_P_wrt_epochs[str(epoch_now)] = copy.deepcopy(par_P_model)
                
            ## mini batch split if requested, but drop it if asked
            if best_accuracy > drop_SGD:
                if (par_U_model["drop_SGD"]  ==  -1): 
                    print("\n Dropping SGD!")
                    par_U_model["drop_SGD"] = np.copy(best_epoch)  
                
                number_mini_batches = 1  # Flag drop of stochasticity

            mini_batch_splitting = mini_batches(X,Y, number_mini_batches)	
            mini_batch_now_cost =0

            ## Minibatch SGD
            for mini_batch_now in mini_batch_splitting:
                
                X_mini_batch_now, Y_mini_batch_now = mini_batch_now

                if orthodox_dt == True:
                    #calculate diameters and update U
                    dt_U, par_U_model["max_diam"] = \
                        propagator.orthodox_dt(par_U_model)
                    par_U_model['dt'] = min(given_dt, dt_U)
                    par_U_model['M'], par_U_model['M_inv'], _ = \
                        Diffusion().id_minus_matrix(
                            Nx, eps, par_U_model['dt'], dx, Neumann = Neumann)
        
                    #calculate diameters and update P
                    dt_P, par_P_model["max_diam"] = \
                        propagator.orthodox_dt(par_P_model)
                    par_P_model['dt'] = min(given_dt, dt_P)
                    
                    ## save hist U
                    dt_hist["U"].append(np.copy(par_U_model['dt']))
                    diameters_hist["U"].append(np.copy(par_U_model["max_diam"]))
                    
                    ## save hist P
                    dt_hist["P"].append(np.copy(par_P_model['dt']))
                    diameters_hist["P"].append(
                        np.copy(par_P_model["max_diam"]))
                    
                ## Do a forward propagation...
                U_flow, _, _ = propagator.forward(X_mini_batch_now, par_U_model)
                P_flow, _, _ = propagator.forward(Phase, par_P_model)
                
                ## ... followed by a backward propagation
                cost_now, derivatives = propagator.backward(
                    U_flow, P_flow, par_U_model, par_P_model,\
                        Y_mini_batch_now,  subordinate)

                ## Then update the weights ...
                propagator.update_weights(
                    par_U_model, par_P_model, derivatives, lng_rate)

                ## ... and compute the averaged cost
                mini_batch_now_cost += cost_now  /  number_mini_batches
                cost_wrt_epochs_non_averaged.append(np.copy(cost_now))

            ## Early stopping
            predict_now, _, accuracy_now = self.predict_and_accuracy(
                X_test, Y_test, par_U_model, par_P_model,\
                    with_phase = with_phase, subordinate = subordinate)

            ## Epoch's cost (regularized due to averaging over minibatches)   
            if epoch_now%print_every == 0:
                print("\n epoch :", epoch_now,"cost", mini_batch_now_cost)
                print("\n accuracy :",accuracy_now)
            cost_wrt_epochs.append(np.copy(mini_batch_now_cost))
            
            accuracies_hist.append(np.copy(accuracy_now))    

            ## Early stopping 
            if (accuracy_now > best_accuracy):
                best_accuracy = np.copy(accuracy_now)
                best_epoch = np.copy(epoch_now)
                best_cost = np.copy(mini_batch_now_cost)
                best_par_U_model = copy.deepcopy(par_U_model)
                best_par_P_model = copy.deepcopy(par_P_model)
                best_prediction_vector = np.copy(predict_now)
                
        self.cost= cost_wrt_epochs 
        self.cost_non_averaged= cost_wrt_epochs_non_averaged 
        self.par_U_model = par_U_model
        self.par_P_model= par_P_model
        self.par_U_wrt_epochs = par_U_wrt_epochs
        self.par_P_wrt_epochs = par_P_wrt_epochs
        self.accuracies_hist = accuracies_hist
        self.dt_hist = dt_hist
        self.diameters_hist = diameters_hist
        ## Early stopping results
        self.best_accuracy = best_accuracy
        self.best_epoch = best_epoch
        self.best_par_U_model = best_par_U_model
        self.best_par_P_model = best_par_P_model 
        self.best_cost = best_cost
        self.best_prediction_vector = best_prediction_vector
                
    def predict_and_accuracy(
            self, X, Y, par_U_model, par_P_model, with_phase = True,
            subordinate = True):
        """
        'predict_and_accuracy' method.

        This method predicts the labels of X for a PSBC with parameters given
        by par_U_model and par_P_model. Accuracy is compared to the true labels 
        Y.

         Parameters
        ----------
        X : numpy.ndarray of size Nx X N_data
            Matrix with features. 
        Y : numpy.ndarray of size 1 X N_data
            Matrix with labels. 
        par_U_model : {dictionary, None}, optional
            Dictionary containing initialization parameters for the U component
            of the PSBC.
        par_P_model : {dictionary, None}, optional
            Dictionary containing initialization parameters for the P component
            of the PSBC.
        with_phase : {bool, True}, optional
            If True the PSBC uses the phase variable,
            otherwise it does not uses it.
        subordinate : {bool, True}, optional
            If True the PSBC is subordinate, otherwise it is not.
       
        Returns
        -------
        Predicted labels as a vector 'predict_vector',
        cost at given parameters as 'cost',
        accuracy when compared to labels Y as 'accuracy_now'.

        predict_vector : numpy.ndarray,
        cost : float,
        accuracy_now : float.
        """
        ##-------------------------------------------
        ## READING PARAMETERS AND INITIALIZING METHODS
        propagator = Propagate()   # forward propagate 
        ptt_cardnlty = par_U_model['ptt_cardnlty']
        basis = par_U_model['basis']
        Nx = par_U_model['Nx']
        ##-------------------------------------------
        
        U_flow, _, _ = propagator.forward(X, par_U_model, Flow_save = False)

        if subordinate:
            P_flow, _, _ = propagator.forward(\
                0.5 * np.ones([ptt_cardnlty, 1]), par_P_model, Flow_save = False)
            ## PREDICTION PART
            p = np.reshape(P_flow,(-1, 1))
            alpha_x_t = np.matmul(basis,p)
        else:
            if with_phase:
                P_flow, _, _ = propagator.forward(\
                    0.5 * np.ones([1, 1]), par_P_model, Flow_save = False)
	            ## PREDICTION PART
                alpha_x_t = np.reshape(P_flow,(1, 1))                
            else:
                P_flow = np.zeros([1, 1])
                alpha_x_t = np.zeros([1, 1])
                
        cost, _, _ = \
             Cost_and_its_derivative(P_flow, U_flow,Y, par_U_model,subordinate)
        predict_vector = \
            np.array(((1 / Nx) * (np.sum((1-2 * alpha_x_t) * U_flow + alpha_x_t,\
                keepdims = True, axis = 0))) > .5, dtype = np.int32)
        accuracy_now = np.sum(Y == predict_vector) / Y.shape[1]

        return predict_vector, cost, accuracy_now


    def predict(
                self, X, par_U_model, par_P_model, with_phase = True,
                subordinate = True):
            """
            'predict' method.

            This method predicts the labels of X for a PSBC with parameters given
            by par_U_model and par_P_model. 
            No accuracy is computed.

            Parameters
            ----------
            X : numpy.ndarray of size Nx X N_data
                Matrix with features. 
            par_U_model : {dictionary, None}, optional
                Dictionary containing initialization parameters for the U component
                of the PSBC.
            par_P_model : {dictionary, None}, optional
                Dictionary containing initialization parameters for the P component
                of the PSBC.
            with_phase : {bool, True}, optional
                If True the PSBC uses the phase variable,
                otherwise it does not uses it.
            subordinate : {bool, True}, optional
                If True the PSBC is subordinate, otherwise it is not.
        
            Returns
            -------
            Predicted labels as a vector 'predict_vector'.

            predict_vector : numpy.ndarray,
            """
            
            Y_artificial =  np.zeros([1, X.shape[1]])

            predict_vector, _, _ =\
                self.predict_and_accuracy(X, Y_artificial,\
                    par_U_model, par_P_model,\
                        with_phase, subordinate)

            return np.squeeze(predict_vector)
    

#####################################################################
### INITIALIZE_DATA CLASS
#####################################################################

class Initialize_Data:
    """
    This class preprocess the data, normalizing it.
    """
        
    def __init__(self):
        """
        Class initializer. No returned value. 
        """
        return None

    def normalize(self, Z, sigma = .2):
        """
        'normalize' method.

        This method normalizes the data. The range of each 
        coordinate of the normalized data takes values 
        in between [0.5 - sigma/2, 0.5 + sigma/2]
        Values cetralized at .5, 
        
        Parameters
        ----------
       
        Returns
        -------
        Normalized data as 'Z_normalized', 
        minimum value of non-normalized data as 'min_vals',
        maximum value of non-normalized data as 'max_vals'.

        Z_normalized : numpy.ndarray,
        min_vals : float,
        max_vals : float.
        """
        sigma = min(1, max(0,sigma))   # enforce 0 <= sigma <= 1
        scaler = MinMaxScaler()
        scaler.fit(Z)
        Z_normalized = scaler.transform(Z)
        Z_normalized = .5 - (sigma / 2) + sigma * Z_normalized

        return Z_normalized, scaler.data_min_, scaler.data_max_

    def denormalize(
            self, Z, min_vals,
            max_vals, sigma = .2):
        """
        'denormalize' method.
    
        This method puts the data back to its original scale.
        Of the non-normalized data the method uses its  minimum value
        min_vals, its original maxum value max_vals, and sigma.
        The non-normalized data is transformed by

        A  = ( 1 / sigma ) * ( Z - .5 + sigma /2)

        and then Z_2 = min_vals + A * (max_vals - min_vals).

        Z_2 is the returned value.
        
        Parameters
        ----------
       
        Returns
        -------
        Non-normalized data 'A'.

        A : numpy.ndarray
        """

        A = (1 / sigma) * (Z - .5 + (sigma / 2))
        A = min_vals + A * (max_vals - min_vals)

        return A

#####################################################################
### SOME FUNCTIONS
#####################################################################

def Cost_and_its_derivative(
        phase_final, U_final, Y,
        par, subordinate):
    """
    'Cost_and_its_derivative' function.

    This function evaluates the cots function at the given parameters, and
    computes the gradients (first derivatives) of the cost function with respect 
    to  variables U and P.
    
    Cost_and_its_derivative(phase_final,U_final,Y)
    
    In this case we would like to return a matrix that has 
         frac{partial f}{partial U_k} 
    in its kth row.
    
    Parameters
    ----------
    phase_final :numpy.ndarray of size Nx X N_data
        Matrix with features in the P commponent, propagated to the last layer. 
    U_final : numpy.ndarray of size Nx X N_data
        Matrix with features in the U commponent, propagated to the last layer. 
    Y : numpy.ndarray of size 1 X N_data
            Matrix with labels. 
    par : dictionary, None}, optional
        Dictionary containing initialization parameters for the U component
        of the PSBC.
    subordinate : {bool, True}, optional
        If True the PSBC is subordinate, otherwise it is not.

    Returns
    -------
    Cost at given parameters as 'cost', 
    derivative of cost w.r.t. P as dcost_0, 
    derivative of cost w.r.t. U as dcost_1.

    cost : float,
    dcost_0 : dictionary,
    dcost_1 : dictionary.
    """
    ##-------------------------------------------
    ## READING PARAMETERS
    basis = par['basis']
    N_data = Y.shape[1]
    ##-------------------------------------------
    
    if subordinate:
        phase_together = np.matmul(basis, phase_final) #  shape Nx times 1
    else:
        phase_together = phase_final
    
    ###... but actually we need a diagonal version of it. 
    #q_p_final_ufinal is a matrix
    q_p_final_u_final = (1 - 2 * phase_together) * U_final + phase_together - Y 
    norm_aux = np.linalg.norm(q_p_final_u_final)
    cost = 1 /(2 * N_data)*np.power(norm_aux,2)
    
     
    ##### Derivative vector
    ## ---- first term : derivative w.r.t. the phase term,
    #frac{\partial Cost}{\partial p_j}     
    unrolled = np.sum(
        q_p_final_u_final * (1 - 2 * U_final), axis = 1, keepdims = True)
    first_entry = np.matmul(basis.T,unrolled)  # shape ptt_cardnlty X 1 
    if (not subordinate):
        first_entry = np.sum(first_entry, keepdims = True)

    ## ---- second term : derivative w.r.t. u,    
    #frac{\partial Cost}{\partial u^{(j)}}
    #return a matrix that has frac{\partial f}{\partial U_k} in its kth row. 
    second_entry = (1 - 2 * phase_together) * q_p_final_u_final 
    
    dcost = ((1 /N_data) * first_entry, (1/N_data) * second_entry.T)
    #####

    return cost, dcost[0], dcost[1]


def mini_batches(X, Y, number_mini_batches = 1):
    """
    'mini_batches' function.

    This function splits the data (X, Y) in 'number_mini_batches' parts.

    Parameters
    ----------
    X : numpy.ndarray of size Nx X N_data
        Matrix with features. 
    Y : numpy.ndarray of size 1 X N_data
        Matrix with labels. 
    number_mini_batches : {int, 1}, optional
        Number of elements in which (X,Y) will be split.

    Returns
    -------
    A list 'minibatches' of tuples with 'number_mini_batches' elements, where the elements
    makeup a partition of the dataset (X,Y).

    minibatches : list.
    """
    
    full_batch_size = X.shape[1]
    number_mini_batches = min(
        full_batch_size, max(number_mini_batches, 1))
    splitting = np.array_split(
        np.random.permutation(full_batch_size), number_mini_batches)
    
    mini_batches = []
    
    for i in range(number_mini_batches):
        X_mini_batch_now = X[:, splitting[i]]
        Y_mini_batch_now = Y[:, splitting[i]]
        ## Create list
        mini_batches.append((X_mini_batch_now, Y_mini_batch_now))
    
    return mini_batches
