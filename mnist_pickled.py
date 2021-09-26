################################################################################
#### HOW TO USE
################################################################################
###
###  This program can be called as 
###  
###  python 3.* mnist_picled.py XX
###  
###  where XX is MNIST or somthing else. 
###  When it is MNIST, the program retrieves the mnist dataset version 1 and split
###  it in train test, sets, saving each class in a separated pickled files.
###  (X_train_0.p denotes 0 digits in the train set etc).
###  When XX is not MNIST, then the program retrieves all pairs of distinct
###  digits {0....9} and pairs them in groups 
###  generate_k_fold_a_b.p
###  which is a dictionary with keys "0"...."4" where each key is of the form
### (train_index, test_index, mean_train_grid, Vt, self.variable_0, self.variable_1)
###  where mean_train_grid is mean(X_train_a_b [train_index], axis =0)
#### and Vt is used for PCA purposes.
###  In this cases things are shuffled with random seed 13 (which can be changed).
#################################################################################


import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import fetch_openml

import itertools as it
import pickle
import sys

###############################################################################################
def select_and_pickle_one_variable (X, Y, variable, type_of = "train"):
    Y_int = Y.astype ("uint8")
    select_var = np.where (Y_int == variable)[0]
    X_selected = X [np.ravel (select_var),:].astype (np.float32)
    
    file_name = "X_" + type_of + "_" + str (variable) + ".p"
    print ("Pickling variable", variable, "as", file_name )
    with open (file_name, 'wb') as save:
        pickle.dump (X_selected, save, protocol = pickle.HIGHEST_PROTOCOL)


class select_split_pickle ():
    
    def __init__ (self):
        pass
    def select_variables_from_pickle (self, variable_0, variable_1, averaged = False, type_of = "train"):
        """
        Warning: the output of this function is not shuffled!
        """
        
        self.variable_0 = min(variable_0, variable_1)
        self.variable_1 = max(variable_0, variable_1)
        assert (self.variable_0 != self.variable_1)
        ### It holds that self.variable_0 < self.variable_1
        file_name_0 = "Pickled_datasets/X_" + type_of + "_" + str (self.variable_0) + ".p"
        file_name_1= "Pickled_datasets/X_" + type_of + "_" + str (self.variable_1) + ".p"

        with open (file_name_0, 'rb') as pickled_dic:
            X_0  = pickle.load (pickled_dic)
        with open (file_name_1, 'rb') as pickled_dic:
            X_1  = pickle.load (pickled_dic)

        X_all = np.r_[X_0, X_1]

        average = None
        if averaged:
            average = np.mean (X_all, axis = 0)
            X_all = X_all - average

        Y_all = np.r_[np.zeros(X_0.shape[0]), np.ones (X_1.shape [0])]
        
        #print ("\nShuffling")
        #p = np.arange(len (Y_all))
        #np.random.shuffle(p)
        #return X_all [p,:], Y_all[p], average    ### If you shuffle you cannot retrieve the indices
        return X_all, Y_all, average

    def k_fold (self,X, Y,  cv, random_state = 13, shuffle = True):
        
        #kf = KFold(n_splits = cv, random_state=random_state, shuffle=shuffle)
        # For stratfied shuffling
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        
        skf = StratifiedKFold(n_splits = cv, random_state=random_state, shuffle=shuffle)
        generate_k_fold = {}
        
        ### Grid ranges
        for i, (train_index, test_index) in enumerate (skf.split (X, Y)):
            X_train_grid, _ = X [train_index], Y [train_index]
            ### Centralization 
            mean_train_grid = np.mean (X_train_grid, axis = 0)
            X_train_grid = X_train_grid - mean_train_grid
            
            _, _, Vt = np.linalg.svd (X_train_grid)
            generate_k_fold [str (i)] = (train_index, test_index, mean_train_grid, Vt, self.variable_0, self.variable_1)
        
        print ("pickling file")
        file_name = "Pickled_datasets/generate_k_fold_" + str (self.variable_0)+"_"+str (self.variable_1) + ".p"
        with open (file_name, 'wb') as save:
            pickle.dump (generate_k_fold, save, protocol = pickle.HIGHEST_PROTOCOL)
        
        print ("Data written as", file_name)
        return generate_k_fold

###############################################################################################

if sys.argv [1] == "MNIST":

    print ("Inporting Mnist dataset and pickling it")
    mnist = fetch_openml('mnist_784', version = 1)

    X, Y = mnist["data"], mnist["target"]

    M = MinMaxScaler (feature_range = (0,1))
    M.fit (X)
    X_norm = M.transform (X)
    X_norm_train, Y_train, X_norm_test, Y_test = X_norm [:60000,:], Y [:60000], X_norm [60000:,:], Y [60000:]

    print ("\nSplitting digits")
    for variable in range(10):
        select_and_pickle_one_variable (X_norm_train, Y_train, variable)    
        select_and_pickle_one_variable (X_norm_test, Y_test, variable, type_of = "test")    

    print ("\nVariables saved")
    print ("Done!")

else:
    print ("\nDoing SVD and preparing for k_fold")
    S = select_split_pickle ()
    for  variable_0, variable_1 in it.combinations (np.arange (10), 2):
        
        print ("Saving variables", variable_0," and ", variable_1)
        X_now, Y_now, _ = S.select_variables_from_pickle (variable_0, variable_1)
        
        _ = S.k_fold (X_now, Y_now, cv = 5)
        
print ("Done!")
