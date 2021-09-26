import itertools as it
import numpy as np
import pickle
import os
import sys

try: os.mkdir ("Grids")
except: pass


for Neumann in ["True","False"]:

    for Nt in [1,2,4]:
        
        for grid_type in ["training","grid_search"]: 

            filename = Neumann + "_" + str (Nt) + ".p"
            EPOCHS = 10
            patience = 10
            train_dt_U = True
            train_dt_P = True
            cv = 5

            if grid_type == "training":
                print ("Creating grid for training")
                cv = 5
                EPOCHS = 20
                filename = "training_"+filename
            else:
                print ("Creating grid for search")
                filename = "grid_search_"+filename

            grid_range = {
                "cv" : cv,
                "Neumann" : Neumann=="True",
                "EPOCHS" : EPOCHS,
                "patience" : patience,
                "Nt" : Nt,
                "train_dt_U" : train_dt_U, 
                "train_dt_P" : train_dt_P
            }

            print ("Creating grid for search")

            with open ("Grids/" + filename, 'wb') as save:
                pickle.dump (grid_range, save, protocol = pickle.HIGHEST_PROTOCOL)


