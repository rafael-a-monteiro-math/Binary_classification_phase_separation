#!/bin/bash

################################################################################
#### HOW TO USE
################################################################################
### This is a program that submits batch jobs. 
###
### It runs as 
### ./PSBC_grid.sh  Neumann  Nt  with_PCA  index  parallel  grid_type  training  subordinate
###
### and receives five parameters 
###
### i) Neumann (bool, True or False) Boundary conditions
###
### ii) Nt (int) number of layers
### 
### iii) with_PCA (bool, True or False) indicates whether or not the model uses
### PCA, 
###
### iv) index (int) indicates a pair of distinct variables in {0,...9}
###
### v) parallel (bool, True or False, whether the model is parallel or not),
###
### vi) grid_type is either "all", "vary_eps", or "vary_Nt"
###
### vii) training is either "training" or "grid_search".
### 
### viii) subordinate (bool, True or False, whether the model is subordinate or not).
###
################################################################################

Neumann=$1
Nt=$2
with_PCA=$3
index=$4
parallel=$5
training=$6
subordinate=$7

name_submission="BC"$Neumann"_Nt"$Nt"_PCA"$with_PCA"_par"$5_$training

echo -e "\nSubmitting $name_submission"

######################################################################3
##### FOR TESTS!
#index=0
#(python3.5 run_minibatches.py $index $with_PCA 4 $parallel)
######################################################################3

if [ $training = "training" ]
then
    echo -e "Training"
    q1=$(qsub -N "$name_submission" -F "$Neumann $Nt $index $with_PCA $parallel $subordinate" submit_train_PSBC.sh)
else
    echo -e "Grid search"
    q1=$(qsub -N "$name_submission" -F "$index $with_PCA $parallel $grid_type $Neumann $Nt" submit_PSBC.sh)
fi
