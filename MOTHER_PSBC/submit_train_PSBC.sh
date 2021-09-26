#!/bin/bash
################################################################################
#### HOW TO USE
################################################################################
### This is a program that submits batch jobs. 
###
### It runs as 
### ./submit_train.sh Nt index with_PCA parallel subordinate
###
### and receives five parameters 
###
### i) Nt (int) number of layers
### 
### ii) index (int) indicates a pair of distinct variables in {0,...9}
###
### iii) with_PCA (bool, True or False) indicates whether or not the model uses
### PCA, 
###
### iv) parallel (bool, True or False, whether the model is parallel or not),
###
### v) subordinate (bool, True or False, whether the model is subordinate or not),
###
################################################################################

#this is  job name line
#this is  execution directory
#PBS -d ./
#this is out.log file
#PBS -o out.log
#this is error.log
#PBS -e err.log
# resources like nodes prcessors and wallclock
#PBS -l nodes=matham03-03:ppn=6
#PBS -l walltime=400:00:00
#PBS -o myscript$PBS_JOBNAME.out
#PBS -e myscript$PBS_JOBNAME.err
ulimit -a > t.log
cat $PBS_NODEFILE >temp.nodes
NP=`cat $PBS_NODEFILE|wc -l`
Neumann=$1
Nt=$2
index=$3
with_PCA=$4
cpu=6
parallel=$5
subordinate=$6
export OMP_NUM_THREADS=$cpu
#mpirun -machinefile $PBS_NODEFILE -np $NP /opt/lammps-12Dec18/src/lmp_icc_openmpi <BLTP.in>out
(python3.6 ../MOTHER_PSBC/train_model.py $Neumann $Nt $index $with_PCA $cpu $parallel $subordinate)
#to submit serial job use qsub -F "parameter" <submitfile >
