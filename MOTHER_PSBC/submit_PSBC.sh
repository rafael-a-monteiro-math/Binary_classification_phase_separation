#!/bin/bash
################################################################################
#### HOW TO USE
################################################################################
### This is a program that submits batch jobs. 
###
### It is called by typing
### ./submit_PSBC.sh index with_PCA parallel grid_type 
###
### where 
###
### i) index (int) indicates a pair of distinct variables in {0,...9}
###
### ii) with_PCA (bool, True or False) indicates whether or not the model uses
### PCA, 
###
### iii) parallel (bool, True or False, if model is parallel or not),
###
### iv) grid_type is either "all", "vary_eps", or "vary_Nt"
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
#PBS -l nodes=matham03-03:ppn=5
#PBS -l walltime=400:00:00
#PBS -o myscript$PBS_JOBNAME.out
#PBS -e myscript$PBS_JOBNAME.err
ulimit -a > t.log
cat $PBS_NODEFILE >temp.nodes
NP=`cat $PBS_NODEFILE|wc -l`
index=$1
with_PCA=$2
cpu=5
parallel=$3
grid_type=$4
Neumann=$5
Nt=$6
export OMP_NUM_THREADS=$cpu
#mpirun -machinefile $PBS_NODEFILE -np $NP /opt/lammps-12Dec18/src/lmp_icc_openmpi <BLTP.in>out
(python3.6 run_minibatches.py $Neumann $Nt $index $with_PCA $cpu $parallel $grid_type)

#to submit serial job use qsub -F "parameter" <submitfile >
