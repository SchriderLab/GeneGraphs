#!/bin/bash

## This is an example of an sbatch script to run a pytorch script
## using Singularity to run the pytorch image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##

#SBATCH --job-name=pytorch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS


# Set SIMG name
SIMG_NAME=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

IFILE=$1
IFILE_VAL=$2
CONFIGS=$3
IDIR=$4
ODIR=$5
N_EPOCHS=$6
LR=$7
WEIGHT_DECAY=$8

mkdir -p ${ODIR}

# GPU with Singularity
echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 ../src/models/train_gcn.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS} --idir ${IDIR} --odir ${ODIR} --n_epochs ${N_EPOCHS} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --verbose
singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 ../src/models/train_gcn.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS} --idir ${IDIR} --odir ${ODIR} --n_epochs ${N_EPOCHS} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --verbose