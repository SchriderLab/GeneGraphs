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
#SBATCH --time=2-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

unset OMP_NUM_THREADS

# Set SIMG path
SIMG_PATH=/nas/longleaf/apps/pytorch_py3/1.4.0/simg

# Set SIMG name
SIMG_NAME=pytorch1.4.0-py3-cuda10.1-ubuntu16.04.simg

IFILE=$1
IFILE_VAL=$2
ODIR=$3
N_EPOCHS=$4
IN_FEATURES=$5
OUT_FEATURES=$6
LINEAR=$7
VARIATIONAL=$8
TAG=$9
ADVERSARIAL=${10}

mkdir -p ${ODIR}

# GPU with Singularity
if [ ADVERSARIAL ]
then
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME python3 ../src/models/train_adversarial_autoencoder.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --in_features ${IN_FEATURES} --out_features ${OUT_FEATURES} --linear ${LINEAR} --variational ${VARIATIONAL} --tag ${TAG} --verbose
    singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME python3 ../src/models/train_adversarial_autoencoder.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --in_features ${IN_FEATURES} --out_features ${OUT_FEATURES} --linear ${LINEAR} --variational ${VARIATIONAL} --tag ${TAG} --verbose
else
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME python3 ../src/models/train_autoencoder.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --in_features ${IN_FEATURES} --out_features ${OUT_FEATURES} --linear ${LINEAR} --variational ${VARIATIONAL} --tag ${TAG} --verbose
    singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME python3 ../src/models/train_autoencoder.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --in_features ${IN_FEATURES} --out_features ${OUT_FEATURES} --linear ${LINEAR} --variational ${VARIATIONAL} --tag ${TAG} --verbose
fi