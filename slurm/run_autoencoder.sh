#!/bin/bash

# Trains autoencoder
# Example: slurm/run_autoencoder.sh train_data.hdf5 val_data.hdf5 autoencoder_output 5 autoencoder_config.txt

#SBATCH --job-name=pytorch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64g
#SBATCH --time=5-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS

SIMG_NAME=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg


IFILE=$1
IFILE_VAL=$2
ODIR=$3
N_EPOCHS=$4
CONFIG=$5
TAG=test

mkdir -p ${ODIR}

# GPU with Singularity
  echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_autoencoder.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --config ${CONFIG} --tag ${TAG} --verbose
  singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_autoencoder.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --odir ${ODIR} --n_epochs ${N_EPOCHS} --config ${CONFIG} --tag ${TAG} --verbose