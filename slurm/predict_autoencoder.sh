#!/bin/bash

## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##

#SBATCH --job-name=predict_autoencoder
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
ODIR=$2
MODEL=$3
DEMOGRAPHIC_MODEL="$4"

mkdir -p ${ODIR}

# GPU with Singularity
echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 ../src/models/predict_autoencoder.py  --ifile ${IFILE} --odir ${ODIR} --model ${MODEL} --demographic_model ${DEMOGRAPHIC_MODEL} --verbose
singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 ../src/models/predict_autoencoder.py --ifile ${IFILE} --odir ${ODIR} --model ${MODEL} --demographic_model ${DEMOGRAPHIC_MODEL} --verbose