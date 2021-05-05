#!/bin/bash

# Runs GCN script
# Example: slurm/run_gcn.sh train.hdf5 val.hdf5 gcn_configs.txt output_dir 5

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
ODIR=$4
N_EPOCHS=$5

mkdir -p ${ODIR}

# GPU with Singularity
if [[ "$6" == "" ]]; then
  echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS}  --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose
  singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS}  --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose
else
  echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS}  --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose --predict_sequences
  singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${CONFIGS}  --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose --predict_sequences
fi