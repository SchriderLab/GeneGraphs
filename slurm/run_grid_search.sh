#!/bin/bash

#SBATCH --job-name=GNN_grid_search
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


IFILE=$1
IFILE_VAL=$2
SEARCH_IDIR=$3
ODIR=$4
N_EPOCHS=$5

mkdir -p ${ODIR}

for config_file in ls $IDIR/*; do 
# GPU with Singularity
echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 /src/models/train_gcn.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${config_file} --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose
singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 /src/models/train_gcn.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${config_file} --odir ${ODIR} --n_epochs ${N_EPOCHS} --verbose
; done