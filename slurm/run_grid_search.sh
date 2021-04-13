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

# Set SIMG name
SIMG_NAME=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

main() 
{
    IFILE=$1
    IFILE_VAL=$2
    CONFIGS_DIR=$3
    ODIR=$4
    for config in $CONFIGS_DIR/*; do
        echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py  --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${config} --odir ${ODIR} --verbose
        singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 src/models/train_gcn.py --ifile ${IFILE} --ifile_val ${IFILE_VAL} --config ${config} --odir ${ODIR} --verbose  
    done
    echo "done launching jobs!"
}

main $@