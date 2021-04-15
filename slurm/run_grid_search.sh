#!/bin/bash

#SBATCH --job-name=GNN_grid_search_launch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=run-%j.log
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

main() 
{
    IFILE=$1
    IFILE_VAL=$2
    CONFIGS_DIR=$3
    ODIR=$4
    for CONFIG in $CONFIGS_DIR/*; do
        echo sbatch slurm/run_config.sh ${IFILE} ${IFILE_VAL} ${CONFIG} ${ODIR}
        sbatch slurm/run_config.sh ${IFILE} ${IFILE_VAL} ${CONFIG} ${ODIR} 
    done
    echo "done launching jobs!"
}

main $@