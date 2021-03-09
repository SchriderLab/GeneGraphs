#!/bin/bash
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=4
#SBATCH -J embeddings
#SBATCH -o embeddings.%A.out
#SBATCH -e embeddings.%A.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=lswhiteh@email.unc.edu

#Replace these with default examples before release
cd /overflow/dschridelab/projects/GeneGraphs/embeddings
source activate compbio

python convert_data.py ${1}