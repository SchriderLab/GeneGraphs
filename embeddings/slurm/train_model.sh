#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -p volta-gpu
#SBATCH --qos=gpu_access 
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -J train_newrep
#SBATCH -o train_newrep.%A.out
#SBATCH -e train_newrep.%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu

cd /overflow/dschridelab/projects/GeneGraphs/embeddings
source activate compbio

python tree_network.py lstm