#!/bin/bash
#SBATCH --time=3-00:00:00
##SBATCH -p volta-gpu
##SBATCH --qos=gpu_access 
##SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=128G
#SBATCH -J train_newrep
#SBATCH -o train_newrep.%A.out
#SBATCH -e train_newrep.%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu

cd /overflow/dschridelab/projects/GeneGraphs/embeddings
source activate treenets

python node_network.py ${1}