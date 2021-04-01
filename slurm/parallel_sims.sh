#!/bin/bash

main() 
{
    replicates=$1
    outdir=$2
    real=$3
    shift; shift; shift;
    for arg in "$@"; do
        if [ $real == "real" ]
        then
        sbatch -p general -N 1 --mem 50g -n 1 -t 7-0:00:00 --wrap="python3 ../src/data/simulate_msprime.py --outdir ${arg}/${outdir} --id test --length 0.1 --model ${arg} --replicates ${replicates}"
        else
        sbatch -p general -N 1 --mem 50g -n 1 -t 7-0:00:00 --wrap="python3 ../src/data/simulate_msprime.py --outdir ${arg}/${outdir} --id test --length 0.1 --model ${arg} --replicates ${replicates} --inferred"
        fi
    done
    echo "done launching jobs!"
}

main $@