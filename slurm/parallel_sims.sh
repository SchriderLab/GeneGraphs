#!/bin/bash

# simulates data for each demographic model specified
# note: A third argument must be specified, even if you are not using real trees (probably a way to fix this)
# example for inferred trees: slurm/parallel_sims.sh 1000 output_sims blah single_pulse_uni_AB constant_2_pop single_pulse_bi
# example for real trees: slurm/parallel_sims.sh 1000 output_sims real single_pulse_uni_AB constant_2_pop single_pulse_bi
main() 
{
    replicates=$1
    outdir=$2
    shift; shift;
    if [ $3 == "real" ]
    then
    shift;
    for arg in "$@"; do
        sbatch -p general -N 1 --mem 50g -n 1 -t 7-0:00:00 --wrap="python3 src/data/simulate_msprime.py --outdir ${outdir}/${arg} --id test --length 0.1 --model ${arg} --replicates ${replicates}"
    done
    else
    for arg in "$@"; do
        sbatch -p general -N 1 --mem 50g -n 1 -t 7-0:00:00 --wrap="python3 src/data/simulate_msprime.py --outdir ${outdir}/${arg} --id test --length 0.1 --model ${arg} --replicates ${replicates} --inferred"
    done
    fi
    echo "done launching jobs!"
}

main $@