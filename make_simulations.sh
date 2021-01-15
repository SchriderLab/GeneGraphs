#!/bin/bash

main() 
{
    replicates=$1
    outdir=$2
    ofile=$3
    shift; shift; shift;
    for arg in "$@"; do
        python src/data/simulate_msprime.py --outdir "${outdir}/${arg}" --id test --length 0.1 --model $arg --replicates $replicates
    done
    python3 src/data/format.py --idir $outdir --ofile "${ofile}.hdf5"
    echo "done with formatting!"
}

main $@
