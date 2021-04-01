#!/bin/bash

main() 
{
    replicates=$1
    outdir=$2
    real=$3
    shift; shift; shift;
    for arg in "$@"; do
        python /overflow/dschridelab/projects/GeneGraphs/src/data/simulate_msprime.py --outdir "${outdir}/${arg}" --id test --length 0.1 --model $arg --replicates $replicates
    done
    if [ $real == "real" ]
    then
        python3 /overflow/dschridelab/projects/GeneGraphs/src/data/format.py --idir $outdir --ofile "${outdir}.hdf5" --real 
    else
        python3 /overflow/dschridelab/projects/GeneGraphs/src/data/format.py --idir $outdir --ofile "${outdir}.hdf5"
    fi
    echo "done with formatting!"
}

main $@
