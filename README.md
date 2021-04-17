# GeneGraphs

Work in progress.  To do:
- [ ] Make generalized Classifier class
- [ ] Make time / sequence based classifier models (GCN + 1D CNN and/or RNN)

Generate some graphs with msprime (Choices of demographic model to simulate are `constant`, `growth`, `reduction`, `constant_2pop`, 
`constant_3pop`, `single_pulse_uni_AB`, `single_pulse_uni_BA`, `single_pulse_bi`, `multi_pulse_uni_AB`, 
`multi_pulse_uni_BA`, `multi_pulse_bi`, `continuous_uni_AB`, `continuous_uni_BA`, `continuous_bi`):

You can generate them in parallel with the parallel_sims.sh shell:
```
slurm/parallel_sims.sh 1000 /proj/dschridelab/output_dir real constant_2pop single_pulse_uni_AB continuous_bi
```

Then you have to format the data:

```
python3 src/data/format.py --idir test_out/ --ofile test.hdf5
```

Alternatively, you can generate and format them simultaneously with the make_simulations.sh shell.
Example:
```
make_simulations.sh 1000 ./output_dir real constant_2pop single_pulse_uni_AB multi_pulse_bi continuous_uni_AB
```
However, this process is not parallelized (so it is best for small data sizes)

Train an auto-encoder (this script doesn't yet save a model, but it is functional assuming the dependencies are installed)

```
python3 src/models/train_autoencoder.py --ifile test.hdf5
```
