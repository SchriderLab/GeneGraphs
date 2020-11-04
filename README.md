# GeneGraphs

Work in progress.  To do:
- [x] Routines for generating and formatting data from Ariella's demographic models
- [x] AE and VAE GCN models
- [ ] Make GCN classifier
- [ ] Make training script(s) save models, output training results, work with > 1 GPUs
- [ ] Make other classifiers temporal and otherwise and potentially other auto-encoders
- [ ] Add flexibility for node and edge features (right now: node features (time (0 to 1) + hot encoded population), no edge features for now)

Generate some graphs with msprime (Choices of demographic model to simulate are `constant`, `growth`, `reduction`, `constant_2pop`, 
`constant_3pop`, `single_pulse_uni_AB`, `single_pulse_uni_BA`, `single_pulse_bi`, `multi_pulse_uni_AB`, 
`multi_pulse_uni_BA`, `multi_pulse_bi`, `continuous_uni_AB`, `continuous_uni_BA`, `continuous_bi`):

```
python src/data/simulate_msprime.py --outdir test_out --id test --length 0.1 --model constant_2pop --replicates 100
```

Format the data:

```
python3 src/data/format.py --idir test_out/ --ofile test.hdf5
```

Train an auto-encoder (this script doesn't yet save a model, but it is functional assuming the dependencies are installed)

```
python3 src/models/train_autoencoder.py --ifile test.hdf5
```
