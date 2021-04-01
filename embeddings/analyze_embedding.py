import os
import sys
from glob import glob
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

for h5file in glob(
    "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/*.hdf5"
):
    h5name = h5file.split("/")[-1].split(".")[0].split("_")[-3:]
    nums_trees = []
    nums_nodes = []

    h5dat = h5py.File(h5file, "r")
    h5_data = h5dat[list(h5dat.keys())[0]]
    seqs = list(h5_data.keys())

    for seq in tqdm(seqs, desc="Trees"):
        nums_trees.append(len(h5_data[seq]))

        for tree in list(h5_data[seq].keys()):
            nums_nodes.append(len(h5_data[seq][tree]["embedding"]))

    plt.hist(nums_nodes, bins=20, histtype=u"step", label=h5name)

plt.title("Number of Nodes in Trees")
plt.legend()
plt.savefig("nodes_hist.png")