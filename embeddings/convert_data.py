import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import h5py
import numpy as np
import networkx as nx
from nodevectors import Node2Vec
import sys
import os
from tqdm import tqdm


def iterate_trees(h5file, demo, VEC_DIMS):
    """
    Loads in HDF5 file of tree sequence data and returns a dict.

    Args:
        h5file (str, optional): HDF5 file containing tree sequence data. Defaults to "/overflow/dschridelab/10e4_test_infer_FINAL.hdf5".
        demo (str): Key from the HDF5 file of which demography model to process.
        VEC_DIMS (int): Number of dimensions of Node2Vec output embeddings. Tuneable.

    Returns:
        dict: Condensed format of data with keys=class labels and values=list of edge list arrays.
    """
    rawh5 = h5py.File(h5file, "r")
    outname = os.path.join(
        os.getcwd(), h5file.split("/")[-1].split(".")[0] + f"_{demo}.hdf5"
    )
    outh5 = h5py.File(outname, "a")

    for seq in tqdm(list(rawh5[demo].keys()), desc="Embedding trees..."):
        for tree in list(rawh5[demo][str(seq)].keys()):
            edges = np.vstack(rawh5[demo][str(seq)][str(tree)]["edge_index"][:]).T
            attr = rawh5[demo][str(seq)][str(tree)]["x"][:]
            embedding = convert_graph(edges, VEC_DIMS)
            # rawh5[demo][str(seq)][str(tree)]["embedding"] = embedding
            outh5.create_dataset(
                f"{demo}/{seq}/{tree}/embedding", data=embedding, compression="lzf",
            )
            outh5.create_dataset(
                f"{demo}/{seq}/{tree}/attr", data=attr, compression="lzf",
            )
    outh5.close()
    rawh5.close()


def convert_graph(edges, VEC_DIMS):
    """
    Converts edgelists to networkx graphs and then fits node2vec model on them.
    Saves embeddings to dictionary of arrays.

    Args:
        edges (np.ndarray): Edgelist single tree in a tree sequence to be embedded.
        VEC_DIMS (int): Number of dimensions of Node2Vec output embeddings. Tuneable.

    Returns:
        np.ndarray: Embedded tree with dimensions nodes x embedded_dim

    Lots of hyperparams to walk through,
    could do grid search or smarter learning through
    iterative training of resulting model that feeds back into this.
    Good chance for meta-learning?
    """
    _g = nx.from_edgelist(edges)
    n2v = Node2Vec(n_components=VEC_DIMS)
    print(n2v)
    emb = n2v.fit_transform(_g)  # More hyperparams here
    print(emb.shape)

    return emb


def main():
    """
    1. Reads in HDF5 data
    2. Converts edgelists to graphs
    3. Computes graph embeddings for each graph
    4. Saves NPZ of graph embeddings for each sample.

    """
    VEC_DIMS = 128
    demo = sys.argv[1]
    h5file = "/overflow/dschridelab/projects/GeneGraphs/embeddings/test1.hdf5"
    iterate_trees(h5file, demo, VEC_DIMS)


if __name__ == "__main__":
    main()
