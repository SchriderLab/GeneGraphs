import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import h5py
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from tensorflow import keras


def load_data(h5file):
    """
    Loads in HDF5 file of tree sequence data and returns a dict. 

    Args:
        h5file (str, optional): HDF5 file containing tree sequence data. Defaults to "/overflow/dschridelab/10e4_test_infer_FINAL.hdf5".

    Returns:
        dict: Condensed format of data with keys=class labels and values=list of edge list arrays.
    """
    rawh5 = h5py.File(h5file)
    labs = list(rawh5.keys())
    out_dict = {}

    for lab in labs[:1]:
        print(lab)
        out_dict[lab] = []
        for batch in list(rawh5[lab].keys())[:1]:
            for rep in list(rawh5[lab][str(batch)].keys())[:1]:
                edges = rawh5[lab][str(batch)][str(rep)]["edge_index"][:]
                out_dict[lab].append(np.vstack(edges).T)
                # print(np.vstack(edges).shape)
    return out_dict


def convert_graphs(raw_dict, VEC_DIMS):
    """
    Converts edgelists to networkx graphs and then fits node2vec model on them.
    Saves embeddings to dictionary of arrays.

    Args:
        raw_dict (dict): Dictionary of raw data, keys are labels, vals are lists of edge arrays.
        VEC_DIMS (int): Number of dimensions of Node2Vec output embeddings. Tuneable.

    Returns:
        dict: Dictionary of output embeddings, keys are labels, vals are Node2Vec embeddings in array form.
    """
    embedding_dict = {}
    for key, vals in raw_dict.items():
        embedding_dict[key] = []
        for arr in vals:
            print(arr.shape)
            _g = nx.from_edgelist(arr)
            """
            Lots of hyperparams to walk through, 
            could do grid search or smarter learning through 
            iterative training of resulting model that feeds back into this.
            Good chance for meta-learning?
            """
            n2v = Node2Vec(_g, dimensions=VEC_DIMS, workers=1)
            model = n2v.fit()  # More hyperparams here
            embedding_dict[key].append(
                HadamardEmbedder(keyed_vectors=model.wv).as_keyed_vectors()
            )
        print(embedding_dict[key]["0"])
        embedding_dict[key] = np.stack(embedding_dict[key])

    return embedding_dict


def main():
    """
    1. Reads in HDF5 data
    2. Converts edgelists to graphs
    3. Computes graph embeddings for each graph
    4. Saves NPZ of graph embeddings for each sample.

    TODO: Check if these edgelists should be sequences or just individual samps.
    """
    VEC_DIMS = 128
    h5file = "/overflow/dschridelab/10e4_test_infer_FINAL.hdf5"
    outfile = h5file.split(".")[0].split("/")[-1] + ".npz"

    data_dict = load_data(h5file)
    embeddings_dict = convert_graphs(data_dict, VEC_DIMS)

    np.savez(outfile, **embeddings_dict)


if __name__ == "__main__":
    main()
