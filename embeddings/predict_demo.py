import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
import h5py
from tqdm import tqdm

import plot_utils as pu

import node_network as nodenet


def load_data():
    h5filelist = [
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_constant_2pop.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_uni_AB.hdf5",
    ]

    samps = []
    labs = []
    lab_dict = {}
    for i in range(len(h5filelist)):
        raw_h5_data = h5py.File(h5filelist[i], "r")
        h5_data = raw_h5_data[list(raw_h5_data.keys())[0]]
        seqs = list(h5_data.keys())
        lab_dict[h5filelist[i].split("/")[-1].split(".")[0]] = i
        for j in seqs:
            samps.append(h5_data[j])
            labs.append(i)

    return lab_dict, samps, labs


def test_model(model, PADDING, samps, labs, modname, device):
    model.eval()
    test_X_list = []
    for j in tqdm(samps):
        tree_list = []
        for tree_id in list(j.keys()):
            tree_list.append(
                nodenet.pad_nparr(np.array(j[tree_id]["embedding"]), PADDING)
            )
        test_X_list.append(nodenet.pad_3d_nparr(np.stack(tree_list), PADDING))

    test_arr = np.stack(test_X_list)
    print(test_arr.shape)

    with torch.no_grad():
        test_out = model(torch.from_numpy(test_arr).float().to(device)).float()
    probs = list(torch.sigmoid(test_out).cpu().numpy())

    nodenet.evaluate_model(probs, labs, modname)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nodenet.CNNClassifier().float().to(device)
    model.load_state_dict(torch.load("cnn_10class_node_vec.pth"))

    lab_dict, samps, labs = load_data()
    test_model(model, 100, samps, labs, "cnn", device)


if __name__ == "__main__":
    main()