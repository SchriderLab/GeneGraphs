import os
import logging, argparse

import torch
from data_on_the_fly_classes import DataGenerator

import h5py
import numpy as np
from scipy.sparse import lil_matrix

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# define the numpy version of the sigmoid function
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--odir", default = "test_pred")
    parser.add_argument("--model", default = "None")
    parser.add_argument("--demographic_model", default = "constant_2pop")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    demographic_models = []
    if " " in args.demographic_model:
        demographic_models = args.demographic_model.split(" ")
    else:
        demographic_models = list(args.demographic_model)


    # be on the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    model = torch.load(args.model)
    model = model.to(device)

    # set in eval mode. is necessary for some layers such as some normalization layers (dropout, etc.)
    model.eval()

    # get a generator to scroll through the data, don't downsample it
    # for now we have to just get a sample...even a single batch is too big for CPU RAM
    generator = DataGenerator(h5py.File(args.ifile, 'r'), models = demographic_models, downsample = True)

    for ix in range(len(generator)):
        batch, y = generator[ix]
        batch = batch.to(device)

        with torch.no_grad():
            # keep everything as Numpy arrays in RAM (maybe pricey to send them to the RAM...but way easier)
            z = model.encode(batch.x, batch.edge_index).detach().cpu().numpy()
            n_nodes = z.shape[0]

            edge_index = batch.edge_index.detach().cpu().numpy().astype(np.int32)
            index = batch.batch.detach().cpu().numpy()

            # make
            A = np.zeros((n_nodes, n_nodes))
            for i in range(edge_index.shape[1]):
                A[edge_index[0,i], edge_index[1,i]] = 1

            for i in range(int(np.max(index))):
                # nodes in graph i
                index_ = list(np.where(index == i)[0])

                # graph label
                label = y.detach().cpu().numpy().astype(np.int32)[i]

                # features in graph i
                z_ = z[index_,:]
                # predicted connections
                A_pred_ = sigmoid(z_.dot(z_.T))

                # take the slice of A that only includes the nodes for this graph
                A_ = A[np.ix_(index_, index_)]

                bc_loss = -np.mean(np.multiply(A_.flatten(), np.log(A_pred_.flatten()))
                                   + np.multiply((1 - A_.flatten()), np.log(1 - A_pred_.flatten())))
                accuracy = accuracy_score(A_.flatten(), np.round(A_pred_).flatten())

                assert len(y.detach().numpy()) == round(len(A)/len(A_))
                np.savez(os.path.join(args.odir, '{:06d}_{:06d}.npz'.format(ix, i)), Z = z_, A = A_,
                         A_pred = A_pred_, acc = accuracy, loss = bc_loss, label = label)


















if __name__ == '__main__':
    main()

