import os
import sys
import itertools

import os
import logging, argparse
import itertools
import numpy as np

import platform
import torch
from torch_geometric.nn import GCNConv, GAE, VGAE

from vae import GCNEncoder, VariationalGCNEncoder, LinearEncoder, VariationalLinearEncoder, FLLReconLoss
from data_on_the_fly_classes import DataGenerator
from collections import deque
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix

from torch_geometric.utils.convert import to_networkx

import h5py

import networkx as nx
import matplotlib.pyplot as plt

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ifile_val", default="None")

    parser.add_argument("--odir", default = "None")

    parser.add_argument("--n_epochs", default = "5")

    parser.add_argument("--in_features", default = "6")
    parser.add_argument("--out_features", default = "16")

    parser.add_argument("--linear", action = "store_true")
    parser.add_argument("--variational", action = "store_true")

    parser.add_argument("--tag", default = "test")

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
            logging.debug('root: cleared output directory {0}'.format(args.odir))

    return args

def main():
    args = parse_args()

    num_features = int(args.in_features)
    out_channels = int(args.out_features)

    if not args.variational:
        if not args.linear:
            model = GAE(GCNEncoder(num_features, out_channels))
        else:
            model = GAE(LinearEncoder(num_features, out_channels))
    else:
        if args.linear:
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        else:
            model = VGAE(VariationalGCNEncoder(num_features, out_channels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    generator = DataGenerator(h5py.File(args.ifile, 'r'))
    validation_generator = DataGenerator(h5py.File(args.ifile_val, 'r'))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    losses = deque(maxlen=2000)
    criterion = FLLReconLoss()

    val_loss = np.inf

    for i in range(int(args.n_epochs)):
        model.train()

        n_steps = len(generator)
        print("Len: ", len(generator)) ## delete this
        for j in range(n_steps):
            batch, y = generator[j]
            batch = batch.to(device)

            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_index)
            loss = criterion(z, batch.edge_index, batch.batch)

            if args.variational:
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (j + 1) % 100 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}".format(i+1,
                                                                       args.n_epochs, j + 1, n_steps,
                                                                        np.mean(losses)))
        generator.on_epoch_end()

        model.eval()

        validation_losses = []

        for j in range(len(validation_generator)):
            batch, y = generator[j]
            batch = batch.to(device)

            with torch.no_grad():
                z = model.encode(batch.x, batch.edge_index)
                loss = criterion(z, batch.edge_index, batch.batch)

                if args.variational:
                    loss = loss + (1 / batch.num_nodes) * model.kl_loss()

                validation_losses.append(loss.item())

        validation_generator.on_epoch_end()

        logging.info('root: Epoch: {}/{}, validation loss: {:.4f}'.format(i+1,
                                                                       args.n_epochs, np.mean(validation_losses)))
        if np.mean(validation_losses) < val_loss:
            torch.save(model, os.path.join(args.odir, '{0}.model'.format(args.tag)))
            val_loss = np.mean(validation_losses)

    if args.odir is not None:
        np.savetxt(os.path.join(args.odir, '{0}.evals'.format(args.tag)), np.array([np.mean(losses), val_loss]))
    else:
        print("Training not saved. No output directory provided.")





if __name__ == '__main__':
    main()