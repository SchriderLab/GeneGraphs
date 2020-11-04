import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import torch
from torch_geometric.nn import GCNConv, GAE, VGAE

from vae import GCNEncoder, VariationalGCNEncoder, LinearEncoder, VariationalLinearEncoder
from data_on_the_fly_classes import DataGeneratorAE

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
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--n_epochs", default = "5")

    parser.add_argument("--in_features", default = "4")
    parser.add_argument("--out_features", default = "16")

    parser.add_argument("--linear", action = "store_true")
    parser.add_argument("--variational", action = "store_true")

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

    generator = DataGeneratorAE(h5py.File(args.ifile, 'r'))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    model.train()

    for i in range(int(args.n_epochs)):
        for j in range(len(generator)):
            batch = generator[j].to(device)

            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_index)
            loss = model.recon_loss(z, batch.edge_index)
            if args.variational:
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()

            print('Epoch {0}, step {1}: got loss of {2}'.format(i, j, loss.item()))

            loss.backward()
            optimizer.step()

        generator.on_epoch_end()


if __name__ == '__main__':
    main()