import os
import sys
import itertools

import os
import logging, argparse
import itertools
import numpy as np

import platform
import torch
from torch_geometric.nn import GCNConv, GAE, VGAE, ARGVA, ARGA

from vae import BaseModel, GCNEncoder, VariationalGCNEncoder, LinearEncoder, VariationalLinearEncoder, FLLReconLoss, Discriminator, TransformerEncoder
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
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--ifile_val", default="None")

    parser.add_argument("--odir", default="None")

    parser.add_argument("--n_epochs", default="5")

    parser.add_argument("--in_features", default="6", nargs='+')
    parser.add_argument("--out_features", default="16", nargs='+')

    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--depth", default="2")
    parser.add_argument("--layer_type", default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--o_activation", default=None)
    parser.add_argument("--num_edge_features", default=None)
    parser.add_argument("--model_type", default="sequential")
    parser.add_argument("--num_heads", default=None, nargs='+')

    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--variational", action="store_true")

    parser.add_argument("--adversarial", action="store_true")

    parser.add_argument("--tag", default="test")

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


def train_autoencoder(args, encoder, generator, validation_generator, device, criterion, losses, val_loss):

    if args.model_type == 'variational':
        model = VGAE(encoder)
    else:
        model = GAE(encoder)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(int(args.n_epochs)):
        model.train()

        n_steps = len(generator)
        print("Len: ", len(generator))
        for j in range(n_steps):
            batch, y = generator[j]
            batch = batch.to(device)

            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_index)
            loss = criterion(z, batch.edge_index, batch.batch)

            if args.model_type == 'variational':
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (j + 1) % 100 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}".format(i + 1,
                                                                                    args.n_epochs, j + 1, n_steps,
                                                                                    np.mean(losses)))
        generator.on_epoch_end()

        validation(model, i, validation_generator, criterion, args, device, losses)


def train_adversarial_autoencoder(args, encoder, generator, validation_generator, device, criterion, losses, val_loss):

    if args.model_type == 'variational':
        model = ARGVA(encoder, Discriminator(int(args.out_features)))
    else:
        model = ARGA(encoder, Discriminator(int(args.out_features)))
    model = model.to(device)

    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.01)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=0.01)

    for i in range(int(args.n_epochs)):
        model.train()

        n_steps = len(generator)
        print("Len: ", len(generator))
        for j in range(n_steps):
            batch, _ = generator[j]
            batch = batch.to(device)

            encoder_optimizer.zero_grad()
            z = model.encode(batch.x, batch.edge_index)

            # discriminator optimization
            for _ in range(5):
                model.discriminator.train()
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(z)
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # encoder optimization
            loss = criterion(z, batch.edge_index, batch.batch)
            if args.model_type == 'variational':
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()
            loss.backward()
            encoder_optimizer.step()
            losses.append(loss.item())

            if (j + 1) % 100 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}".format(i+1,
                                                                       args.n_epochs, j + 1, n_steps,
                                                                        np.mean(losses)))
        generator.on_epoch_end()

        validation(model, i, validation_generator, criterion, args, device, losses)


def validation(model, i, validation_generator, criterion, args, device, losses):

    model.eval()
    val_loss = np.inf

    validation_losses = []
    validation_disc_losses = []

    for j in range(len(validation_generator)):
        batch, _ = validation_generator[j]
        batch = batch.to(device)

        with torch.no_grad():
            z = model.encode(batch.x, batch.edge_index)
            if args.model_type == 'adversarial':
                valid_disc_loss = model.discriminator_loss(z)
            loss = criterion(z, batch.edge_index, batch.batch)

            if args.model_type == 'variational':
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()

            validation_losses.append(loss.item())
            if args.model_type == 'adversarial':
                validation_disc_losses.append(valid_disc_loss.item())

    validation_generator.on_epoch_end()
    if args.model_type == 'adversarial':
        logging.info('root: Epoch: {}/{}, validation loss: {:.4f}, discriminator loss: {:.4f}'.format(i + 1,
                                                                      args.n_epochs, np.mean(validation_losses), np.mean(valid_disc_loss)))
    else:
        logging.info('root: Epoch: {}/{}, validation loss: {:.4f}'.format(i + 1, args.n_epochs, np.mean(validation_losses)))

    if np.mean(validation_losses) < val_loss:
        torch.save(model, os.path.join(args.odir, '{0}.model'.format(args.tag)))
        val_loss = np.mean(validation_losses)

    if args.odir is not None:
        np.savetxt(os.path.join(args.odir, '{0}.evals'.format(args.tag)), np.array([np.mean(losses), val_loss]))
    else:
        print("Training not saved. No output directory provided.")


def main():
    args = parse_args()

    if args.transformer:
        num_features = [int(x) for x in args.in_features]
        out_channels = [int(x) for x in args.out_features]
        depth = int(args.depth)
        num_heads = [int(x) for x in args.num_heads]

    else:
        num_features = int(args.in_features)
        out_channels = int(args.out_features)

    if args.base:
        if args.num_heads is not None:
            num_heads = int(args.num_heads[0])
        else:
            num_heads = args.num_heads
        encoder = BaseModel(num_features, out_channels, int(args.depth), args.layer_type,
                            args.activation, args.o_activation, args.num_edge_features,
                            args.model_type, num_heads)

    else:
        if args.transformer:
            encoder = TransformerEncoder(num_features, out_channels, num_heads, depth)
        elif args.variational:
            if args.linear:
                encoder = VariationalLinearEncoder(num_features, out_channels)
            else:
                encoder = VariationalGCNEncoder(num_features, out_channels)
        else:
            if not args.linear:
                encoder = GCNEncoder(num_features, out_channels)
            else:
                encoder = LinearEncoder(num_features, out_channels)
    print(encoder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = DataGenerator(h5py.File(args.ifile, 'r'))
    validation_generator = DataGenerator(h5py.File(args.ifile_val, 'r'))

    losses = deque(maxlen=2000)
    val_loss = np.inf
    criterion = FLLReconLoss()

    if args.adversarial:
        train_adversarial_autoencoder(args, encoder, generator, validation_generator, device, criterion, losses,
                                      val_loss)
    else:
        train_autoencoder(args, encoder, generator, validation_generator, device, criterion, losses, val_loss)


if __name__ == '__main__':
    main()
