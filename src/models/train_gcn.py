import argparse
import torch
import torch.nn.functional as F
import h5py
import configparser
from data_on_the_fly_classes import DataGenerator
from gcn import GCN, Classifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--ifile_val", default="None")
    parser.add_argument("--config", default="None")

    parser.add_argument("--odir", default="None")

    parser.add_argument("--n_epochs", default="5")
    parser.add_argument("--lr", default="None") # lr is specified in configs file
    parser.add_argument("--weight_decay", default="None")

    # parser.add_argument("--in_features", default = "6")
    # parser.add_argument("--out_features", default = "2")

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

    # # input features
    # num_features = int(args.in_features)
    # # number of classes (demographic models) to predict
    # out_channels = int(args.out_features)
    config = configparser.ConfigParser()
    config.read(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(config)
    model.to(device)

    # data generator objects for training and validation respectively
    generator = DataGenerator(h5py.File(args.ifile, 'r'))
    validation_generator = DataGenerator(h5py.File(args.ifile_val, 'r'))

    # default optimizer for now
    if args.lr != "None":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_params", "lr")))

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)

    for epoch in range(int(args.n_epochs)):
        model.train()
        # change back to 1000
        for j in range(len(generator)):
            batch, y = generator[j]
            batch = batch.to(device)
            y = y.to(device)

            # print(y.shape)

            optimizer.zero_grad()
            y_pred = model(batch.x, batch.edge_index, batch.batch)

            loss = F.nll_loss(y_pred, y)

            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)
            accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()

            # change back to 100
            if (j + 1) % 100 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}, Loss: {:.3f}, Acc: {:.3f}".format(epoch+1,
                                                                       args.n_epochs, j + 1,
                                                                        np.mean(losses), np.mean(accuracies)))

        generator.on_epoch_end()

        val_losses = []
        val_accs = []

        model.eval()

        with torch.no_grad():
            for j in range(len(validation_generator)):
                batch, y = validation_generator[j]
                batch = batch.to(device)
                y = y.to(device)

                y_pred = model(batch.x, batch.edge_index, batch.batch)
                loss = F.nll_loss(y_pred, y)

                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                y_pred = np.argmax(y_pred, axis=1)

                val_accs.append(accuracy_score(y, y_pred))
                val_losses.append(loss.detach().item())

        logging.info('root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(epoch + 1, np.mean(val_losses), np.mean(val_accs)))
        
        validation_generator.on_epoch_end()



if __name__ == "__main__":
    main()