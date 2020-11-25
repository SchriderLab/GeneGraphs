import argparse
import torch
import torch.nn.functional as F
import h5py
from data_on_the_fly_classes import DataGeneratorGCN
from gcn import GCN

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from collections import deque

from sklearn.metrics import accuracy_score

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--n_epochs", default = "5")
    parser.add_argument("--lr", default = "0.01")
    parser.add_argument("--weight_decay", default = "5e-4")

    parser.add_argument("--in_features", default = "4")
    parser.add_argument("--out_features", default = "2")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    num_features = int(args.in_features)
    out_channels = int(args.out_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, out_channels)
    model.to(device)

    generator = DataGeneratorGCN(h5py.File(args.ifile, 'r'))
    optimizer = torch.optim.Adam(model.parameters(), lr = float(args.lr))

    losses = deque(maxlen=2000)
    accuracies = deque(maxlen=2000)

    model.train()
    for epoch in range(int(args.n_epochs)):
        for j in range(len(generator)):
            batch, y = generator[j]
            batch = batch.to(device)
            y = y.to(device)

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

            if (j + 1) % 100 == 0:
                print("Epoch: {}/{}, Step: {}, Loss: {:.3f}, Acc: {:.3f}".format(epoch+1,
                                                                       args.n_epochs, j + 1,
                                                                       np.mean(losses), np.mean(accuracies)))
        generator.on_epoch_end()

    # need to change to test_data
    model.eval()
    # _, pred = model(data).max(dim=1)
    # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    # acc = correct / int(data.test_mask.sum())
    # print('Accuracy: {:.4f}'.format(acc))

if __name__ == "__main__":
    main()
