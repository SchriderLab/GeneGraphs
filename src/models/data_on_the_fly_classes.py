import sys
import numpy as np

from torch_geometric.data import Data, Batch, DataLoader
import torch

import random

# first iteration of basic DataGenerator class
# that grabs a whole tree-sequence as a batch for training an AE or VAE
class DataGeneratorAE(object):
    def __init__(self, ifile, model = 'constant_2pop'):
        self.model = model

        self.keys = list(ifile[self.model].keys())
        self.ifile = ifile

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        ix = 0
        key = self.keys[ix]

        del self.keys[ix]

        X = np.array(self.ifile[self.model][key]['x'])
        indices = np.array(self.ifile[self.model][key]['edge_index'])

        batch = Batch.from_data_list([Data(x = torch.FloatTensor(X[k]), edge_index = torch.LongTensor(indices[k])) for k in range(indices.shape[0])])

        return batch

    def on_epoch_end(self):
        self.keys = list(self.ifile.keys())

class DataGeneratorGCN(object):
    def __init__(self, ifile, models = None):
        if models is None:
            self.models = list(ifile.keys())
            print(self.models)
        else:
            self.models = models

        self.ifile = ifile
        self.on_epoch_end()

    def __len__(self):
        return int(np.min([len(self.keys[u]) for u in self.keys.keys()]))

    def __getitem__(self, index):
        ix = 0

        X = []
        indices = []
        y = []

        for model in self.models:
            model_index = self.models.index(model)
            key = self.keys[model][ix]

            del self.keys[model][ix]

            X.append(np.array(self.ifile[model][key]['x']))
            indices.append(np.array(self.ifile[model][key]['edge_index']))
            y.append(np.ones(len(indices[-1]))*model_index)

        X = np.vstack(X)
        indices = np.vstack(indices)
        y = torch.LongTensor(np.hstack(y).astype(np.int32))

        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(indices.shape[0])])

        return batch, y

    def on_epoch_end(self):
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        for key in self.keys.keys():
            random.shuffle(self.keys[key])





