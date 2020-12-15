import sys
import numpy as np

from torch_geometric.data import Data, Batch, DataLoader
import torch

import sys
import numpy as np

from torch_geometric.data import Data, Batch, DataLoader
import torch

import random

class DataGenerator(object):
    def __init__(self, ifile, models = None, downsample = True, downsample_rate = 0.05):
        if models is None:
            self.models = list(ifile.keys())
        else:
            self.models = models

        self.downsample = downsample
        self.downsample_rate = downsample_rate
        self.ifile = ifile

        self.on_epoch_end()

    def __len__(self):
        return int(np.min([len(self.keys[u]) for u in self.keys.keys()]))

    def __getitem__(self, index):
        X = []
        indices = []
        y = []

        for model in self.models:
            model_index = self.models.index(model)
            key = self.keys[model][0]

            del self.keys[model][0]
            l = self.ifile[model][key]['x'].shape[0]

            if self.downsample:
                ix = list(np.random.choice(range(l), int(np.round(l * self.downsample_rate)), replace = False))
            else:
                ix = list(range(l))

            X.append(np.array(self.ifile[model][key]['x'])[ix])
            indices.append(np.array(self.ifile[model][key]['edge_index'])[ix])
            y.append(np.ones(len(ix))*model_index)

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