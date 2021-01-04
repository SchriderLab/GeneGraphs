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
    def __init__(self, ifile, models = None, downsample = False, downsample_rate = 0.05):
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

            del self.keys[model][0] # can't just delete this. Need to use on_epoch_end
            skeys = self.ifile[model][key].keys()

            for skey in skeys:
                X.append(np.array(self.ifile[model][key][skey]['x']))
                indices.append(np.array(self.ifile[model][key][skey]['edge_index']))
                y.append(model_index)

        y = torch.LongTensor(np.hstack(y).astype(np.int32))

        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, y

    def on_epoch_end(self):
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        for key in self.keys.keys():
            random.shuffle(self.keys[key])