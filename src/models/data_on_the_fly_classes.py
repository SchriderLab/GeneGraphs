import sys
import numpy as np

from torch_geometric.data import Data, Batch, DataLoader
import torch

# first iteration of basic DataGenerator class
# that grabs a whole tree-sequence as a batch for training an AE or VAE
class DataGeneratorAE(object):
    def __init__(self, ifile):
        self.keys = list(ifile.keys())
        self.ifile = ifile

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        ix = 0
        key = self.keys[ix]

        del self.keys[ix]

        X = np.array(self.ifile[key]['x'])
        indices = np.array(self.ifile[key]['edge_index'])

        batch = Batch.from_data_list([Data(x = torch.FloatTensor(X[k]), edge_index = torch.LongTensor(indices[k])) for k in range(indices.shape[0])])

        return batch



