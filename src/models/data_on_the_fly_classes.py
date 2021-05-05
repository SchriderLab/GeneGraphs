import numpy as np
from torch_geometric.data import Data, Batch, DataLoader
import torch
import random


class DataGenerator(object):
    def __init__(self, ifile, sequences=False, models=None, n_samples_per=5):
        if models is None:
            self.models = list(ifile.keys())
        else:
            self.models = models

        # hdf5 file we are reading from
        self.ifile = ifile

        # how many tree sequences from each demographic model are included in a batch?
        self.n_samples_per = n_samples_per

        # if true, we predict on entire tree sequences, otherwise we predict on each tree
        self.sequences = sequences

        # shuffle the keys / make sure they are all there (keys are deleted
        # during training or validation so as not to repeat samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(np.min([len(self.keys[u]) for u in self.keys.keys()]) / self.n_samples_per))

    def __getitem__(self, index):
        # features, edge_indices, and label
        X = []
        indices = []
        y = []

        ranges = []

        # only used if we're training on sequences
        trees_in_sequence = []

        for model in self.models:
            # grab n_samples_per for each model
            for ix in range(self.n_samples_per):
                model_index = self.models.index(model)
                key = self.keys[model][0]

                del self.keys[model][0]
                skeys = self.ifile[model][key].keys()

                # for each tree
                i = 0
                tree_ranges = np.array([])
                for skey in skeys:
                    X.append(np.array(self.ifile[model][key][skey]['x']))
                    indices.append(np.array(self.ifile[model][key][skey]['edge_index']))
                    tree_ranges = np.append(tree_ranges, self.ifile[model][key][skey]["tree_range"])

                    if not self.sequences:
                        y.append(model_index)

                    i += 1

                if self.sequences:
                    y.append(model_index)
                    trees_in_sequence.append(i)

                ranges.append(tree_ranges)

        y = torch.LongTensor(np.hstack(y).astype(np.float32))

        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, y, trees_in_sequence, ranges

    def on_epoch_end(self):
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        for key in self.keys.keys():
            random.shuffle(self.keys[key])