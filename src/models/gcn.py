import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
import configparser
import numpy as np


# function to create MLP model
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


# simple GCN model with 2 GC layers and a MLP
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels * 2)
        self.conv2 = GCNConv(in_channels * 2, in_channels * 4)

        self.mlp = Seq(
            MLP([in_channels * 4, 32]), Dropout(0.5), MLP([32, 64]), Dropout(0.5),
            Lin(64, out_channels))

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        out = global_max_pool(x, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


def get_encoder(config):
    """Takes config file and returns the model that that config file specifies"""

    in_channels = [int(x) for x in config.get('encoder_params', 'in_channels').split(',')]
    out_channels = [int(x) for x in config.get('encoder_params', 'out_channels').split(',')]
    if ',' in config.get('encoder_params', 'num_heads'):
        num_heads = [int(x) for x in config.get('encoder_params', 'num_heads').split(',')]
    else:
        num_heads = None

    layer_type = config.get("encoder_params", "layer_type")

    if layer_type == "transformer_conv":
        if config.get("transformer_params", "edge_dim") == "None":
            return TransformerEncoder(in_channels, out_channels, depth=int(config.get('encoder_params', 'depth')),
                                      num_heads=num_heads, dropout=float(config.get("transformer_params", "dropout")))
        else:
            return TransformerEncoder(in_channels, out_channels, depth=int(config.get('encoder_params', 'depth')),
                                  num_heads=num_heads, edge_dim=int(config.get("transformer_params", "edge_dim")),
                                  dropout=float(config.get("transformer_params", "dropout")))
    elif layer_type == "gat_conv":
        return GATEncoder(in_channels, out_channels, depth=int(config.get('encoder_params', 'depth')),
                          num_heads=num_heads, negative_slope=float(config.get("gat_params", "negative_slope")),
                          dropout=float(config.get("gat_params", "dropout")))
    elif layer_type == "gcn_conv":
        return GCNEncoder(in_channels, out_channels, int(config.get('encoder_params', 'depth')),
                          normalize=config.getboolean("gcn_params", "normalize"))


def get_mlp(config):
    """takes config file and returns the MLP that that file specifies"""

    channels = config.get("mlp_params", "channels")
    channels = channels.split(',')
    channels = [int(x) for x in channels]
    return MLP(channels, config.getboolean("mlp_params", "batch_norm"))


def get_pooling(config):
    """takes config file and returns the pooling layer that that file specifies"""

    pool_type = config.get("pooling_params", "pooling_type")
    if pool_type == 'None':
        return None
    else:
        return pool_type


def apply_pooling(pooling, x, batch):
    """given pooling layer type, data, and batch, returns the data with specified pooling applied to it"""

    if pooling == 'global_max_pool':
        return global_max_pool(x, batch)
    elif pooling == 'global_add_pool':
        return global_add_pool(x, batch)
    elif pooling == 'global_mean_pool':
        return global_mean_pool(x, batch)
    else:
        print("specified pooling type is not supported")
        raise Exception


# generic classifier that takes any config file and constructs it
class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.encoder = get_encoder(config)
        self.pooling = get_pooling(config)
        self.mlp = get_mlp(config)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index)
        if self.pooling is not None:
            x = apply_pooling(self.pooling, x, batch)
        """ data is put into (num_trees_in_batch, embedding_dim) """
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


# Graph Transformer Model called on by a config maker
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, depth: int, num_heads=1, edge_dim=None, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = depth
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.layers = nn.ModuleList()

        assert len(self.in_channels) == len(self.out_channels) == self.depth

        for i in range(self.depth):
            if isinstance(self.num_heads, list):
                if self.edge_dim is not None:
                    self.layers.append(TransformerConv(self.in_channels[i], self.out_channels[i],
                                                       self.num_heads[i], edge_dim=self.edge_dim, dropout=self.dropout))
                else:
                    self.layers.append(TransformerConv(self.in_channels[i], self.out_channels[i],
                                                       self.num_heads[i], dropout=self.dropout))
            elif self.edge_dim is not None:
                self.layers.append(TransformerConv(self.in_channels[i], self.out_channels[i], edge_dim=self.edge_dim,
                                                   dropout=self.dropout))
            else:
                self.layers.append(TransformerConv(self.in_channels[i], self.out_channels[i], dropout=self.dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x


# GCN model called on by a config maker
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, depth: int, normalize=False):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.normalize = normalize
        self.layers = nn.ModuleList()

        len(self.in_channels) == len(self.out_channels) == self.depth

        for i in range(self.depth):
            self.layers.append(GCNConv(self.in_channels[i], self.out_channels[i], normalize=self.normalize))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x


# GAT model called on by a config maker
class GATEncoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, depth: int, num_heads, negative_slope=.2, dropout=0.0):
        super(GATEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = depth
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.layers = nn.ModuleList()

        assert len(self.in_channels) == len(self.out_channels) == self.depth

        for i in range(self.depth):
            if isinstance(self.num_heads, list):
                self.layers.append(GATConv(self.in_channels[i], self.out_channels[i], heads=self.num_heads[i],
                                               negative_slope=self.negative_slope, dropout=self.dropout))
            else:
                self.layers.append(GATConv(self.in_channels[i], self.out_channels[i],
                                           negative_slope=self.negative_slope, dropout=self.dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x


# in progress for creating generic model for classifying tree sequences via an LSTM
class SequenceClassifier(nn.Module):
    def __init__(self, config, batch_size, input_size):
        super(SequenceClassifier, self).__init__()
        self.encoder = get_encoder(config)
        self.pooling = get_pooling(config)
        self.batch_size = batch_size
        self.input_size = input_size
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=16, num_layers=32, batch_first=True)
        self.lstm_pooling = nn.MaxPool1d(3, stride=2)
        self.relu = nn.ReLU()
        self.pad_dim_1 = 70 # num of trees in sequence
        self.linear = nn.Linear(490, 20)
        self.batch_norm = nn.BatchNorm1d(20)
        self.output = nn.Linear(20, 10) # need this to output num classes

    def init_hidden(self):
        return torch.randn(32, self.batch_size, 16), torch.randn(32, self.batch_size, 16)

    def repeat_trees(self, tensor_slices, ranges, pad_size):
        """ This function is incomplete and does not produce the correct
            output. It attempts to pad sequences by repeating trees based
            on their coverage over the sequence, rather than padding with
            merely a certain integer like -1 as is currently implemented"""
        new_tensor_slices = []
        for sequence, range_array in zip(tensor_slices, ranges):
            new_sequence = torch.tensor([])
            for i, (tree, tree_range) in enumerate(zip(sequence, range_array)):
                for repeat in range(int(tree_range * pad_size)+1):
                    if i == 0 and repeat == 0:
                        new_sequence = tree
                    else:
                        new_sequence = torch.vstack((new_sequence, tree))
            new_tensor_slices.append(new_sequence)
        for t1, t2 in zip(tensor_slices, new_tensor_slices):
            print("old: " + str(t1.shape))
            print("new: " + str(t2.shape))

        return new_tensor_slices

    def forward(self, x, edge_index, batch, trees_in_sequence, device, ranges, first_batch=False):
        x = self.encoder(x, edge_index)
        if self.pooling is not None:
            x = apply_pooling(self.pooling, x, batch)

        spot = 0
        tensor_slices = []

        for i in range(len(trees_in_sequence)):
            if i == 0:
                pad_dim_one = self.pad_dim_1 - x[spot:spot+trees_in_sequence[i]].shape[0]

                first_pad = torch.ones((pad_dim_one, 16)) * -1
                tensor_slices.append(torch.cat((x[spot:spot+trees_in_sequence[i]], first_pad)))
            else:
                tensor_slices.append(x[spot:spot+trees_in_sequence[i]])
            # tensor_slices.append(x[spot:spot + trees_in_sequence[i]])

            spot += trees_in_sequence[i]

        # repeat_trees = self.repeat_trees(tensor_slices, ranges, self.pad_dim_1)

        """ data is put into (50, 70, x) or (batch_size, padded_num_trees, features_per_tree) """
        x_new = torch.nn.utils.rnn.pad_sequence(tensor_slices, batch_first=True, padding_value=-1.).to(device)
        x_new, _ = self.lstm(x_new, self.init_hidden())
        x_new = self.lstm_pooling(x_new)
        x_new = self.relu(x_new)
        x_new = x_new.reshape(self.batch_size, -1)
        x_new = self.linear(x_new)
        x_new = self.batch_norm(x_new)
        x_new = self.output(x_new)
        return F.log_softmax(x_new, dim=1)

