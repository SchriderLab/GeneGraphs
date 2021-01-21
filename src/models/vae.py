import os.path as osp

import argparse
import torch
import torch_geometric.nn as g_nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, GENConv, GINEConv, NNConv, TransformerConv
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges, batched_negative_sampling
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout

EPS = 1e-5


class FLLReconLoss(nn.Module):
    def __init__(self):
        super(FLLReconLoss, self).__init__()
        self.decoder = InnerProductDecoder()

    def forward(self, z, edge_index, batch):
        pos_loss = -torch.log(
            self.decoder(z, edge_index, sigmoid=True) + EPS).mean()

        neg_edge_index = batched_negative_sampling(edge_index, batch, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]), Dropout(0.25))
        for i in range(1, len(channels))
    ])


# this can be significantly simplified using list comprehensions and the examples
# seen here: https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
# under "dynamic sequential"
# for now, only use activations for non-graph networks
class BaseModel(nn.Module):
    def __init__(self, in_channels, out_channels, depth, layer_type='nnconv', activation=None,
                 o_activation=None, num_edge_features=None, model_type='sequential', num_heads=None):
        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.model_type = model_type
        self.layers = nn.ModuleList()
        if num_edge_features is not None:
            self.num_edge_features = int(num_edge_features)
        self.num_heads = num_heads

        # set hidden layer activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = activation

        # set output activation
        if o_activation == 'softmax':
            self.o_activation = nn.Softmax()
        elif o_activation == 'log_softmax':
            self.o_activation = nn.LogSoftmax()
        else:
            self.o_activation = o_activation

        if depth <= 2:
            self.hidden_channels = self.out_channels
        else:
            self.hidden_channels = self.in_channels * 2

        for i in range(depth):
            # add appropriate layer
            if layer_type == "linear":
                self.layers.append(nn.Linear(self.in_channels, self.hidden_channels))
            elif layer_type == "gcnconv":
                self.layers.append(GCNConv(self.in_channels, self.hidden_channels))
            elif layer_type == "genconv":
                self.layers.append(GENConv(self.in_channels, self.hidden_channels))
            elif layer_type == "gineconv":
                # need to check whether we're getting a channels size explosion for
                # large depths here
                self.layers.append(GINEConv(MLP(self.in_channels, self.in_channels*2,
                                                self.in_channels*4, self.hidden_channels)))
            elif layer_type == "nnconv":
                self.layers.append(NNConv(self.in_channels, self.hidden_channels,
                                            MLP([self.num_edge_features, self.num_edge_features*2,
                                                self.num_edge_features*4, self.in_channels*self.hidden_channels]), aggr='mean'))
            elif layer_type == "transformerconv":
                if i == depth - 1:
                    self.layers.append(TransformerConv(self.in_channels, self.hidden_channels))
                else:
                    self.layers.append(TransformerConv(self.in_channels, self.hidden_channels, self.num_heads))
            else:
                print("Specified layer type not supported")
                raise Exception


            # add activation
            if i != depth - 1:
                if self.activation is not None:
                    self.layers.append(self.activation)
            elif self.o_activation is not None:
                self.layers.append(self.o_activation)

            # resize channels
            if self.model_type != 'variational' or i != depth - 2:
                self.in_channels = self.hidden_channels
                self.hidden_channels *= 2
                # possible to remove this?
                if self.num_heads is not None:
                    self.in_channels *= self.num_heads

            # fixing channel sizes for output layers
            if i == depth - 2 and self.model_type == 'sequential':
                self.hidden_channels = self.out_channels

            if i >= depth - 3 and self.model_type == 'variational':
                self.activation = None
                self.hidden_channels = self.out_channels

    def forward(self, x, edge_index=None):
        if edge_index is not None:
            if self.model_type == 'sequential':
                for layer in self.layers[:-1]:
                    x = layer(x, edge_index).relu()
                x = self.layers[-1](x, edge_index)
                return x
            elif self.model_type == 'variational':
                for layer in self.layers[:-2]:
                    x = layer(x, edge_index).relu()
                return self.layers[-2](x, edge_index), self.layers[-1](x, edge_index)
        elif self.model_type == 'sequential':
            for layer in self.layers:
                x = layer(x)
            return x
        elif self.model_type == 'variational':
            for layer in self.layers[:-2]:
                x = layer(x)
            return self.layers[-2](x), self.layers[-1](x)


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(nn.Module):
    def __init__(self, out_channels):
        super(Discriminator, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# just using this for debugging
if __name__ == "__main__":
    test = BaseModel(5, 12, 6, layer_type='gcnconv', activation='relu', model_type='variational')
    print(test)
    print(test.layers[-1])
    print(test.layers[:-2])
    test2 = BaseModel(24, 12, 3, layer_type='linear', activation='relu', o_activation=None)
    print(test2)