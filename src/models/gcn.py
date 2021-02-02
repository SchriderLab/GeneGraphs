import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, TransformerConv, GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from vae import BaseModel
import configparser


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


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
    in_channels = [int(x) for x in config.get('encoder_params', 'in_channels').split(',')]
    out_channels = [int(x) for x in config.get('encoder_params', 'out_channels').split(',')]
    if ',' in config.get('encoder_params', 'num_heads'):
        num_heads = [int(x) for x in config.get('encoder_params', 'num_heads').split(',')]
    else:
        num_heads = None

    return Encoder(in_channels, out_channels, int(config.get('encoder_params', 'depth')),
                   num_heads=num_heads)


def get_mlp(config):
    channels = config.get("mlp_params", "channels")
    channels = channels.split(',')
    channels = [int(x) for x in channels]
    return MLP(channels, config.getboolean("mlp_params", "batch_norm"))


def get_pooling(config):
    pool_type = config.get("pooling_params", "pooling_type")
    if pool_type == 'None':
        return None
    else:
        return pool_type


def apply_pooling(pooling, x, batch):
    if pooling == 'global_max_pool':
        return global_max_pool(x, batch)
    elif pooling == 'global_add_pool':
        return global_add_pool(x, batch)
    elif pooling == 'global_mean_pool':
        return global_mean_pool(x, batch)
    else:
        print("specified pooling type is not supported")
        raise Exception


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
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)  # I can make this a general call to any output activation


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, layer_type='nnconv', num_heads=None):
        super(Encoder, self).__init__()
        self.parameters = locals()
        self.layers = nn.ModuleList()

        assert all(len(in_channels) == len(out_channels) == depth)

        for i in range(depth):
            if layer_type == 'nnconv':
                self.layers.append(NNConv(self.parameters['in_channels'][i], self.parameters['out_channels'][i]))
            elif layer_type == 'transformerconv':
                self.layers.append(
                    TransformerConv(self.parameters['in_channels'][i], self.parameters['out_channels'][i],
                                    self.parameters['num_heads'][i]))
            elif layer_type == 'gcnconv':
                self.layers.append(GCNConv(self.parameters['in_channels'][i], self.parameters['out_channels'][i]))
            else:
                print("specified layer type not supported")
                raise Exception

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x
