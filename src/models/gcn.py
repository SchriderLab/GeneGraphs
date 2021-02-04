import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
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
                   config.get('encoder_params', 'layer_type'), num_heads=num_heads)


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
    def __init__(self, in_channels, out_channels, depth, layer_type, num_heads=None):
        super(Encoder, self).__init__()
        self.parameters = locals()
        self.layers = nn.ModuleList()

        assert all(len(in_channels) == len(out_channels) == depth)

        for i in range(depth):
            if layer_type == 'genconv':
                self.layers.append(GENConv(self.parameters['in_channels'][i], self.parameters['out_channels'][i]))
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

        assert all(len(self.in_channels) == len(self.out_channels) == self.depth)

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


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, depth: int, normalize=False):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.normalize = normalize
        self.layers = nn.ModuleList()

        assert all(len(self.in_channels) == len(self.out_channels) == self.depth)

        for i in range(self.depth):
            self.layers.append(GCNConv(self.in_channels[i], self.out_channels[i], normalize=self.normalize))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, num_heads, depth: int, negative_slope=.2, dropout=0.0):
        super(GATEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = depth
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.layers = nn.ModuleList()

        assert all(len(self.in_channels) == len(self.out_channels) == self.depth)

        for i in range(self.depth):
            if isinstance(self.num_heads, list):
                if self.edge_dim is not None:
                    self.layers.append(GATConv(self.in_channels[i], self.out_channels[i], self.num_heads[i],
                                               negative_slope=self.negative_slope, dropout=self.dropout))
                else:
                    self.layers.append(GATConv(self.in_channels[i], self.out_channels[i],
                                               self.num_heads[i], negative_slope=self.negative_slope,
                                               dropout=self.dropout))
            elif self.edge_dim is not None:
                self.layers.append(GATConv(self.in_channels[i], self.out_channels[i],
                                           negative_slope=self.negative_slope, dropout=self.dropout))
            else:
                self.layers.append(GATConv(self.in_channels[i], self.out_channels[i],
                                           negative_slope=self.negative_slope, dropout=self.dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index).relu()
        return x
