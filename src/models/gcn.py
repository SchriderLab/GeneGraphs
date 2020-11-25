import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels*2)
        self.conv2 = GCNConv(in_channels*2, in_channels*4)

        self.mlp = Seq(
            MLP([in_channels*4, 32]), Dropout(0.5), MLP([32, 64]), Dropout(0.5),
            Lin(64, out_channels))

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        out = global_max_pool(x, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)




