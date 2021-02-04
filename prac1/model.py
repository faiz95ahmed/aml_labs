import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GNN, self).__init__()
        # GCNConv layers use summation for aggregation
        self.conv_layers = nn.ModuleList([GCNConv(input_dim, input_dim) for i in range(num_layers)])
        self.pool = lambda x: x.sum(axis=0)
        self.head = nn.Linear(input_dim, 1)
        # construct network

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        x_intermediate = x
        for i in range(16):
            x_intermediate = F.relu(self.conv_layers[i](x_intermediate, edge_index))
        return F.relu(self.head(x_intermediate))
