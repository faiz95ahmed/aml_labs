import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class GNN(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GNN, self).__init__()
        # GCNConv layers use summation for aggregation
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([GCNConv(input_dim, input_dim) for i in range(num_layers)])
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim),
                                 nn.ReLU(),
                                 nn.Linear(input_dim, input_dim),
                                 nn.ReLU(),
                                 nn.Linear(input_dim, 1),
                                 nn.Sigmoid())
        # construct network

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, batch):
        x_intermediate = x
        for i in range(self.num_layers):
            x_intermediate = F.relu(self.conv_layers[i](x_intermediate, edge_index))
        pooled = global_add_pool(x_intermediate, batch)
        pred_label = self.mlp(pooled)
        return pred_label