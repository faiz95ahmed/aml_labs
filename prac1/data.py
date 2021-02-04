import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as graph_hash
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data as GeomData

# generate a graph with cycle of length n
def full_cycle(n):
    G = nx.Graph()
    nx.add_cycle(G, range(n))
    return G

# generate a graph with 2 cycles of size i and j
def split_cycle(i, j):
    G = nx.Graph()
    nx.add_cycle(G, range(i))
    nx.add_cycle(G, range(i, i+j))
    return G

# generate a list of pairs of (full cycle, split cycle) for all valid split cycle conformations
def cycle_graphs(n):
    return [(full_cycle(n), split_cycle(i, n-i)) for i in range(3, n-2)]

def to_torch_geom(g):
    forward_edges = list(g.edges)
    backward_edges = [(v, u) for u, v in forward_edges]
    edges = forward_edges + backward_edges
    edge_index = torch.LongTensor(edges).t().contiguous()
    num_nodes = max(list(g.nodes))+1
    features_np = np.zeros((num_nodes, 50))
    first_cycle = nx.find_cycle(g)
    if (len(first_cycle) == num_nodes):
        label = torch.Tensor([1]*num_nodes)
    else:
        label = torch.Tensor([0]*num_nodes)
    return GeomData(edge_index=edge_index, x=torch.Tensor(features_np), y=label)
    # return (edge_index, torch.Tensor(features_np), label)

def get_geom_data():
    # generate all cycle_graphs
    all_cycle_graphs = dict([(i, cycle_graphs(i)) for i in range(6, 16)])
    ###########################################################
    ## sanity check ###########################################
    sane = True
    for v in all_cycle_graphs.values():
        for g1, g2 in v:
            sane = sane and (graph_hash(g1) == graph_hash(g2))
    print(sane)
    ###########################################################
    # convert to torch_geometric.data.Data objects
    geom_data_pairs = []
    for v in all_cycle_graphs.values():
        for g1, g2 in v:
            geom_data_pairs.append((to_torch_geom(g1), to_torch_geom(g2)))
    return geom_data_pairs

def get_folds(num_folds):
    data = get_geom_data()
    num_pairs = len(data)
    fold_size = num_pairs // num_folds
    folds = {}
    for i in range(num_folds):
        final_fold = i == (num_folds - 1)
        if final_fold:
            fold_pairs = data[i*fold_size:]
        else:
            fold_pairs = data[i*fold_size:(i+1)*fold_size]
        folds[i] = []
        for g1, g2 in fold_pairs:
            folds[i].append(g1)
            folds[i].append(g2)
    return folds

