from data import get_folds
from model import GNN
from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch
from torch import normal, zeros, ones, cat
from tqdm import tqdm

def get_data_loaders(data, test_idx):
    test_data = data[test_idx]
    batch_size = min(len(test_data), 32)
    train_data = []
    for i in range(len(data)):
        if i != test_idx:
            train_data = train_data + data[i]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(rni=False):    
    # load data
    geom_data = get_folds(5)
    metrics = []
    MAX_EPOCH = 200
    model = GNN(50, 16)
    for i in range(5):
        print("trial " + str(i)+"/5...")
        train_loader, test_loader = get_data_loaders(geom_data, i)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=0.0001 if rni else 0.0005)
        criterion = torch.nn.functional.binary_cross_entropy
        epoch_iter = tqdm(range(MAX_EPOCH))
        for epoch in epoch_iter:
            model.train()
            train_loss = 0.0
            num_nodes = 0
            for batch in train_loader:
                model.zero_grad()
                e, x, y = batch.edge_index, batch.x, batch.y
                num_nodes += y.shape[0]
                x2 = randomise_half_columns(x) if rni else x
                output = model(x2, e).squeeze()
                loss = criterion(output, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            avg_train_loss = train_loss / num_nodes
            epoch_iter.set_description(f"Epoch {epoch}/{MAX_EPOCH}, current training loss: {round(avg_train_loss, 4)}")
        print("Testing...")
        model.eval()
        with torch.no_grad():
            test_perf = 0.0
            num_nodes = 0
            for batch in test_loader:
                model.zero_grad()
                e, x, y = batch.edge_index, batch.x, batch.y
                num_nodes += y.shape[0]
                x2 = randomise_half_columns(x) if rni else x
                output = model(x2, e).squeeze()
                batch_perf = (y == output.detach()).sum().item()
                test_perf += batch_perf
        average_test_perf = test_perf / num_nodes
        metrics.append(average_test_perf)
        print(f"Test set performance: {average_test_perf}")

def print_box(text):
    t_len = len(text) + 2
    n = 60 - t_len
    a = n // 2
    b = n - a
    print("*"*60)
    print(("*"*a)+ " " + text + " " + ("*"*b))
    print("*"*60)

def randomise_half_columns(x):
    x_shape = (x.shape[0], x.shape[1]//2)
    rand_shape = (x.shape[0], x.shape[1] - (x.shape[1]//2))
    determ_features = x[:, 0:x_shape[1]]
    rand_features = normal(zeros(rand_shape), ones(rand_shape))
    return cat((determ_features, rand_features), 1)

print_box("BASELINE")
train()
#print_box("RANDOM NODE INITIALIZATION")
#train(True)