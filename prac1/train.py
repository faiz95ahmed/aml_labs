from data import get_folds
from model import GNN
from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch
from torch import normal, zeros, ones, cat, ge
from tqdm import tqdm

def get_data_loaders(data, test_idx):
    test_data = data[test_idx]
    batch_size = min(len(test_data), 32)
    train_data = []
    for i in range(len(data)):
        if i != test_idx:
            train_data = train_data + data[i]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(rni=False, num_repeats=1, num_layers=16):    
    # load data
    num_folds = 5
    geom_data = get_folds(num_folds)
    metrics = []
    model = GNN(50, num_layers)
    for i in range(num_folds*num_repeats):
        print("trial " + str(i+1)+"/"+str(num_folds*num_repeats)+"...")
        learning_rate_inv_mul = 5 * (torch.pow(torch.tensor(10), torch.tensor((i // num_folds)/(num_repeats - 1))).item()) if rni else 1
        MAX_EPOCH = 40 * int(learning_rate_inv_mul)
        train_loader, test_loader = get_data_loaders(geom_data, i % num_folds)
        model.reset_parameters()
        learning_rate = 0.001 / learning_rate_inv_mul
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.functional.binary_cross_entropy
        epoch_iter = tqdm(range(MAX_EPOCH))
        for epoch in epoch_iter:
            model.train()
            train_loss = 0.0
            num_nodes = 0
            for batch in train_loader:
                model.zero_grad()
                e, x, y, b = batch.edge_index, batch.x, batch.y, batch.batch
                num_nodes += y.shape[0]
                x2 = randomise_half_columns(x) if rni else x
                output = model(x2, e, b)
                loss = criterion(output, y.unsqueeze(1), reduction='sum')
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            avg_train_loss = train_loss / num_nodes
            epoch_iter.set_description(f"Epoch {epoch+1}/{MAX_EPOCH}, current training loss: {round(avg_train_loss, 4)}")
        print("Testing...")
        model.eval()
        with torch.no_grad():
            test_perf = 0.0
            num_nodes = 0
            for batch in test_loader:
                model.zero_grad()
                e, x, y, b = batch.edge_index, batch.x, batch.y, batch.batch
                num_nodes += y.shape[0]
                x2 = randomise_half_columns(x) if rni else x
                output = model(x2, e, b)
                rounded_output = ge((output.detach().squeeze()), 0.5).float()
                batch_perf = (y == rounded_output).sum().item()
                test_perf += batch_perf
        average_test_perf = test_perf / num_nodes
        metrics.append((MAX_EPOCH, learning_rate, average_test_perf))
        print(f"Test set performance: {average_test_perf}")
    print_box("ALL METRICS, RNI: " + str(rni)+", NUM LAYERS: "+str(num_layers))
    for i in range(len(metrics) // num_folds):
        lr = metrics[i*num_folds][1]
        epochs = metrics[i*num_folds][0]
        perf = sum([p for _, _, p in metrics[i*num_folds:(i+1)*num_folds]])/num_folds
        print("epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", average test performance: "+str(round(perf, 4)))

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

# print_box("BASELINE")
# train()
# print_box("RANDOM NODE INITIALIZATION")
# train(True, 10)
print_box("RANDOM NODE INITIALIZATION, 4 LAYERS")
train(True, 2, 4)
print_box("RANDOM NODE INITIALIZATION, 8 LAYERS")
train(True, 2, 8)