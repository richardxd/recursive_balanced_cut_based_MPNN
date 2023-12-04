import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import message_passing_networks as mpn
from utils import visualize

# def load_dataset(train_split, test_split, val_split):
#     '''
#     load the dataset and output train, test, val by specified split
#     '''
#     dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
#     dataset = dataset.shuffle()
#     num_data = len(dataset)
#     train_split = int(num_data * train_split)
#     test_split = int(num_data * test_split)
#     val_split = int(num_data * val_split)
#     train_dataset = dataset[:train_split]
#     test_dataset = dataset[train_split:train_split + test_split]
#     val_dataset = dataset[train_split + test_split:]
#     return train_dataset, test_dataset, val_dataset


def train(model, optimizer, data, criterion):
    '''
    train the model
    '''

    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def eval(model, data):
    '''
    evaluate the model
    '''
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # use the class with highest probability
    test_correct = pred[data.test_mask] == data.y[
        data.test_mask]  # check against ground-truth labels
    test_acc = int(test_correct.sum()) / int(
        data.test_mask.sum())  # Derive ratio of correct predictions
    return test_acc


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    model = mpn.GCN(dataset.num_features,
                    dataset.num_classes,
                    hidden_channels=16).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)
    for epoch in range(1, 201):
        loss = train(model, optimizer, data, criterion)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = eval(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

    # using visualize function from utils.py
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    out.cpu()
    visualize(out, color=data.y.cpu())