import os
import sys
import yaml

import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

# from tqdm import tqdm

import wandb

sys.path.append('../')
import mylogger

os.environ['WANDB_SILENT'] = 'true'
DEVICE = 'cuda'


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, gnn_layer_hidden, last_layer_hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, gnn_layer_hidden)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(gnn_layer_hidden, 1)
        self.dropout = nn.Dropout(0.8)
        self.fc1 = nn.LazyLinear(last_layer_hidden)
        self.fc2 = nn.Linear(last_layer_hidden, 1)
        # self.fc = nn.LazyLinear(1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.squeeze(x, 1)
        x = self.fc1(x)
        prediction = self.fc2(x)
        return torch.squeeze(prediction)


# class GCNModel(torch.nn.Module):
#     def __init__(self, in_channels, gnn_layer_hidden_dim, last_layer_hidden_dim):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, gnn_layer_hidden_dim)
#         self.conv2 = SAGEConv(gnn_layer_hidden_dim, 1)
#         self.relu = nn.ReLU()
#         self.fc = nn.LazyLinear(1)

#         # self.fc1 = nn.LazyLinear(last_layer_hidden_dim)
#         # self.fc2 = nn.Linear(last_layer_hidden_dim, 1)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = torch.squeeze(x, 1)
#         prediction = self.fc(x)
#         # x = self.fc1(x)
#         # prediction = self.fc2(x)
#         return prediction


class MDGraphDataset(torchdata.Dataset):
    def __init__(self, dfs, cnadicts, datadirs):
        # 0sec の値が異常値なので取り除く
        self.dfs = dfs
        self.cnadicts = cnadicts
        self.datadirs = datadirs
        self.datalen = 0
        self.datalens = []
        for df in self.dfs:
            self.datalens.append(self.datalen)
            self.datalen += len(df)
        self.datalens.append(np.inf)

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx: int):
        new_idx, df, cnadict, datadir = self.fetch_object_by_idx(idx)
        step = df.iloc[new_idx, 0]  # Step
        cna = cnadict[step]
        x, edge_index = load_data(step, datadir)
        target = df.iloc[new_idx, 3]  # PotEng
        x = np.concatenate([x, cna[:, np.newaxis]], 1)
        x = to_tensor(x).float()
        edge_index = to_tensor(edge_index)
        target = to_tensor(target).float().float()

        return x, edge_index, target

    # idx に対して適切なオブジェクトを選択する
    def fetch_object_by_idx(self, idx: int):
        for i, length in enumerate(self.datalens):
            if idx < self.datalens[i + 1]:
                new_idx = idx - length
                return new_idx, self.dfs[i], self.cnadicts[i], self.datadirs[i]


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_cna(cna_path):
    cna_dict = dict()
    for file in cna_path.iterdir():
        with open(file, 'r') as f:
            li = np.array([int(s.strip()) for s in f.readlines()])
            li = np.where(li >= 1, 1.0, 0.0)
        a = int(file.stem)
        cna_dict[a] = li
    return cna_dict


def load_data(step, datadir):
    feature_path = datadir / 'x' / f'{step}.npy'
    edges_path = datadir / 'edges' / f'{step}.npy'
    x = np.load(feature_path)
    label_index = np.load(edges_path)
    edge_index = np.concatenate([label_index, label_index[[1, 0], :]], 1)
    return x, edge_index


def to_tensor(input_array):
    return torch.tensor(input_array).to(DEVICE)


def create_csv(dfs, prediction):
    df = pd.concat(dfs)
    df['Prediction'] = prediction
    df = df[['PotEng', 'Prediction']]
    df.to_csv('prediction.csv', index=False)


def train(dataset_dict, logger, CFG):
    EPOCHS = CFG['epochs1']
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']

    # shuffle 使うためだけに入れた
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    in_channels = train_dataset[0][0].size()[1]
    gnn_layer_hidden = CFG['gnn_layer_hidden_dim']
    last_layer_hidden = CFG['last_layer_hidden_dim']

    model = GCNModel(in_channels, gnn_layer_hidden, last_layer_hidden).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr1'])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_loss = 10000000

    if CFG['wandb']:
        wandbrun = wandb.init(project=CFG['project_name'], name=CFG['exp_name'])

    for epoch in range(0, EPOCHS):
        # training
        sum_loss = 0
        # for x, edge, label in tqdm(dataset, dynamic_ncols=True):
        model.train()
        for x, edge, label in train_loader:
            x, edge, label = torch.squeeze(x, 0), torch.squeeze(edge, 0), torch.squeeze(label, 0)
            x, edge, label = x.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
            pred = model(x, edge)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()

        # logging the loss
        scheduler.step()
        train_loss = sum_loss / len(train_dataset)
        logger.info(f'EPOCH: {epoch} train_loss: {train_loss:.8g}')

        # validation
        sum_loss = 0
        model.eval()
        for x, edge, label in valid_dataset:
            with torch.no_grad():
                x, edge, label = x.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
                pred = model(x, edge)
                loss = criterion(pred, label)
                sum_loss += loss.item()

        valid_loss = sum_loss / len(valid_dataset)
        logger.info(f'EPOCH: {epoch} valid_loss: {valid_loss:.8g}')

        if CFG['wandb']:
            wandb.log({'train_loss': train_loss,
                       'valid_loss': valid_loss,
                       'lr': scheduler.get_last_lr()[0]})

        # determine the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_model = model
            torch.save(best_model.state_dict(), 'best.pth')

    logger.info(f'best_epoch: {best_epoch} best_loss: {best_loss:.8g}')
    torch.save(model.state_dict(), 'last.pth')

    if CFG['wandb']:
        wandbrun.finish()
    return best_model


def infer(model, dataset):
    predictions = []
    with torch.no_grad():
        for x, edge, label in dataset:
            prediction = model(x, edge).cpu().numpy()[0]
            predictions.append(prediction)
    return predictions


def display(logger, CFG):
    logger.info("=" * 60)
    logger.info('GNN model')
    logger.info('model config')
    logger.info(f"- lr : {CFG['lr1']}")
    logger.info(f"- epochs : {CFG['epochs1']}")
    logger.info(f"- gnn_layer_hidden : {CFG['gnn_layer_hidden_dim']}")
    logger.info(f"- last_layer_hidden : {CFG['last_layer_hidden_dim']}")
    logger.info(f"- training_data : {CFG['train_dirs']}")
    logger.info(f"- validation_data : {CFG['valid_dirs']}")
    logger.info("=" * 60)


def main():
    logger = mylogger.init_logger(log_file='train.log')
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    seed_everything(CFG['seed'])
    display(logger, CFG)

    dataset_dict = dict()
    for type in ['train', 'valid']:
        dfs = []
        cnadicts = []
        cutoffdirs = []
        for dir in CFG[f'{type}_dirs']:
            DATADIR = Path('../../dataset') / str(dir)
            CUTOFFDIR = DATADIR / str(CFG['cutoff'])
            cna_path = DATADIR / 'cna'
            csv_path = DATADIR / 'thermo.csv'
            df = pd.read_csv(csv_path)
            ground_energy = df.iloc[0, 3]  # Potential energy of 0sec
            df = df.iloc[1:, :]
            df['PotEng'] = df['PotEng'] - ground_energy
            cnadict = load_cna(cna_path)
            dfs.append(df)
            cnadicts.append(cnadict)
            cutoffdirs.append(CUTOFFDIR)
        dataset = MDGraphDataset(dfs, cnadicts, cutoffdirs)
        dataset_dict[type] = dataset

    train(dataset_dict, logger, CFG)


if __name__ == '__main__':
    main()
