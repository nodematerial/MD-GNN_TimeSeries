import os
import yaml
import random

import numpy as np
import pandas as pd

from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch_geometric.nn import SAGEConv

os.environ['WANDB_SILENT'] = 'true'

device = 'cuda'


class ExtractMiddleModel(torch.nn.Module):
    def __init__(self, in_channels, gnn_layer_hidden, last_layer_hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, gnn_layer_hidden)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(gnn_layer_hidden, 1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(last_layer_hidden)
        self.fc2 = nn.Linear(last_layer_hidden, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.squeeze(x, 1)
        vector = self.fc1(x)
        return vector


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def get_vector(dataset, model):
    vectors = []
    with torch.no_grad():
        for x, edge, _ in dataset:
            vectors.append(model(x, edge).cpu().numpy())
    return vectors


def save_vector(vectors, path):
    os.makedirs(path, exist_ok=True)
    for i, vector in enumerate(vectors):
        np.save(path / f'vec{i}.npy', vector)


def load_data(step, datadir):
    feature_path = datadir / 'x' / f'{step}.npy'
    edges_path = datadir / 'edges' / f'{step}.npy'
    x = np.load(feature_path)
    label_index = np.load(edges_path)
    edge_index = np.concatenate([label_index, label_index[[1, 0], :]], 1)
    return x, edge_index


def to_tensor(input_array):
    return torch.tensor(input_array).to(device)


def create_csv(dfs, prediction):
    df = pd.concat(dfs)
    df['Prediction'] = prediction
    df = df[['PotEng', 'Prediction']]
    df.to_csv('prediction.csv', index=False)


def load_dataset(CFG):
    dfs = []
    cnadicts = []
    cutoffdirs = []
    for dir in CFG['all_dirs']:
        DATADIR = Path('../../../dataset') / str(dir)
        CUTOFFDIR = DATADIR / str(CFG['cutoff'])
        cna_path = DATADIR / 'cna'
        csv_path = DATADIR / 'thermo.csv'
        # 0sec の値が異常値なので取り除く
        df = pd.read_csv(csv_path)
        ground_energy = df.iloc[0, 3]  # 最初のステップの位置エネルギー
        df = df.iloc[1:, :]
        df['PotEng'] = df['PotEng'] - ground_energy
        cnadict = load_cna(cna_path)
        dfs.append(df)
        cnadicts.append(cnadict)
        cutoffdirs.append(CUTOFFDIR)
    dataset = MDGraphDataset(dfs, cnadicts, cutoffdirs)
    return dfs, dataset

def load_one_condition(dir, CFG):
    DATADIR = Path('../../../dataset') / str(dir)
    CUTOFFDIR = DATADIR / str(CFG['cutoff'])
    cna_path = DATADIR / 'cna'
    csv_path = DATADIR / 'thermo.csv'
    # 0sec の値が異常値なので取り除く
    df = pd.read_csv(csv_path)
    ground_energy = df.iloc[0, 3]  # 最初のステップの位置エネルギー
    df = df.iloc[1:, :]
    df['PotEng'] = df['PotEng'] - ground_energy
    cnadict = load_cna(cna_path)
    dataset = MDGraphDataset([df], [cnadict], [CUTOFFDIR])
    return dataset

def calc_one_condition(dir, CFG):
    dataset = load_one_condition(dir, CFG)
    in_channels = dataset[0][0].size()[1]
    gnn_layer_hidden = CFG['gnn_layer_hidden_dim']
    last_layer_hidden = CFG['last_layer_hidden_dim']
    model = ExtractMiddleModel(in_channels, gnn_layer_hidden, last_layer_hidden).to(device)
    weight = torch.load('best_gnn.pth')
    model.load_state_dict(weight)
    return get_vector(dataset, model)

def main():
    print('[3-1 build_vector.py]')
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    seed_everything(CFG['seed'])
    for dir in CFG['all_dirs']:
        save_path = Path(f'../../../dataset/{dir}/vectors')
        vectors = calc_one_condition(dir, CFG)
        save_vector(vectors, save_path)

if __name__ == '__main__':
    main()
