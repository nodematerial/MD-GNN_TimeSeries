import os
import yaml
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

os.environ['WANDB_SILENT'] = 'true'

DEVICE = 'cuda'


class Vector2PotModel(torch.nn.Module):
    def __init__(self, gnn_layer_hidden, last_layer_hidden):
        super().__init__()
        self.conv1 = SAGEConv(9, gnn_layer_hidden)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(gnn_layer_hidden, 1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(last_layer_hidden)
        self.fc2 = nn.Linear(last_layer_hidden, 1)

    def forward(self, vector):
        potential = self.fc2(vector)
        return potential


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_vectors(datalen):
    vectors = []

    for i in range(datalen):
        # vector = np.load(f'./vectors/vec{i}.npy')
        vector = np.load(f'./predicted_vectors/pvec{i}.npy')
        vectors.append(vector)
    return vectors


def get_potentials(model, vectors, addlen, thermo_data_per_condition):
    potentials = []
    with torch.no_grad():
        for i, vector in enumerate(vectors):
            if i % thermo_data_per_condition == 0:
                potentials.extend([0] * addlen)
            potential = model(to_tensor(vector))
            potentials.append(potential.item())
    return potentials


def to_tensor(input_array):
    return torch.tensor(input_array).to(DEVICE)


def main():
    print('[3-1 vec2pot.py]')
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    seed_everything(CFG['seed'])
    datalen = (CFG['thermo_data_per_condition'] - CFG['sequence_length']) * len(CFG['all_dirs'])
    per_condition = CFG['thermo_data_per_condition'] - CFG['sequence_length']

    vectors = load_vectors(datalen)
    gnn_layer_hidden = CFG['gnn_layer_hidden_dim']
    last_layer_hidden = CFG['last_layer_hidden_dim']
    model = Vector2PotModel(gnn_layer_hidden, last_layer_hidden).to(DEVICE)
    weight = torch.load('best_gnn.pth')
    model.load_state_dict(weight)
    potentials = get_potentials(model, vectors, CFG['sequence_length'], per_condition)
    print(len(potentials))

    others = pd.read_csv('prediction.csv')
    # potentials を追加
    others['TimeSeries'] = potentials
    others.to_csv('prediction_with_time.csv', index=False)


if __name__ == '__main__':
    main()
