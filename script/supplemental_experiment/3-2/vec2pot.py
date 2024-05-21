# 潜在ベクトルは 先の10 タイムステップまで予測しているので、
# うまく同じタイムステップのものを抽出して、平均をとる操作を行う
#     num           prediction_length
#  prediction    10, 11, 12, 13, 14, 15
#                11, 12, 13, 14, 15, 16
#                12, 13, 14, 15, 16, 17
#                13, 14, 15, 16, 17, 18
#                14, 15, 16, 17, 18, 19
#                15, 16, 17, 18, 19, 20
#                16, 17, 18, 19, 20, 21
#                17, 18, 19, 20, 21, 22
#                18, 19, 20, 21, 22, 23
#                19, 20, 21, 22, 23, 24


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


def load_vectors(datalen_per_condition, all_dirs):
    all_vectors = {}

    for condition in all_dirs:
        vectors = []
        for i in range(datalen_per_condition):
            vector = np.load(f'predicted_vectors/{condition}/pvec{i}.npy')
            vectors.append(vector)
        all_vectors[condition] = np.array(vectors)
    return all_vectors


def get_potentials(model, all_vectors,
                   seq_len,
                   pred_len,
                   thermo_data_per_condition,
                   predlen_per_condition):
    potentials = []

    for condition in all_vectors.keys():
        # seq_len 分の 0 を追加
        potentials.extend([0] * seq_len)

        vectors = all_vectors[condition]
        with torch.no_grad():
            for i in range(predlen_per_condition):
                current_latents = []
                for j in range(pred_len):
                    pos_for_timestep = i - j
                    if 0 <= pos_for_timestep and pos_for_timestep < thermo_data_per_condition:
                        current_latent = vectors[pos_for_timestep][j]
                        current_latents.append(current_latent)

                avg_current_latent = np.mean(current_latents, axis=0)

                potential = model(to_tensor(avg_current_latent))
                potentials.append(potential.item())

    return potentials


def to_tensor(input_array):
    return torch.tensor(input_array).to(DEVICE)


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    seed_everything(CFG['seed'])
    datalen_per_condition = CFG['thermo_data_per_condition'] - CFG['sequence_length'] - CFG['prediction_length'] + 1
    predlen_per_condition = CFG['thermo_data_per_condition'] - CFG['sequence_length']

    vectors = load_vectors(datalen_per_condition, CFG['all_dirs'])
    gnn_layer_hidden = CFG['gnn_layer_hidden_dim']
    last_layer_hidden = CFG['last_layer_hidden_dim']
    model = Vector2PotModel(gnn_layer_hidden, last_layer_hidden).to(DEVICE)
    weight = torch.load('best_gnn.pth')
    model.load_state_dict(weight)
    potentials = get_potentials(model, vectors, CFG['sequence_length'], CFG['prediction_length'],
                                datalen_per_condition, predlen_per_condition)

    others = pd.read_csv('prediction.csv')
    # potentials を追加
    others['TimeSeries'] = potentials
    others.to_csv('prediction_with_time.csv', index=False)


if __name__ == '__main__':
    main()
