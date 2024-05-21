
import os
import yaml
import time

import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict

DEVICE = 'cuda'
os.environ['WANDB_SILENT'] = 'true'


# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prediction_length):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim + 1
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.num_layers = num_layers
        self.predition_length = prediction_length

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim * self.predition_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc1(out[:, -1, :])
        pred = pred.view(-1, self.predition_length, self.output_dim)
        # Residual connection
        pred = pred + x[:, -1, :-1].unsqueeze(1)

        return pred


class FullyConnectedModel(nn.Module):
    def __init__(self, last_layer_hidden):
        super().__init__()
        self.fc2 = nn.Linear(last_layer_hidden, 1)

    def forward(self, latent_vector):
        prediction = self.fc2(latent_vector)
        return torch.squeeze(prediction)


def infer(gnn_model, lstm_model, data, future_predictions_num, temp, prediction_length=1):
    predictions = []
    previous_latent_vectors_dict = defaultdict(list)

    # FIXME: 最初の数ステップに対しても、遡ってprediction_length個の潜在ベクトルの平均を取れるようにする
    # FIXME: Also, for the first few steps, take the average of prediction_length latent vectors back

    lstm_model.eval()
    for timestep in range(future_predictions_num):
        # 過去の複数の潜在ベクトルから現在のtimestepの潜在ベクトルを推定する
        # これを複数回行い、推定した潜在ベクトルの平均を取り、次の潜在ベクトル(確定)として用いる
        # 
        # Estimate the latent vector of the current timestep from multiple past latent vectors
        # Do this multiple times, take the average of the estimated latent vectors, and use it as the next
        # latent vector (determined)
        #
        # [expample]
        # p: previous, n: new
        # t1 t2 t3 t4 t5 t6 t7 t8   t9 t10
        #
        # p1 p2 p3 p4 p5 n1 n2 [n3]
        #    p1 p2 p3 p4 p5 n1 [n2] n3
        #       p1 p2 p3 p4 p5 [n1] n2 n3
        #
        # z_t8 = [n1] + [n2] + [n3] / 3

        prev = previous_tensor(data, prediction_length, temp)
        new_latent_vectors = lstm_model(prev)
        new_latent_vectors = torch.squeeze(new_latent_vectors).cpu().detach().numpy()

        for i in range(prediction_length):
            position = timestep + i
            previous_latent_vectors_dict[position].append(new_latent_vectors[i, :])

        avg_nexttime_latent_vector = np.mean(previous_latent_vectors_dict[timestep], axis=0)
        data.append(avg_nexttime_latent_vector)

        avg_nexttime_latent_vector = to_tensor(avg_nexttime_latent_vector)
        prediction = gnn_model(avg_nexttime_latent_vector).cpu().detach().numpy()
        predictions.append(prediction)
    return predictions, data


# 引数の値に応じた、過去の潜在ベクトルのリストを返す
# 0 の場合は現在最新のものから prediction_length 個の潜在ベクトルを返す
# 1 の場合は一つ前のものから prediction_length 個の潜在ベクトルを返す...
# 
# Return a list of past latent vectors according to the argument value
# If 0, return prediction_length latent vectors from the latest one
# If 1, return prediction_length latent vectors from the previous one...
def previous_tensor(data, prediction_length, temp):
    dat = np.array(data[-prediction_length:])

    temps = np.full((dat.shape[0], 1), temp)
    dat = np.concatenate([dat, temps], axis=1)
    previous_tensor = to_tensor(dat)

    previous_tensor = previous_tensor.view(1, previous_tensor.shape[0], previous_tensor.shape[1])
    return previous_tensor


def to_tensor(li):
    return torch.tensor(np.array(li)).to(DEVICE).float()


def get_data(sequence_length, dir):
    # latent_vectorをロード
    latent_vectors = []
    for i in range(sequence_length):
        latent_vector = np.load(f'../../../dataset/{dir}/vectors/vec{i}.npy')
        latent_vectors.append(latent_vector)

    return latent_vectors


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        latent_dim = CFG['lstm_params']['input_dim']
        lstm_hidden_dim = CFG['lstm_params']['hidden_dim']
        lstm_num_layers = CFG['lstm_params']['num_layers']
        sequence_length = CFG['sequence_length']
        future_predictions_num = CFG['future_predictions_num']
        prediction_length = CFG['prediction_length']
        weight_path = CFG['weight_file']
        all_dirs = CFG['all_dirs']
        all_temps = CFG['all_temps']

    gnn_weights = torch.load('./fully_connected.pth')
    gnn_model = FullyConnectedModel(latent_dim).to(DEVICE)
    gnn_model.load_state_dict(gnn_weights)

    lstm_weights = torch.load(weight_path)
    lstm_model = LSTMModel(latent_dim, lstm_hidden_dim, lstm_num_layers, prediction_length).to(DEVICE)
    lstm_model.load_state_dict(lstm_weights)

    for _, (dir, temp) in enumerate(zip(all_dirs, all_temps)):
        print(dir)
        start_at = time.time()
        init_data = get_data(sequence_length, dir)

        predictions, data = infer(gnn_model, lstm_model, init_data, future_predictions_num, temp, prediction_length)
        elapsed_time = time.time() - start_at
        print(f'elapsed time: {elapsed_time:.2f} [sec]')

        # list を csvにする
        predictions = np.array(predictions)
        print(len(predictions))

        os.makedirs(f'./longtime_predictions/{temp}', exist_ok=True)
        np.savetxt(f'./longtime_predictions/{temp}/time.csv', predictions, delimiter=',')

        # data(latent_vector)を保存
        os.makedirs(f'./predicted_vec/{temp}', exist_ok=True)
        for i, latent_vector in enumerate(data):
            np.save(f'./predicted_vec/{temp}/vec{i}.npy', latent_vector)


if __name__ == '__main__':
    main()
