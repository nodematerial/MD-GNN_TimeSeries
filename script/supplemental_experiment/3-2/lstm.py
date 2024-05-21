# sequence length から次の 10 step 計算する

import os
import sys
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

sys.path.append('../../')

import mylogger

DEVICE = 'cuda'
os.environ['WANDB_SILENT'] = 'true'


# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, CFG):
        super(LSTMModel, self).__init__()
        self.input_dim = CFG['lstm_params']['input_dim'] + 1
        self.hidden_dim = CFG['lstm_params']['hidden_dim']
        self.output_dim = CFG['lstm_params']['input_dim']
        self.num_layers = CFG['lstm_params']['num_layers']
        self.predition_length = CFG['prediction_length']
        self.dropout = CFG['lstm_params']['dropout']

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=self.dropout,
                            batch_first=CFG['lstm_params']['batch_first'])
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim * self.predition_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc1(out[:, -1, :])
        pred = pred.view(-1, self.predition_length, self.output_dim)
        # Residual connection
        pred = pred + x[:, -1, :-1].unsqueeze(1)

        return pred


# カスタムデータセットクラスの定義
class TimeSeriesDataset(torchdata.Dataset):
    def __init__(self, CFG, type='train'):
        self.data = {}
        self.CFG = CFG
        self.type = type
        self.datalen_per_condition = calc_datalen_per_condition(CFG)
        self.dirs = CFG[f'{self.type}_dirs']
        self.temps = CFG[f'{self.type}_temps']

        for dir in self.dirs:
            array = []
            for i in range(CFG['thermo_data_per_condition']):
                latent_vector = np.load(f'../../../dataset/{dir}/vectors_additional_exp/vec{i}.npy')
                array.append(latent_vector)
            self.data[dir] = np.array(array)

    def __len__(self):
        return self.datalen_per_condition * len(self.dirs)

    def __getitem__(self, idx):
        condition_id = idx // self.datalen_per_condition
        condition = self.dirs[condition_id]
        temp = float(self.temps[condition_id])

        start = idx % self.datalen_per_condition
        end = start + self.CFG['sequence_length']
        condition_data = self.data[condition]

        x = condition_data[start:end]
        temps = np.full((x.shape[0], 1), temp)
        x = np.concatenate([x, temps], axis=1)
        label = condition_data[end: end + self.CFG['prediction_length']]

        return x, label


def to_tensor(input_array):
    return torch.tensor(input_array).to(DEVICE).float()


def train(logger, CFG):
    EPOCHS = CFG['epochs']

    model = LSTMModel(CFG).to(DEVICE)
    train_dataset = TimeSeriesDataset(CFG, 'train')
    valid_dataset = TimeSeriesDataset(CFG, 'valid')
    train_loader = torchdata.DataLoader(train_dataset, **CFG['loader_params']['train'])
    valid_loader = torchdata.DataLoader(valid_dataset, **CFG['loader_params']['valid'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_loss = 10000000

    if CFG['wandb']:
        wandbrun = wandb.init(project=CFG['project_name'], name=CFG['exp_name'])

    for epoch in range(0, EPOCHS):
        sum_loss = 0
        for x, label in train_loader:
            x, label = to_tensor(x), to_tensor(label)
            model.train()
            pred = model(x)
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
        for x, label in valid_loader:
            with torch.no_grad():
                x, label = to_tensor(x), to_tensor(label)
                pred = model(x)
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

    logger.info(f'best_epoch: {best_epoch} best_loss: {best_loss:.8g}')
    torch.save(model.state_dict(), 'best_lstm.pth')

    if CFG['wandb']:
        wandbrun.finish()
    return best_model


def infer(model, CFG):
    dataset = TimeSeriesDataset(CFG, 'all')
    predictions = []
    with torch.no_grad():
        for x, label in dataset:
            x = to_tensor(x).view(1, x.shape[0], x.shape[1])
            prediction = model(x).cpu().numpy()[0]
            predictions.append(prediction)
    return predictions


def calc_datalen_per_condition(CFG):
    return CFG['thermo_data_per_condition'] - CFG['sequence_length'] - CFG['prediction_length'] + 1


def save_for_each_condition(predictions, CFG):
    for dirs in CFG['all_dirs']:
        os.makedirs(f'./predicted_vectors/{dirs}', exist_ok=True)
        datalen_per = calc_datalen_per_condition(CFG)
        for i in range(len(predictions)):
            current_condition_id = i // datalen_per
            current_condition = CFG['all_dirs'][current_condition_id]
            os.makedirs(f'./predicted_vectors/{current_condition}', exist_ok=True)

            timestep = i % datalen_per
            np.save(f'./predicted_vectors/{current_condition}/pvec{timestep}.npy', predictions[i])


def display(logger, CFG):
    logger.info("=" * 60)
    logger.info('model config')
    logger.info('LSTM model')
    logger.info(f"- lr : {CFG['lr']}")
    logger.info(f"- epochs : {CFG['epochs']}")
    logger.info(f"- input_dim : {CFG['lstm_params']['input_dim']}")
    logger.info(f"- hidden_dim : {CFG['lstm_params']['hidden_dim']}")
    logger.info(f"- num_layers : {CFG['lstm_params']['num_layers']}")
    logger.info(f"- training_data : {CFG['train_dirs']}")
    logger.info(f"- validation_data : {CFG['valid_dirs']}")
    logger.info("=" * 60)


def main():
    print('[3-2 lstm.py]')
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
    logger = mylogger.init_logger(log_file='train.log')
    display(logger, CFG)

    best_model = train(logger, CFG)
    predictions = infer(best_model, CFG)
    save_for_each_condition(predictions, CFG)


if __name__ == '__main__':
    main()
