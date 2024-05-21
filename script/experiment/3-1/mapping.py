import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


def main():
    print('[3-1 mapping.py]')
    os.makedirs('images', exist_ok=True)

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        interval = CFG['interval']
        data_per_con = CFG['thermo_data_per_condition']
        all_dirs = CFG['all_dirs']
        exp_name = CFG['exp_name']
        BASE_ENERGY = CFG['BASE_ENERGY']
        csv_path = 'prediction_with_time.csv'

    df = pd.read_csv(csv_path)
    A, B, C = df['PotEng'], df['Prediction'], df['TimeSeries']
    for i, name in enumerate(all_dirs):
        truth = A.iloc[data_per_con * i: data_per_con * (i + 1)]
        pred = B.iloc[data_per_con * i: data_per_con * (i + 1)]
        pred_lstm = C.iloc[data_per_con * i: data_per_con * (i + 1)]

        truth += BASE_ENERGY[i]
        pred += BASE_ENERGY[i]
        pred_lstm += BASE_ENERGY[i]
        x = np.linspace(1000, interval * data_per_con, data_per_con)

        masked_to = 20
        masked_pred_lstm = np.ma.array(pred_lstm, mask=False)
        masked_pred_lstm.mask[:masked_to] = True

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig = plt.figure(figsize=[9, 6])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(x, truth, label='ground truth', linewidth=1.4)
        ax1.plot(x, masked_pred_lstm, label='prediction with LSTM', linewidth=1.4)

        plt.xticks([0, 30000, 60000, 90000], fontsize=12)
        plt.savefig(f'images/{exp_name}_{i}.png')


if __name__ == '__main__':
    main()
