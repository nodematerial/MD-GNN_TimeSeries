import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


def plot_prediction(temp, base_energy, interval, future_predictions_num):
    df1 = pd.read_csv(f'./longtime_predictions/{temp}/thermo.csv')
    df2 = pd.read_csv(f'./longtime_predictions/{temp}/time.csv', names=['Time'])

    gt = df1['PotEng'].values
    time = df2['Time'].values + base_energy
    x = np.linspace(interval, interval * future_predictions_num, future_predictions_num)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # plot twin axix data
    fig = plt.figure(figsize=[18, 6])
    ax1 = fig.add_subplot(1, 1, 1)
    #ax1.plot(x, gt, label='ground truth', linewidth=1.2)
    ax1.plot(x, time, label='prediction with LSTM', linewidth=1.8)

    plt.xticks([0, 300000, 600000, 900000], fontsize=12)
    plt.savefig(f'images/{temp}.png')


def main():
    print('[3-4 longtime_mapping.py]')
    os.makedirs('images/Ni', exist_ok=True)

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        temps = CFG['all_temps']
        interval = CFG['interval']
        future_predictions_num = CFG['future_predictions_num']
        BASE_ENERGY = CFG['BASE_ENERGY']

    for temp, base_energy in zip(temps, BASE_ENERGY):
        plot_prediction(temp, base_energy, interval, future_predictions_num)


if __name__ == '__main__':
    main()

