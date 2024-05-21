import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


def plot_prediction(temp, base_energy):
    print(temp)
    df1 = pd.read_csv(f'./predictions/{temp}/thermo.csv')
    df2 = pd.read_csv(f'./predictions/{temp}/time.csv', names=['Time'])

    gt = df1['PotEng'].values + base_energy
    time = df2['Time'].values[:len(gt)] + base_energy

    x = np.linspace(1000, 90000, 300)

    masked_to = 20
    time = np.ma.array(time, mask=False)
    time.mask[:masked_to] = True

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # plot twin axix data
    fig = plt.figure(figsize=[9, 6])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, gt, label='ground truth', linewidth=1.5)
    ax1.plot(x, time, label='prediction with LSTM', linewidth=1.5)

    plt.xticks([0, 30000, 60000, 90000], fontsize=12)
    plt.savefig(f'images/{temp}.png')


def main():
    print('[3-3 mapping.py]')
    os.makedirs('images', exist_ok=True)

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        temps = CFG['all_temps']
        BASE_ENERGY = CFG['BASE_ENERGY']
    for temp, base_energy in zip(temps, BASE_ENERGY):
        plot_prediction(temp, base_energy)


if __name__ == '__main__':
    main()
