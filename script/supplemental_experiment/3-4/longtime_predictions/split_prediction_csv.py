import os
import yaml
import pandas as pd

with open('../config.yml', 'r') as yml:
    CFG = yaml.load(yml, Loader=yaml.SafeLoader)
    thermo_data_per_condition = CFG['thermo_thermo_data_per_condition']
    all_temps = CFG['all_temps']

csv = pd.read_csv('prediction.csv')['PotEng']

assert len(csv) == thermo_data_per_condition * len(all_temps)

# For each temp, extract thermo_data_per_condition data and split them

for i, temp in enumerate(all_temps):
    os.makedirs(str(temp), exist_ok=True)
    splited = csv.iloc[i * thermo_data_per_condition: (i + 1) * thermo_data_per_condition]
    splited.to_csv(f'{temp}/thermo.csv', index=False)
