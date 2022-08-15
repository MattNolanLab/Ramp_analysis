#%%
# Perform ramp score analysis on final aggregated dataframe from Harry 20220319

import pickle
import pandas as pd
from aggregate_analysis import compute_ramp_score
from tqdm import tqdm
import numpy as np 
import matplotlib.pylab as plt
import seaborn as sns

#%% Load previous data
cohorts = [2,3,4,5,7]
output_path = '/mnt/datastore/Teris/CurrentBiology_2022'

for cohort in cohorts:

    #load data
    with open(f'{output_path}/cohort{cohort}_binned.pkl', 'rb') as f:
            data_list = pickle.load(f)

    # Do ramp anaylsis on each cell
    df_ramp = []
    for data in tqdm(data_list):
        df_ramp.append(compute_ramp_score(data))
    # save
    df_ramp = pd.concat(df_ramp)
    df_ramp.to_pickle(f'{output_path}/cohort{cohort}_rampscore.pkl')

#%% Combine rampscore together
output_path = '/mnt/datastore/Teris/CurrentBiology_2022'
cohorts = [2,3,4,5,7]
df_list = []
for cohort in cohorts:

    df = pd.read_pickle(f'{output_path}/cohort{cohort}_rampscore.pkl')
    df_list.append(df)


df_ramp = pd.concat(df_list)
df_ramp.to_pickle(f'{output_path}/all_rampscore.pkl')

