# Perform ramp score analysis on final aggregated dataframe from Harry 20220319

# Directly use the binned data from Sarah
#%%
import pickle
import re
import shutil
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import seaborn as sns
import tqdm
import xarray as xr
from tqdm import tqdm

import glmneuron
import setting
from aggregate_analysis import bin_data

# %% Load data 
cohorts = [2,3,4,5,7]
output_path = '/mnt/datastore/Teris/CurrentBiology_2022/'

for cohort in cohorts:
    print(f'Processing cohort {cohort}')
    df_path = f'/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort{cohort}_with_OF_unsmoothened.pkl'
    df_spatial = pd.read_pickle(df_path)

    
    data= []
    for i in tqdm(range(len(df_spatial))):
        d = bin_data(df_spatial.iloc[i], cohort)
        data.append(d)

    # save
    with open(f'{output_path}/cohort{cohort}_binned.pkl', 'wb') as f:
        pickle.dump(data,f)
