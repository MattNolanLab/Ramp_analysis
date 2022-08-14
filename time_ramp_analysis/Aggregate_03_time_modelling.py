#%%
# Perform time analysis on final aggregated dataframe from Harry 20220319

import pickle
import pandas as pd
from aggregate_analysis import time_modelling
from tqdm import tqdm
import numpy as np 
import matplotlib.pylab as plt
import seaborn as sns


#%%
cohorts = [2,3,4,5,7]
output_path = '/mnt/datastore/Teris/CurrentBiology_2022'

#%%
# cohort = 3
for cohort in cohorts:
    print(f'Processing cohort {cohort}')

    with open(f'{output_path}/cohort{cohort}_binned.pkl', 'rb') as f:
        data_list = pickle.load(f)

    #%% Do timing modeling
    df_timemodel_list = []
    df4r_list = []
    for d in tqdm(data_list):
        (df_model, df4r) = time_modelling(d)
        if df_model is not None:
            df_timemodel_list.append(df_model)
            df4r_list.append(df4r)

    #%% Save output
    df4r = pd.concat(df4r_list)
    df4r.to_pickle(f'{output_path}/cohort{cohort}_df4r.pkl')

    with open(f'{output_path}/cohort{cohort}_timemodel.pkl', 'wb') as f:
        pickle.dump(df_timemodel_list, f)

# # %%
# df = pd.read_pickle(f'{output_path}/cohort2_df4r.pkl')

# #%%

# df2 = df[(df.session_id=='245_D18_2018-11-10_11-07-50') & (df.cluster_id == 9)]
# plt.plot(df2.speed)

# # %%
# with open(f'{output_path}/cohort2_binned.pkl','rb') as f:
#     data_list = pickle.load(f)

# %%
