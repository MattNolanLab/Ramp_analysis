import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from scipy import signal
#from utils import getWrappedSubplots

from tqdm.auto import tqdm
import numpy as np
from scipy import stats
import scipy

def run_lm_basic(df):
    df["lm_result_outbound"] = ""
    df["lm_result_homebound"] = ""

    # run lm model and store slope and rsqusred
    for cluster in range(len(df)):
        rates=np.array(df.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        rates_out = rates[30:90]
        position = np.arange(60)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(position, rates_out)
        # determine slope direction from lm results
        lm_result = []
        if slope > 0:
            lm_result = 'Positive'
        elif slope < 0:
            lm_result = 'Negative'
        df.at[cluster, 'lm_result_outbound'] = lm_result # add data to dataframe

        rates_home = rates[110:170]
        position = np.arange(60)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(position, rates_home)
        if slope > 0:
            lm_result = 'Positive'
        elif slope < 0:
            lm_result = 'Negative'
        df.at[cluster, 'lm_result_homebound'] = lm_result # add data to dataframe
    return df


def peakAnalysis3(df, curveRange, col_suffix='', lm_check_col=None, range_thres=0, slope_thres=0, fr_col=None ):
    # perform peak analysis on a dataframe
    # it assume the df has lm_result
    # it takes Sarah's average rate directly
    maxima_locs = []
    max_locs_cm = []
    min_locs_cm = []
    maxima_types=[]
    maxima_locs_cm =[]
    smooth_curve = []
    df = df.copy()

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        #loc = np.array(df.iloc[i].pos)

        curve = np.array(row[fr_col])
        curve = curve[curveRange]
        # Impute NaN values with mean
        curve[np.isnan(curve)] = np.nanmean(curve)

        # print(i)
        if not np.any(np.isnan(curve)):
            curve = signal.savgol_filter(curve, 11,0) #smooth the curve first

        smooth_curve.append(curve)
        # print(curve.shape)

        maxima_type = np.nan

        #Perform additonal check of the signal range to make sure the maxima is meaningful
        if (curve.max()-curve.min()) < range_thres:
            maxima_loc = np.nan
            maxima_type = np.nan

        c_max = np.max(curve)
        c_min = np.min(curve)
        c_mean = np.mean(curve)
        max_idx = np.argmax(curve)
        min_idx = np.argmin(curve)


        if lm_check_col is not None:
            lm_result = row[lm_check_col]

            if lm_result=='Positive':
                # Choose the max point
                maxima_loc = max_idx
                maxima_type = 'max'

            elif lm_result =='Negative':
                maxima_loc = min_idx
                maxima_type = 'min'
            else:
                maxima_loc = np.nan
        else:
            # Determine the direction of the curve by the difference between the peak and the mean

            if(c_max-c_mean)>(c_mean-c_min):
                maxima_loc = max_idx
                maxima_type = 'max'
            else:
                maxima_loc = min_idx
                maxima_type = 'min'

        maxima_locs.append(maxima_loc)
        maxima_types.append(maxima_type)

        if not np.isnan(maxima_loc):
            # print(curveRange)
            # print(maxima_loc)
            maxima_locs_cm.append(curveRange[maxima_loc])
            max_locs_cm.append(curveRange[max_idx])
            min_locs_cm.append(curveRange[min_idx])
        else:
            maxima_locs_cm.append(np.nan)
            max_locs_cm.append(np.nan)
            min_locs_cm.append(np.nan)


    df['maxima'+col_suffix] = maxima_locs
    df['maxima_cm'+col_suffix] = maxima_locs_cm
    df['smooth_curve'+col_suffix] = smooth_curve
    df['maxima_type'+col_suffix] = maxima_types
    df['min_cm'+col_suffix] = min_locs_cm
    df['max_cm'+col_suffix] = max_locs_cm

    return df




def run_peak_analysis(df):

    out_range = np.arange(30,100) # from 30 to  100 cm, slightly beyond the reward zone x*2-30
    home_range = np.arange(100,170)# 110 to 200, 0 to 30
    #all_range = np.arange(200)

    # run lm analysis
    df = run_lm_basic(df)
    # run peak detection

    df_out = peakAnalysis3(df,out_range, col_suffix='_normal', lm_check_col='lm_result_outbound', fr_col = 'Avg_FR_beaconed_rewarded')
    df_home = peakAnalysis3(df,home_range, col_suffix='_normal', lm_check_col='lm_result_homebound', fr_col = 'Avg_FR_beaconed_rewarded')

    df_out = peakAnalysis3(df_out,out_range, col_suffix='_reward', lm_check_col='lm_result_outbound', fr_col = 'FR_reset_at_reward')
    df_home = peakAnalysis3(df_home,home_range, col_suffix='_reward', lm_check_col='lm_result_homebound', fr_col = 'FR_reset_at_reward')

    df_out = peakAnalysis3(df_out,out_range,col_suffix='_fs',  lm_check_col='lm_result_outbound', fr_col = 'FR_reset_at_FS')
    df_home = peakAnalysis3(df_home,home_range, col_suffix='_fs', lm_check_col='lm_result_homebound', fr_col = 'FR_reset_at_FS')

    # combine dataframe together
    df_out['peak_region'] = 'outbound'
    df_home['peak_region'] = 'homebound'

    dfs_comb = pd.concat([df_out, df_home])

    return dfs_comb


