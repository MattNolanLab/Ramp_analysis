import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import seaborn as sns
import xarray as xr
from palettable.colorbrewer.qualitative import Paired_8 as colors
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm import tqdm

import glmneuron
import setting
from notebooks.analysis.peak_detect import (findBestRampScore2, findBreakPt,
                                            getRampScore4, makeLinearGrid)
from utils import *


def bin_data(spatial_firing, cohort):

    speed = np.array(spatial_firing.spike_rate_in_time[1]).real
    position=np.array(spatial_firing.spike_rate_in_time[2]).real
    trial_num = np.array(spatial_firing.spike_rate_in_time[3]).real
    trial_type = np.array(spatial_firing.spike_rate_in_time[4]).real
    
    spiketrain = np.array(spatial_firing.spike_rate_in_time[0]).real*4
    spiketrain = spiketrain[None,:]

    if len(spiketrain)>0:
        spiketrain=np.stack(spiketrain).T
    else:
        spiketrain = None

    # %% Add position
    time = np.arange(position.size) /(1000/setting.binSize) 
    position_xr = xr.DataArray(
        position,
        dims=("time"),
        coords={"time": pd.TimedeltaIndex(time, unit="s")},
    )

    #%% Add spikes
    if spiketrain is not None:
        neuron = [0]
        # add spike
        spiketrains_xr = xr.DataArray(
            spiketrain,
            dims=("time", "neuron"),
            coords={
                "time": pd.TimedeltaIndex(time, unit="s"),
                "neuron": neuron,
            },
        )
    else:
        spiketrains_xr = None

    #%% Trial type
    if spiketrain is not None:
        trial_type_xr = xr.DataArray(
            trial_type,
            dims=("time"),
            coords={"time": pd.TimedeltaIndex(time, unit="s")},
        )

    else:
        trial_type_xr = None

    #%% Add trial number
    if spiketrain is not None:

        trial_number_xr = xr.DataArray(
            trial_num,
            dims={"time"},
            coords={"time": pd.TimedeltaIndex(time, unit="s")},
        )
    else:
        trial_number_xr = None


    #%% Load reward info
    if spiketrain is not None:
        trial_number_rs = trial_number_xr.data
        rewarded_trials = spatial_firing.rewarded_trials

        # Match with the trial number
        isRewarded = np.empty_like(trial_number_rs)
        isRewarded[:] = False

        for rt in rewarded_trials:
            isRewarded[trial_number_rs == rt] = True

        isRewarded_xr = xr.DataArray(
            isRewarded, dims={"time"}, coords=trial_number_xr.coords
        )


    else:
        isRewarded_xr = None

    #%% Make on hot vector 
    (posgrid, posvec, pos) = glmneuron.make_1d_map(
        position_xr.data, setting.position_bin, 0, setting.trackLength
    )

    (speedgrid, speedvec, speed) = glmneuron.make_1d_map(
        speed, setting.speed_bin, 0, 50
    )

    accel = np.diff(speed,prepend=0)/(setting.binSize/1000)
    (accelgrid, accelvec, accel) = glmneuron.make_1d_map(
        accel, setting.accel_bin, -100,100
    ) 

    #%% make xarray
    posgrid_xr = xr.DataArray(
        posgrid,
        dims=("time", "pos_bins"),
        coords={"time": position_xr.time, "pos_bins": posvec},
    )

    speedgrid_xr = xr.DataArray(
        speedgrid,
        dims=("time", "speed_bins"),
        coords={"time": position_xr.time, "speed_bins": speedvec},
    )

    accelgrid_xr = xr.DataArray(
        accelgrid,
        dims=("time", "accel_bins"),
        coords={"time": position_xr.time, "accel_bins": accelvec},
    )


    speed_xr = xr.DataArray(speed, dims=("time"), coords={"time": position_xr.time})
    accel_xr = xr.DataArray(accel, dims=("time"), coords={"time": position_xr.time})

    #%% Record down the cluster id
    cluster_id_xr = xr.DataArray([spatial_firing.cluster_id],
        dims=('neuron'), coords={'neuron':neuron})

    #%% Combine into one dataset
    binned_data = xr.Dataset(
        {
            "spiketrain": spiketrains_xr,
            "position": position_xr,
            "speed": speed_xr,
            "accel": accel_xr,
            "pos_grid": posgrid_xr,
            "speed_grid": speedgrid_xr,
            "accel_grid": accelgrid_xr,
            "trial_number": trial_number_xr,
            "trial_type": trial_type_xr,
            "isRewarded": isRewarded_xr,
            'cluster_id': cluster_id_xr
        }
    )

    # remove nan
    binned_data = binned_data.dropna(dim="time")

    #%% add some attributes
    path_parts = spatial_firing.session_id

    # use Regex to extract metadata from folder name
    pattern = re.compile(r"(?P<animal>.*)_D(?P<session>\d+)_(?P<date>.+)_(?P<time>.+)")
    m = pattern.match(path_parts)

    binned_data.attrs["session_id"] = spatial_firing.session_id
    binned_data.attrs["cohort"] = cohort
    binned_data.attrs["animal"] = m["animal"]
    binned_data.attrs["session"] = int(m["session"])
    binned_data.attrs["date"] = m["date"]
    binned_data.attrs["time"] = m["time"]

    # also store the analysis setting
    varName = [x for x in dir(setting) if not x.startswith("__")]
    binned_data.attrs.update({k: v for k, v in setting.__dict__.items() if k in varName})

    #%% Save data
    return binned_data

def compute_ramp_score(data):
    # the input should be a xarray dataset
    if data.spiketrain.data.item(0) is None:
        # No spikes found
        data = None

    #%%
    '''
    - bin the firing rate according to location
    - get the average firing rate for each bin
    - smooth the curve a bit to fit the inflation points
    - calculate the ramp score for all combination of these inflation points
    - return the maximum inflation points pair and rampscore
    '''

    #%% Pre-processing data
    if data:
        pos_grid = data.pos_grid.data
        speed = data.speed.data
        spiketrain = data.spiketrain.data
        pos_binned = np.argmax(pos_grid,axis=1)*(setting.trackLength/setting.position_bin)
        pos_bins = np.arange(setting.trackLength, step=setting.trackLength/setting.position_bin).astype(int)
        isRewarded = data.isRewarded.data
        spiketrain = gaussian_filter1d(spiketrain,2,axis=0) #smooth
        spiketrain[spiketrain<0] = 0 #avoid negative firing rate
        trial_type = data.trial_type

        # %% Calculate the average firing rate at each location
        sel_idx = (speed>3)
        pos_binned_filt = pos_binned[sel_idx]
        spiketrain_filt = spiketrain[sel_idx,:]
        trial_type_filt = trial_type[sel_idx]

        #only included rewarded trial

        #%% Calculate rampscore
        pos_range = [[30,90],[110,170],[0,200]]
        ramp_type = ['outbound','homebound','all']
        trial_types_num = [0,1,2,None]
        trial_type_name=['beaconed','non-beaconed','probe','all']
        ramp_scores = []

        for p_range,rt in zip(pos_range, ramp_type):

            bps = []
            fr_smooth = []
            for tt,tt_name in zip(trial_types_num,trial_type_name): # distinguish bewteen different trial type
                if tt is not None:
                    trial_type_idx = (trial_type_filt==tt)
                    spiketrain_tmp = spiketrain_filt[trial_type_idx,:]
                    pos_binned_tmp = pos_binned_filt[trial_type_idx]
                else:
                    spiketrain_tmp = spiketrain_filt
                    pos_binned_tmp = pos_binned_filt

                if spiketrain_tmp.shape[0] > 30*(1000/setting.binSize):
                    #only do analysis on enough data (>30s)
                    for n in data.neuron.data:
                
                        pos_bin_range = pos_bins[(pos_bins >= p_range[0]) & (pos_bins < p_range[1])]
                        score,bp,meanCurve,normCurve = findBestRampScore2(spiketrain_tmp[:,n],
                            pos_binned_tmp,
                            pos_bin_range)

                        ramp_scores.append({
                            'score':score,
                            'breakpoint': bp,
                            'cluster_id': data.cluster_id.data[n],
                            'fr_smooth': meanCurve,
                            'ramp_region': rt,
                            'pos_bin': pos_bin_range,
                            'trial_type': tt_name
                        })
                else:
                    for n in data.neuron.data:
                        ramp_scores.append({
                            'score': np.nan,
                            'breakpoint': np.nan,
                            'cluster_id': data.cluster_id.data[n],
                            'fr_smooth': np.nan,
                            'ramp_region': rt,
                            'pos_bin': np.nan,
                            'trial_type': tt_name
                        })


        df_ramp = pd.DataFrame(ramp_scores)
        df_ramp['session_id'] = data.session_id

    
        return df_ramp


def time_modelling(data):

    if data.spiketrain.data.item(0) is None:
        # No spikes found
        return None, None
        
    #%% Identify which trials are rewarded
    trial_number = data.trial_number.data
    isRewarded = data.isRewarded.data
    isRewarded_trials = []
    unique_trials = np.unique(trial_number)
    dt = setting.binSize/1000 # in second

    

    for i in unique_trials:
        isRewarded_trials.append(isRewarded[trial_number==i][0])

    isRewarded_trials = np.array(isRewarded_trials)

    rewarded_trials = unique_trials[isRewarded_trials==1]

    if len(rewarded_trials)<5:
        return None, None

    #%% 
    # For each rewarded trial, identify the time where the animal first enter the reward zone
    # fig,ax = getWrappedSubplots(6,len(rewarded_trials),(3,3),dpi=100)
    position = data.position.data
    speed = data.speed.data
    reward_start_idx = np.zeros_like(rewarded_trials,dtype=np.int)
    trial_start_idx = np.zeros_like(rewarded_trials,dtype=np.int)

    for i,rt in enumerate(rewarded_trials):
        try:
            idx = np.where((trial_number==rt))[0] #consider start at x=30cm
            idx = idx[5:] #avoid the transisent caused by the rotatory encoding going back to zero
            position_trial = position[idx]
            trial_start_idx[i] = idx[0] + np.where(position_trial>30)[0][0]
            reward_start_idx[i] = np.where(position_trial>90)[0][0]+idx[0] - trial_start_idx[i] #relative to trial start

            # Do some basic test
            if reward_start_idx[i] < 3:
                #impossible to finish in 300s, probably something wrong in data
                reward_start_idx[i] = -1
                trial_start_idx[i] = -1
        except:
            # Probably some error ocurred in the trial detection
            trial_start_idx[i] = -1
            reward_start_idx[i] = -1

    
    # Remove invalid trials
    trial_start_idx = trial_start_idx[trial_start_idx>=0]
    reward_start_idx = reward_start_idx[reward_start_idx>=0]

    #%% Plot the raw firing rate vs time for each neuron
    spiketrain = data.spiketrain.data.astype(np.float)
    spiketrain = gaussian_filter1d(spiketrain,2,axis=0) #smooth
    spiketrain[spiketrain<0]=0

    #%%  Separate into short and long trial

    #Find the threshold for short and long trials
    median_time = np.median(reward_start_idx)
    short_thres = np.percentile(reward_start_idx,25)
    long_thres = np.percentile(reward_start_idx,75)

    trials_short= (reward_start_idx<=short_thres)
    trials_long = (reward_start_idx>long_thres)
    trials_middle = ~(trials_short | trials_long)


    trial_end_short = reward_start_idx[trials_short].min()
    trial_end_long  = reward_start_idx[trials_long].min()
    trial_end_middle = reward_start_idx[trials_middle].min()

    # Add in trial length related information
    trial_length_type = np.empty_like(position) 
    trial_length_type[:] = np.nan
    time_relative_outbound = np.empty_like(position) #time relative to the start of outbound region
    time_relative_outbound[:] = np.nan
    position_relative_outbound = np.empty_like(position) #time relative to the start of outbound region
    position_relative_outbound[:] = np.nan

    for i, t in enumerate([trials_short, trials_middle, trials_long]):
        trials = trial_start_idx[t]
        trial_length = reward_start_idx[t]
        for t,rl in zip(trials,trial_length):
            trial_length_type[t:t+rl] = i
            time_relative_outbound[t:t+rl] = np.arange(rl)*dt
            position_relative_outbound[t:t+rl] = position[t:t+rl] - position[t]

    trial_length_type_xr = xr.DataArray(data = trial_length_type, dims=('time'), coords={'time':data.time})
    time_relative_outbound = xr.DataArray(data = time_relative_outbound, dims=('time'), coords={'time':data.time})
    position_relative_outbound = xr.DataArray(data = position_relative_outbound, dims=('time'), coords={'time':data.time})

    data['trial_length_type'] = trial_length_type_xr
    data['time_relative_outbound'] = time_relative_outbound
    data['position_relative_outbound'] = position_relative_outbound

    data.attrs['trial_length_type_enum'] = ['short','middle','long']



    # %% Save

    data['reward_start_idx'] = reward_start_idx
    data['trial_start_idx'] = trial_start_idx
    data.attrs['speed_thres'] = setting.speed_thres

    # %% Export data to process in R
    fr = data.spiketrain.data/dt
    dff = pd.DataFrame({
        'position': data.position.data,
        'speed': data.speed.data,
        'spiketrain': fr.tolist(), #change to spike rate
        'trial_length_type': data.trial_length_type.data,
        'time_relative_outbound': data.time_relative_outbound.data,
        'position_relative_outbound': position_relative_outbound.data,
        'trial_number': trial_number,
        'session_id': data.session_id,
        'animal': data.animal,
        'cohort': data.cohort, 
        'cluster_id':data.cluster_id.data[0]
    })
    dff.dropna(inplace=True) #only include the outbound data
    dff.reset_index(inplace=True)

    return (data, dff)

if __name__ =='__main__':
    compute_ramp_score(None)