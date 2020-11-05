import numpy as np
import Python_PostSorting.ExtractFiringData
import math
import pandas as pd


'''

## shuffle analysis

'''

## Code shuffles spike times for entire sessions 1000 times to generate a 1000 shuffled datasets for each neuron.
## Adapted from Tennant et al., 2018

### --------------------------------------------------------------------------------------------------- ###

### SHUFFLE SPIKES IN SPACE


def add_column_to_dataframe(spike_data):
    spike_data["shuffled_spike_num_on_trials"] = ""
    spike_data["shuffled_spike_rate_on_trials"] = ""
    return spike_data


def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def shuffle_analysis1(cluster_firings, trialids):
    shuffledtrials = np.zeros((0, 3))
    for trial in range(1,int(trialids)): # Create sample data with 100 trials
        trial_data = cluster_firings[cluster_firings[:,1] == trial,:]# get data only for each tria
        type=np.nanmedian(trial_data[:,2])
        shuffledtrial = shuffle_stops_multiple(trial_data,trial, type) # shuffle the locations of stops in the trial
        shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop
    return shuffledtrials


def shuffle_stops( spikes,trial, type):
    shuffled_spikes = np.copy(spikes) # this is required as otherwise the original dataset would be altered
    np.random.shuffle(shuffled_spikes[:,0])
    return shuffled_spikes


def shuffle_stops_multiple( spikes,trial, type):
    shufflen=1000
    shuffled_trials = np.zeros((spikes.shape[0]))
    for n in range(shufflen):
        shuffled_spikes = np.copy(spikes) # this is required as otherwise the original dataset would be altered
        np.random.shuffle(shuffled_spikes[:,0])
        shuffled_trials = np.vstack((shuffled_trials, shuffled_spikes[:,0]))
    avg_shuffled_trials=np.nanmean(shuffled_trials, axis=0)
    shuffled_spikes[:,0] = avg_shuffled_trials
    return shuffled_spikes


def make_location_bins(shuffled_cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(shuffled_cluster_firings[:,1])
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def add_data_to_dataframe(cluster_index, shuffled_cluster_firings, spike_data):
    bin_array = make_location_bins(shuffled_cluster_firings)
    sn=[]
    sn.append(np.array(shuffled_cluster_firings[:,0]))
    sn.append(np.array(shuffled_cluster_firings[:,1]))
    sn.append(np.array(shuffled_cluster_firings[:,2]))
    if bin_array.shape[0] == shuffled_cluster_firings.shape[0]:
        sn.append(bin_array)
    else:
        print("bins are not the right size")
    spike_data.at[cluster_index, 'shuffled_spike_num_on_trials'] = list(sn)
    return spike_data


def generate_shuffled_data(spike_data, processed_position_data):
    spike_data=add_column_to_dataframe(spike_data)

    for cluster in range(len(spike_data)):
        cluster_firings = extract_firing_info(spike_data, cluster)
        max_trial = np.max(cluster_firings[:,1])-1
        shuffled_cluster_firings = shuffle_analysis1(cluster_firings, max_trial)
        spike_data = add_data_to_dataframe(cluster, shuffled_cluster_firings, spike_data)

        shuffled_rate_map= Python_PostSorting.ExtractFiringData.extract_shuffled_firing_num_data(spike_data, cluster)
        #shuffled_rate_map = normalise_shuffled_spike_number_by_time(shuffled_rate_map, shuffled_time)
        spike_data = add_rate_data_to_dataframe(cluster, shuffled_cluster_firings, shuffled_rate_map, spike_data)
    return spike_data


def shuffle_dwell_time(dwell_time):
    for n in range(1000):
        np.random.shuffle(dwell_time)
    return dwell_time


def generate_multi_shuffled_data(spike_data):
    print('generating shuffled data')
    spike_data=add_column_to_dataframe(spike_data)

    for cluster in range(len(spike_data)):
        cluster_firings = extract_firing_info(spike_data, cluster)
        max_trial = np.max(cluster_firings[:,1])+1
        shuffled_cluster_firings = shuffle_analysis1(cluster_firings, max_trial)
        spike_data = add_data_to_dataframe(cluster, shuffled_cluster_firings, spike_data)

        #shuffled_rate_map= Python_PostSorting.ExtractFiringData.extract_shuffled_firing_num_data(spike_data, cluster)
        #shuffled_time = shuffle_dwell_time(pd.Series(spike_data.at[cluster, 'binned_time_ms_per_trial']))
        #shuffled_rate_map = normalise_shuffled_spike_number_by_time(shuffled_rate_map, shuffled_time)
        #shuffled_rate_map['normalised_firing_rate'] = np.nan_to_num(np.where(shuffled_rate_map['firing_rate'] >0, shuffled_rate_map['firing_rate'],0)) # uncomment if shuffling spike number
        #spike_data = add_rate_data_to_dataframe(cluster, shuffled_cluster_firings, shuffled_rate_map, spike_data)
    return spike_data


'''

## firing rate maps for shuffle analysis

'''


def normalise_shuffled_spike_number_by_time(shuffled_rate_map, processed_position_data_dwell_time):
    shuffled_rate_map['dwell_time'] = processed_position_data_dwell_time
    shuffled_rate_map['normalised_firing_rate'] = np.where(shuffled_rate_map['firing_rate'] >0, shuffled_rate_map['firing_rate']/shuffled_rate_map['dwell_time'], 0)
    shuffled_rate_map['normalised_firing_rate'].fillna(0, inplace=True)
    shuffled_rate_map.loc[shuffled_rate_map['normalised_firing_rate'] > 250, 'normalised_firing_rate'] = 0
    return shuffled_rate_map


def add_rate_data_to_dataframe(cluster_index, shuffled_cluster_firings, shuffled_rate_map, spike_data):
    bin_array = make_location_bins(shuffled_cluster_firings)
    sr=[]
    sr.append(np.array(shuffled_rate_map['normalised_firing_rate']))
    sr.append(np.array(shuffled_rate_map['trial_number'], dtype=np.int16))
    sr.append(np.array(shuffled_rate_map['trial_type'], dtype=np.int16))
    if bin_array.shape[0] == shuffled_rate_map.shape[0]:
        sr.append(bin_array)
    else:
        print("bins are not the right size")
    spike_data.at[cluster_index, 'shuffled_spike_rate_on_trials'] = list(sr)

    return spike_data



'''

## rolling average for smoothing

'''

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:] / window


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out




### --------------------------------------------------------------------------------------------------- ###

### SHUFFLE SPIKES IN TIME

## Code shuffles spike rate (calculaed in time) for entire sessions 1000 times to generate a 1000 shuffled datasets for each neuron.


def add_columns_to_dataframe(spike_data):
    spike_data["shuffled_spikes_in_time"] = ""
    return spike_data


def extract_time_binned_data(spike_data, cluster_index):
    rate=np.array(spike_data.iloc[cluster_index].spike_rate_in_time[0])*10
    speed=np.array(spike_data.iloc[cluster_index].spike_rate_in_time[1])
    position=np.array(spike_data.iloc[cluster_index].spike_rate_in_time[2])
    trials=np.array(spike_data.iloc[cluster_index].spike_rate_in_time[3], dtype= np.int32)
    types=np.array(spike_data.iloc[cluster_index].spike_rate_in_time[4], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types, position, speed)))
    return cluster_firings


def shuffle_spikes_in_time( spikes):
    shufflen=1000
    shuffled_trials = np.zeros((spikes.shape[0]))
    for n in range(shufflen):
        shuffled_spikes = np.copy(spikes) # this is required as otherwise the original dataset would be altered
        np.random.shuffle(shuffled_spikes[:,0])
        shuffled_trials = np.vstack((shuffled_trials, shuffled_spikes[:,0]))
    avg_shuffled_trials=np.nanmean(shuffled_trials, axis=0)
    #sd_shuffled_trials=np.nanstd(shuffled_trials, axis=0)
    shuffled_spikes[:,0] = avg_shuffled_trials
    return shuffled_spikes


def shuffle_analysis(cluster_firings, trialids):
    #shuffledtrials = np.zeros((0, 5))
    #for trial in range(1,int(trialids)):
        #trial_data = cluster_firings[cluster_firings[:,1] == trial,:]# get data only for each trial
    shuffledtrial = shuffle_spikes_in_time(cluster_firings) # shuffle the locations of spikes in the trial
    #shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop
    return shuffledtrial


def generate_shuffled_data_for_time_binned_data(spike_data):
    print('generating shuffled data')
    spike_data=add_columns_to_dataframe(spike_data)

    for cluster in range(len(spike_data)):
        cluster_firings = extract_time_binned_data(spike_data, cluster)
        max_trial = np.max(cluster_firings[:,1])+1
        shuffled_cluster_firings = shuffle_analysis(cluster_firings, max_trial)
        spike_data = add_data_to_dataframe1(cluster, shuffled_cluster_firings, spike_data)
    return spike_data



def add_data_to_dataframe1(cluster_index, shuffled_cluster_firings, spike_data):
    sn=[]
    sn.append(np.array(shuffled_cluster_firings[:,0])) # rate
    sn.append(np.array(shuffled_cluster_firings[:,4])) # speed
    sn.append(np.array(shuffled_cluster_firings[:,3])) # position
    sn.append(np.array(shuffled_cluster_firings[:,1])) # trials
    sn.append(np.array(shuffled_cluster_firings[:,2])) # types

    spike_data.at[cluster_index, 'shuffled_spikes_in_time'] = list(sn)
    return spike_data
