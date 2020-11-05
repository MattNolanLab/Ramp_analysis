import pandas as pd
import numpy as np
from scipy import stats

def extract_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.iloc[cluster_index].spike_rate_on_trials[0], 'trial_number' :  spike_data.iloc[cluster_index].spike_rate_on_trials[1], 'trial_type' :  spike_data.iloc[cluster_index].spike_rate_on_trials[2]})
    return cluster_firings


def extract_smoothed_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0], 'trial_number' :  np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype=np.int16), 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings


def extract_firing_num_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].spike_num_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].spike_num_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].spike_num_on_trials[2]})
    return cluster_firings


def extract_shuffled_firing_num_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[2], 'bins' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[3]})
    return cluster_firings


def extract_shuffled_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[2], 'bins' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[3]})
    return cluster_firings


def extract_average_speed_data(spike_data, cluster_index):
    cluster_speed = pd.DataFrame({ 'speed_b' :  spike_data.loc[cluster_index].average_speed[0], 'speed_nb' :  spike_data.loc[cluster_index].average_speed[1], 'speed_p' :  spike_data.loc[cluster_index].average_speed[2]})
    return cluster_speed


def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==0)
    nbeaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==1)
    probe_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==2)
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def reshape_and_average_over_trials(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings, max_trial_number):
    beaconed_cluster_firings[beaconed_cluster_firings>100] = 100
    nonbeaconed_cluster_firings[nonbeaconed_cluster_firings>100] = 100
    probe_cluster_firings[probe_cluster_firings>100] = 100

    beaconed_reshaped_hist = np.reshape(beaconed_cluster_firings, (int(beaconed_cluster_firings.size/200),200))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_cluster_firings, (int(nonbeaconed_cluster_firings.size/200), 200))
    probe_reshaped_hist = np.reshape(probe_cluster_firings, (int(probe_cluster_firings.size/200), 200))
    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)
    average_beaconed_spike_rate = np.nan_to_num(average_beaconed_spike_rate, copy=True)
    average_nonbeaconed_spike_rate = np.nan_to_num(average_nonbeaconed_spike_rate, copy=True)
    average_probe_spike_rate = np.nan_to_num(average_probe_spike_rate, copy=True)

    average_beaconed_sd = np.nanstd(beaconed_reshaped_hist, axis=0)/2

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16), average_beaconed_sd


def extract_average_shuffled_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_shuffled_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16)


def extract_average_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd


def extract_smoothed_average_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd


def extract_average_firing_num_data(spike_data, cluster_index):
    cluster_firings = extract_firing_num_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd


def extract_smoothed_average_firing_rate_data_tracks(spike_data, cluster_index):
    cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd




### ------------------------------------------------------------------------###

## Functions to split data based on rewarded trials

def split_firing_data_by_reward(cluster_firings, rewarded_trials):
    #rewarded_cluster_firings= cluster_firings[cluster_firings['trial_type'].isin(rewarded_trials)]    #failed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] not in rewarded_trials)
    rewarded_cluster_firings = cluster_firings.loc[cluster_firings['trial_number'].isin(rewarded_trials)]
    rewarded_cluster_firings.reset_index(drop=True, inplace=True)
    return rewarded_cluster_firings



def split_firing_data_by_failure(cluster_firings, rewarded_trials):
    #rewarded_cluster_firings= cluster_firings[cluster_firings['trial_type'].isin(rewarded_trials)]    #failed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] not in rewarded_trials)
    rewarded_cluster_firings = cluster_firings.loc[~cluster_firings['trial_number'].isin(rewarded_trials)]
    rewarded_cluster_firings.reset_index(drop=True, inplace=True)
    return rewarded_cluster_firings



def extract_smoothed_average_firing_rate_data_for_rewarded_trials(spike_data, cluster_index):
    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)
    rewarded_trials = rewarded_trials[rewarded_trials >0]
    cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster_index)
    cluster_firings = split_firing_data_by_reward(cluster_firings, rewarded_trials)
    max_trials = int(np.shape(np.unique(rewarded_trials))[0])
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max_trials)
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, average_beaconed_sd



def extract_smoothed_average_firing_rate_data_for_failed_trials(spike_data, cluster_index):
    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)
    failed_trials = rewarded_trials[rewarded_trials > 0]
    cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster_index)
    cluster_firings = split_firing_data_by_failure(cluster_firings, failed_trials)
    max_trials = int(np.shape(np.unique(failed_trials))[0])
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max_trials)
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate, sd



## for within plot
## split x_position_cm based on trial type for each cluster
def split_firing_by_trial_type(spike_data, cluster):
    trials = np.array(spike_data.at[cluster, 'trial_number'])
    locations = np.array(spike_data.at[cluster, 'x_position_cm'])
    trial_type = np.array(spike_data.at[cluster, 'trial_type'])
    beaconed_locations = np.take(locations, np.where(trial_type == 0)[0]) #split location and trial number
    nonbeaconed_locations = np.take(locations,np.where(trial_type == 1)[0])
    probe_locations = np.take(locations, np.where(trial_type == 2)[0])
    beaconed_trials = np.take(trials, np.where(trial_type == 0)[0])
    nonbeaconed_trials = np.take(trials, np.where(trial_type == 1)[0])
    probe_trials = np.take(trials, np.where(trial_type == 2)[0])

    return beaconed_locations, nonbeaconed_locations, probe_locations, beaconed_trials, nonbeaconed_trials, probe_trials


# for outside plot
## split x_position_cm based on trial type for each cluster
def split_spatial_firing_by_trial_type(spike_data):
    for cluster in range(len(spike_data)):
        trials = np.array(spike_data.loc[cluster, 'trial_number'])
        locations = np.array(spike_data.loc[cluster, 'x_position_cm'])
        trial_type = np.array(spike_data.loc[cluster, 'trial_type'])
        beaconed_locations = np.take(locations, np.where(trial_type == 0)[1]) #split location and trial number
        nonbeaconed_locations = np.take(locations,np.where(trial_type == 1)[1])
        probe_locations = np.take(locations, np.where(trial_type == 2)[1])
        beaconed_trials = np.take(trials, np.where(trial_type == 0)[1])
        nonbeaconed_trials = np.take(trials, np.where(trial_type == 1)[1])
        probe_trials = np.take(trials, np.where(trial_type == 2)[1])

    return beaconed_locations, nonbeaconed_locations, probe_locations, beaconed_trials, nonbeaconed_trials, probe_trials
