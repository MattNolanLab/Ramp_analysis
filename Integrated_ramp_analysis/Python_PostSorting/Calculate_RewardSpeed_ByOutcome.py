import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats


def extract_data_from_frame(spike_data, cluster):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real) # convert from 100 ms sampling rate to Hz
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

    window = signal.gaussian(2, std=3)
    speed = signal.convolve(speed, window, mode='same')/ sum(window)
    data = np.vstack((rates, speed, position, trials, types))
    data=data.transpose()
    return rewarded_trials, data


def split_trials(data, rewarded_trials):
    rates = data[:,0]
    speed = data[:,1]
    position = data[:,2]
    trials = data[:,3]
    types = data[:,4]

    rewarded_rates = rates[np.isin(trials,rewarded_trials)]
    rewarded_speed = speed[np.isin(trials,rewarded_trials)]
    rewarded_position = position[np.isin(trials,rewarded_trials)]
    reward_trials = trials[np.isin(trials,rewarded_trials)]
    reward_types = types[np.isin(trials,rewarded_trials)]
    failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
    failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
    failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
    failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
    failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

    data = np.vstack((rewarded_rates, rewarded_speed, rewarded_position, reward_trials, reward_types))
    hit_data = data.transpose()

    data = np.vstack((failed_rates, failed_speed, failed_position, failed_trials , failed_types))
    miss_data=data.transpose()

    return hit_data, miss_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=2)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def find_avg_speed_in_reward_zone(data):
    trialids = np.array(np.unique(np.array(data[:,3])))
    speeds = np.zeros((trialids.shape[0]))
    for trialcount, trial in enumerate(trialids):
        trial_data = data[data[:,3] == trial,:] # get data only for each trial
        data_in_position = trial_data[trial_data[:,2] >= 90,:]
        data_in_position = data_in_position[data_in_position[:,2] <= 110,:]
        speeds[trialcount] = np.nanmean(data_in_position[:,1])
    return speeds, trialids


def find_confidence_interval(speeds):
    mean, sigma = np.nanmean(speeds), np.nanstd(speeds)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]
    return upper, lower


def catagorise_failed_trials(speed_array_failed, trialids_failed, upper, lower):
    trial_id_runthrough = []
    trial_id_try = []

    for rowcount, row in enumerate(speed_array_failed):
        speed = speed_array_failed[rowcount]
        trial = trialids_failed[rowcount]

        if speed < upper:
            trial_id_try = np.append(trial_id_try, trial)
        elif speed >= upper:
            trial_id_runthrough = np.append(trial_id_runthrough, trial)
    return trial_id_try, trial_id_runthrough


def extract_rewarded_confidence_intervals(data):
    speeds_rewarded, trialids_rewarded = find_avg_speed_in_reward_zone(data)
    upper, lower = find_confidence_interval(speeds_rewarded)
    rewarded_ci = np.array((upper, lower))
    return rewarded_ci



def extract_reward_zone_speed(data):
    trialids = np.array(np.unique(np.array(data[:,3])))
    speeds = np.zeros((trialids.shape[0]))
    for trialcount, trial in enumerate(trialids):
        trial_data = data[data[:,3] == trial,:] # get data only for each trial
        data_in_position = trial_data[trial_data[:,2] >= 90,:]
        data_in_position = data_in_position[data_in_position[:,2] <= 110,:]
        speeds[trialcount] = np.nanmean(data_in_position[:,1])
    return speeds


def extract_reward_zone_speed_failed_trials(speed_array_failed, upper):
    runthrough_list = []
    try_list = []

    for rowcount, row in enumerate(speed_array_failed):
        speed = speed_array_failed[rowcount]
        if speed > upper:
            runthrough_list = np.append(runthrough_list, speed)
        elif speed <= upper:
            try_list = np.append(try_list, speed)
    return try_list, runthrough_list


def calc_histo_speed(spike_data):
    spike_data["rewardzone_speed_try"] = ""
    spike_data["rewardzone_speed_run"] = ""
    spike_data["rewardzone_speed_hit"] = ""
    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)  #load all data
        hit_data, miss_data = split_trials(data, rewarded_trials)   #split on hit/mmiss

        rewarded_speeds = extract_reward_zone_speed(hit_data)
        rewarded_ci = extract_rewarded_confidence_intervals(hit_data)
        speed_array_failed, trialids_failed = find_avg_speed_in_reward_zone(miss_data)

        try_speeds, run_speeds = extract_reward_zone_speed_failed_trials(speed_array_failed, rewarded_ci[0])

        try:
            spike_data.at[cluster,"rewardzone_speed_try"] = list(try_speeds)
            spike_data.at[cluster,"rewardzone_speed_run"] = list(run_speeds)
            spike_data.at[cluster,"rewardzone_speed_hit"] = list(rewarded_speeds)
        except ValueError:
            print("")
    return spike_data

