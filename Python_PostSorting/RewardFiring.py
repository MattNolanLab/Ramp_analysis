import numpy as np
import pandas as pd
import Python_PostSorting.Create2DHistogram
import matplotlib.pylab as plt
from scipy import signal
from scipy import stats



def split_trials_by_reward(spike_data, cluster_index):
    beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster_index)

    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)

    #take firing locations when on rewarded trials
    rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

    #take firing trial numbers when on rewarded trials
    rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]
    return rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers


def add_columns(spike_data):
    spike_data["spikes_in_time_rewarded"] = ""
    spike_data["spikes_in_time_rewarded_b"] = ""
    spike_data["spikes_in_time_rewarded_p"] = ""
    spike_data["spikes_in_time_rewarded_nb"] = ""
    return spike_data


def extract_data_from_frame(spike_data, cluster):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

    window = signal.gaussian(2, std=2)
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

    return rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types


def split_time_data_by_reward(spike_data, prm):
    spike_data = add_columns(spike_data)

    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)

        ## for all trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data, rewarded_trials)
        spike_data = drop_alldata_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for beaconed trials
        data_filtered = data[data[:,4] == 0,:] # filter data for beaconed trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_beaconed_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe trials
        data_filtered = data[data[:,4] == 2,:] # filter data for probe trials & nonbeaconed
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_probe_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe & nonbeaconed trials
        data_filtered = data[data[:,4] != 0,:] # filter data for nonbeaconed trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_nb_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        rewarded_locations = np.array(spike_data.loc[cluster, 'rewarded_locations'])
        rewarded_locations = rewarded_locations[~np.isnan(rewarded_locations)]
        locations = np.array(np.append(rewarded_locations, rewarded_locations[0:14]))
        spike_data.at[cluster,"rewarded_locations"] = list(locations)
    return spike_data


def drop_alldata_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # types
    spike_data.at[cluster_index, 'spikes_in_time_rewarded'] = list(sn)
    return spike_data


def drop_beaconed_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # types
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_b'] = list(sn)
    return spike_data


def drop_probe_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # types
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_p'] = list(sn)
    return spike_data


def drop_nb_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # types
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_nb'] = list(sn)
    return spike_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=2)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate



#_not used currently_
def split_trials_by_failure(spike_data, cluster_index):
    beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster_index)

    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)

    #take firing locations when on rewarded trials
    failed_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials, invert=True)]
    failed_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials, invert=True)]
    failed_probe_position_cm = probe_position_cm[~np.isin(probe_trial_number,rewarded_trials, invert=True)]

    #take firing trial numbers when on rewarded trials
    failed_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials, invert=True)]
    failed_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials, invert=True)]
    failed_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials, invert=True)]

    return failed_beaconed_position_cm, failed_nonbeaconed_position_cm, failed_probe_position_cm, failed_beaconed_trial_numbers, failed_nonbeaconed_trial_numbers, failed_probe_trial_numbers

