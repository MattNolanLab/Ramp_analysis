import numpy as np
import pandas as pd
import random


def add_columns_to_frame(spike_data):
    spike_data["spike_rate_on_trials_redo"] = ""
    return spike_data


def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def extract_smoothed_firing_info(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0], 'trial_number' :  np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype=np.int16), 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings


def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==0)
    nbeaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==1)
    probe_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==2)
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def find_non_beaconed_trial_number(cluster_firings):
    trials = np.array(cluster_firings["trial_number"])
    unique_trials = np.unique(trials)
    unique_trials = unique_trials[~np.isnan(unique_trials)]
    number_of_trials = unique_trials.shape[0]
    return number_of_trials


def find_unique_beaconed_trials(cluster_firings):
    trials = np.unique(np.array(cluster_firings["trial_number"].dropna(axis=0)))
    return trials


def create_random_trial_array(number_of_trials,trials):
    random_float_array = np.random.choice(trials, number_of_trials)
    #random_float_array = np.random.uniform(1, max_trial-1, size=number_of_trials)
    #random_float_array = np.asarray(random_float_array, dtype=np.int8)
    return random_float_array


def make_beaconed_frame_equal_size(b, random_float_array):
    beaconed_trial_type = np.array(b["trial_type"])
    beaconed_trial_number = np.array(b["trial_number"])
    beaconed_firing_rate = np.array(b["firing_rate"])

    beaconed_trial_type = beaconed_trial_type[np.isin(beaconed_trial_number,random_float_array)]
    beaconed_firing_rate = beaconed_firing_rate[np.isin(beaconed_trial_number,random_float_array)]
    beaconed_trial_number = beaconed_trial_number[np.isin(beaconed_trial_number,random_float_array)]

    #beaconed_trial_number = rename_trial_numbers(random_float_array, beaconed_trial_number)
    cluster_firings = pd.DataFrame({ 'firing_rate' :  beaconed_firing_rate, 'trial_number' :  beaconed_trial_number, 'trial_type' : beaconed_trial_type})
    return cluster_firings


def rename_trial_numbers(random_float_array, beaconed_trial_number):
    num_of_trials = np.unique(beaconed_trial_number).shape[0]
    new_trial_indicators = np.arange(1,num_of_trials+1, 1)
    beaconed_trial_number = np.repeat(new_trial_indicators, 200)
    return beaconed_trial_number


def fix_unequal_trial_numbers(spike_data):
    print("randomly selecting beaconed trials so trial numbers are equal across trial types...")
    spike_data = add_columns_to_frame(spike_data)

    for cluster in range(len(spike_data)):
        print("cluster ",cluster)
        cluster_firings = extract_smoothed_firing_info(spike_data, cluster)
        b, nb, p = split_firing_data_by_trial_type(cluster_firings)
        number_of_trials = find_non_beaconed_trial_number(nb)
        max_trial = find_unique_beaconed_trials(b)
        random_float_array = create_random_trial_array(number_of_trials,max_trial)
        beaconed_cluster_firings = make_beaconed_frame_equal_size(b, random_float_array)
        nb.dropna(inplace=True)
        p.dropna(inplace=True)
        cluster_firings = pd.concat([beaconed_cluster_firings, nb, p], ignore_index=True)
        spike_data = add_data_to_frame(spike_data, cluster, cluster_firings)
    return spike_data



def add_data_to_frame(spike_data, cluster_index, cluster_firings):
    sn=[]
    sn.append(np.array(cluster_firings['firing_rate']))
    sn.append(np.array(cluster_firings['trial_number']))
    sn.append(np.array(cluster_firings['trial_type']))
    spike_data.at[cluster_index, 'spike_rate_on_trials_redo'] = list(sn)
    return spike_data
