import numpy as np
import pandas as pd




def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    speed = np.array(spike_data.iloc[cluster_index, "speed_rate_on_trials"])
    bin_array = make_location_bins(trials)

    cluster_firings=np.transpose(np.vstack((rate, bin_array, speed, trials, types)))
    return cluster_firings


def make_location_bins(trials):
    bins=np.arange(1,201,1)
    max_trial = np.max(trials)
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def generate_journey_start(cluster_firings):
    # mark in array where speed < below threshold at the start of the track in a successful trial
    return cluster_firings


def generate_journey_end(cluster_firings):
    #  mark in array where speed < below threshold in the reward zone in a successful trial
    return cluster_firings


def load_raw(spike_data, cluster_index):
    firing_times = np.array(spike_data.at[cluster_index, "firing_times"])
    x_position_cm = np.array(spike_data.at[cluster_index, "x_position_cm"])
    trial_numbers = np.array(spike_data.at[cluster_index, "trial_numbers"])
    data = np.hstack((firing_times, x_position_cm, trial_numbers))
    return data


def calculate_number_of_journeys(cluster_firings):
    journeys = cluster_firings # number of successful journeys
    return journeys


def extract_raw_spikes_for_journey(raw_spikes, j):
    raw_journey_spikes = raw_spikes == j
    return raw_journey_spikes


def extract_each_journey(cluster_firings, spike_data, cluster_index):
    # extract raw data for cluster
    raw_spikes = load_raw(spike_data, cluster_index)
    journeys = calculate_number_of_journeys(cluster_firings)

    # rebin data based on min journey
    for j in journeys:
        j_data = cluster_firings() #extract journey from cluster_firings
        min_location = np.min(j_data) # min location
        max_location = np.max(j_data) # max location
        bin_interval = (max_location - min_location)/200 # interval to rebin data by
        bins = np.arange(min_location, max_location, bin_interval) # number of bins to rescale into

        raw_journey_spikes = extract_raw_spikes_for_journey(raw_spikes, j)
        #rebin_from_raw(cluster_firings, spike_data, cluster_index, raw_journey_spikes, bins)
    return



def rebin_from_raw(cluster_firings, spike_data, cluster_index, raw_journey_spikes, bins):
    # data = np.hstack((firing_times, x_position_cm, trial_numbers)) # gather raw data

    #take data == to trial from
    return


def extract_journey(spike_data):
    print('I am extracting journeys...')
    spike_data["data_journeys"] = ""
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        cluster_firings = extract_firing_info(spike_data, cluster_index)

        cluster_firings = generate_journey_start(cluster_firings)
        cluster_firings = generate_journey_end(cluster_firings)

        extract_each_journey(cluster_firings)



    return spike_data




def add_data_to_dataframe(cluster_index, cluster_firings, binned_acceleration, spike_data, bin_array):
    sr=[]
    sr.append(np.array(binned_acceleration))
    sr.append(np.array(cluster_firings[:,1], dtype=np.int32))
    sr.append(np.array(cluster_firings[:,2], dtype=np.int32))
    sr.append(bin_array)

    spike_data.at[cluster_index, 'acceleration_rate_on_trials'] = list(sr)

    return spike_data

