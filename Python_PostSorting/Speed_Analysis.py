import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import math
from scipy import signal



## histogram of speeds

def generate_speed_histogram(spike_data, recording_folder):
    print('I am calculating speed histogram...')
    spike_data["speed_histogram"] = ""
    save_path = recording_folder + 'Figures/Speed_histogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real/1000
        speed = speed[speed > 0]
        try:
            posrange = np.linspace(0, 100, num=100)
            values = np.array([[posrange[0], posrange[-1]]])
            H, bins = np.histogram(speed, bins=(posrange), range=values)
            plt.plot(bins[1:], H)
            #plt.hist(speed, density=True, bins=50)
            #plt.xlim(-5, 100)
            plt.ylabel('Probability')
            plt.xlabel('Data')
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_histogram_Cluster_' + str(cluster +1) + '_1' + '.png', dpi=200)
            plt.close()

            spike_data.at[cluster,"speed_histogram"] = [H, bins]
        except ValueError:
            continue
    return spike_data


### --------------------------------------------------------------------------------------- ###


## here for testing

def fix_position(position):
    position_fixed = np.zeros((position.shape[0]))
    for rowcount, row in enumerate(position[1:]):
        position_diff = position[rowcount] - position[rowcount-1]
        next_position_diff = position[rowcount]+1 - rowcount
        if position_diff < -110:
            position_fixed[rowcount] = 0
        else:
            position_fixed[rowcount] = position[rowcount]
    return position_fixed


def cumulative_position(position):
    position_fixed = np.zeros((position.shape[0]))
    total_position=0
    for rowcount, row in enumerate(position[1:]):
        position_diff = position[rowcount] - position[rowcount-1]
        if position_diff < -20:
            total_position+=200
        cum_position = position[rowcount] + total_position
        position_fixed[rowcount] = cum_position
    return position_fixed


def calculate_speed_from_position(spike_data, recording_folder):
    print('I am calculating speed from position...')
    save_path = recording_folder + 'Figures/Speed_from_Position'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_path = recording_folder + 'Figures/Speed_from_Position'

    spike_data["speed"] = ""
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real, dtype=np.float32)/1000
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real, dtype=np.float32)
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real, dtype=np.float32)*10
        try:
            position = fix_position(position)
            cum_position = cumulative_position(position)
            pos_diff = np.diff(cum_position)
            pos_diff = pos_diff*4
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)

            plt.scatter(pos_diff, rates[1:], marker='o', s=1)
            plt.scatter(speed, rates, s=1)
            plt.xlim(-50, 60)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_Cluster_' + str(cluster +1) + '_3' + '.png', dpi=200)
            plt.close()

        except IndexError:
            print(session_id)

    return spike_data





### --------------------------------------------------------------------------------------- ###

 ### Bin speed in space


def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def make_location_bins(shuffled_cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(shuffled_cluster_firings[:,1])
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def bin_speed_over_space(speed, position, trials, bin_array, max_trial):
    data = np.transpose(np.vstack((speed,position, trials)))
    bins = np.arange(0,200,1)
    accel_over_trials = np.zeros((len(bins), max_trial))
    for tcount, trial in enumerate(np.arange(1,max_trial+1,1)):
        trial_data = data[data[:,2] == trial,:]# get data only for each trial
        for bcount, bin in enumerate(bins):
            accel_above_position = trial_data[trial_data[:,1] >= bin ,:]# get data only for bins +
            accel_in_position = accel_above_position[accel_above_position[:,1] <= bin+1 ,:]# get data only for bin
            mean_accel = np.nanmean(accel_in_position[:,0])
            accel_over_trials[bcount, tcount] = mean_accel
    return accel_over_trials


def calculate_speed_binned_in_space(recording_folder, spike_data):
    print('I am calculating speed over space...')
    spike_data["speed_rate_on_trials"] = ""
    for cluster in range(len(spike_data)):
        cluster_firings = extract_firing_info(spike_data, cluster)
        bin_array = make_location_bins(cluster_firings)

        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        trials = np.array(spike_data.iloc[cluster].spike_rate_in_time[4], dtype= np.int32)
        max_trial = np.max(trials)+1

        binned_speed = bin_speed_over_space(speed, position, trials, bin_array, max_trial)

        #plot_avg_speed(recording_folder, spike_data, cluster, binned_acceleration)
        #plot_trial_acceleration(recording_folder, spike_data, cluster, binned_acceleration, trials, position, speed)

        spike_data = add_speed_to_dataframe(cluster, binned_speed, spike_data)
    return spike_data



def add_speed_to_dataframe(cluster_index, binned_acceleration, spike_data):
    spike_data.at[cluster_index, 'speed_rate_on_trials'] = binned_acceleration

    return spike_data

