import os
import matplotlib.pylab as plt
import Python_PostSorting.ConvolveRates_FFT
import numpy as np
import cmath
import pandas as pd
from scipy import signal


### --------------------------------------------------------------------------------------------------- ###

### ACCELERATION OVER SPACE



def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def make_location_bins(shuffled_cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(shuffled_cluster_firings[:,1])
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def bin_acceleration_over_space(acceleration, position, trials, bin_array, max_trial):
    data = np.transpose(np.vstack((acceleration,position, trials)))
    bins = np.arange(0,200,1)
    accel_over_trials = np.zeros((len(bins), max_trial+1))
    for tcount, trial in enumerate(np.arange(1,max_trial,1)):
        trial_data = data[data[:,2] == trial,:]# get data only for each trial
        for bcount, bin in enumerate(bins):
            accel_above_position = trial_data[trial_data[:,1] >= bin ,:]# get data only for bins +
            accel_in_position = accel_above_position[accel_above_position[:,1] < bin+1 ,:]# get data only for bin
            mean_accel = np.nanmean(accel_in_position[:,0])
            accel_over_trials[bcount, tcount] = mean_accel
    return accel_over_trials



def plot_avg_acceleration(recording_folder, spike_data, cluster, acceleration):
    spike_data["avg_acceleration"] = ""
    shapes = int(acceleration.size/200) * 200
    acceleration = acceleration[:shapes]
    reshaped_hist = np.reshape(acceleration, (int(acceleration.size/200),200))
    speed = np.array(spike_data.at[cluster, "Speed_averaged"])

    plt.plot(speed, color="Red")
    plt.plot(np.nanmean(reshaped_hist, axis=0), color="Black")
    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_Avg_accel_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return


def plot_trial_acceleration(recording_folder, spike_data, cluster, acceleration):
    speed=np.array(spike_data.loc[cluster].speed_rate_on_trials[0], dtype= np.int32)
    trials=np.array(spike_data.loc[cluster].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    bins=np.arange(1,201,1)
    max_trial = np.max(trials)+1
    position= np.tile(bins,int(max_trial))

    data = np.vstack((trials, speed, position, acceleration))
    trial_data = data[data[:,0] == 10,:] # get data only for one trial

    plt.plot(speed, color="Red")
    plt.plot(trial_data[:,2], trial_data[:,3], color="Black")

    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_Trial_accel_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return


def calculate_acceleration_binned_in_space(recording_folder, spike_data):
    print('I am calculating acceleration over space...')
    spike_data["acceleration_rate_on_trials"] = ""
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        cluster_firings = extract_firing_info(spike_data, cluster)
        bin_array = make_location_bins(cluster_firings)

        acceleration =  np.array(spike_data.loc[cluster_index].spikes_in_time[3])
        speed =  np.array(spike_data.loc[cluster_index].spikes_in_time[2])
        position=np.array(spike_data.loc[cluster_index].spikes_in_time[1])
        trials=np.array(spike_data.loc[cluster_index].spikes_in_time[4], dtype= np.int32)
        max_trial = np.max(trials)

        binned_acceleration = bin_acceleration_over_space(acceleration, position, trials, bin_array, max_trial)

        plot_avg_acceleration(recording_folder, spike_data, cluster, binned_acceleration)
        plot_trial_acceleration(recording_folder, spike_data, cluster, binned_acceleration)

        spike_data = add_rate_data_to_dataframe(cluster, cluster_firings, binned_acceleration, spike_data, bin_array)
    return spike_data



def add_rate_data_to_dataframe(cluster_index, cluster_firings, binned_acceleration, spike_data, bin_array):
    sr=[]
    sr.append(np.array(binned_acceleration))
    sr.append(np.array(cluster_firings[:,1], dtype=np.int32))
    sr.append(np.array(cluster_firings[:,2], dtype=np.int32))
    sr.append(bin_array)

    spike_data.at[cluster_index, 'acceleration_rate_on_trials'] = list(sr)

    return spike_data

