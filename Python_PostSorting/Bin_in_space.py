import numpy as np
import pandas as pd
import Python_PostSorting.plot_utility
import Python_PostSorting.settings
from astropy.convolution import convolve, Gaussian1DKernel


def calculate_total_trial_numbers(raw_position_data,processed_position_data):
    print('calculating total trial numbers for trial types')
    trial_numbers = np.array(raw_position_data['trial_number'])
    trial_type = np.array(raw_position_data['trial_type'])
    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))
    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)
    unique_beaconed_trials = np.unique(beaconed_trials[:,0])
    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)
    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])
    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)
    unique_probe_trials = np.unique(probe_trials[1:,0])

    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)
    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)
    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)
    return processed_position_data


def trial_average_speed(processed_position_data):
    # split binned speed data by trial type
    beaconed = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed = processed_position_data[processed_position_data["trial_type"] == 1]
    probe = processed_position_data[processed_position_data["trial_type"] == 2]

    if len(beaconed)>0:
        beaconed_speeds = Python_PostSorting.plot_utility.pandas_column_to_2d_numpy_array(beaconed["speeds_binned"])
        trial_averaged_beaconed_speeds = np.nanmean(beaconed_speeds, axis=0)
    else:
        trial_averaged_beaconed_speeds = np.array([])

    if len(non_beaconed)>0:
        non_beaconed_speeds = Python_PostSorting.plot_utility.pandas_column_to_2d_numpy_array(non_beaconed["speeds_binned"])
        trial_averaged_non_beaconed_speeds = np.nanmean(non_beaconed_speeds, axis=0)
    else:
        trial_averaged_non_beaconed_speeds = np.array([])

    if len(probe)>0:
        probe_speeds = Python_PostSorting.plot_utility.pandas_column_to_2d_numpy_array(probe["speeds_binned"])
        trial_averaged_probe_speeds = np.nanmean(probe_speeds, axis=0)
    else:
        trial_averaged_probe_speeds = np.array([])

    return trial_averaged_beaconed_speeds, trial_averaged_non_beaconed_speeds, trial_averaged_probe_speeds


def bin_in_space(spike_data, track_length):
    gauss_kernel = Gaussian1DKernel(Python_PostSorting.settings.guassian_std_for_smoothing_in_space_cm/Python_PostSorting.settings.vr_bin_size_cm)

    for cluster in range(len(spike_data)):

        speeds_binned_in_space = []

        for trial_number in range(1, max(spike_data["max_trial_number"]+1)):
            trial_x_position_cm = np.array(spike_data.loc[cluster, 'x_position_cm'][np.array(spike_data.loc[cluster, 'trial_number', cluster]) == trial_number], dtype="float64")
            trial_speeds = np.array(spike_data.loc[cluster, 'speed_per200ms'][np.array(spike_data.loc[cluster, 'trial_number']) == trial_number], dtype="float64")

            pos_bins = np.arange(0, track_length, Python_PostSorting.settings.vr_bin_size_cm)# 1cm space bins

            if len(pos_bins)>1:
                # calculate the average speed and position in each space bin
                speed_bin_means, pos_bin_edges = np.histogram(trial_x_position_cm, pos_bins, weights=trial_speeds)
                speed_bin_means = speed_bin_means/np.histogram(trial_x_position_cm, pos_bins)[0]

                # and smooth
                speed_bin_means = convolve(speed_bin_means, gauss_kernel)

            else:
                speed_bin_means = []


            speeds_binned_in_space.append(speed_bin_means)

        spike_data.at[cluster, "speeds_binned_in_space"] = speeds_binned_in_space

    return spike_data
