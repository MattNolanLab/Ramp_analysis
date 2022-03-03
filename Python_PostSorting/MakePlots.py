import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import numpy as np
import Python_PostSorting.ExtractFiringData
import math
import Python_PostSorting.Split_DataByReward
import pandas as pd
from scipy import stats
from scipy import signal
import Python_PostSorting.MakePlots_Behaviour


"""

## cluster spatial firing properties
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate plots

"""


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    convolved_rate = signal.convolve(rate, window, mode='same')/sum(window)
    return convolved_rate


def test():
    return


def find_max_y_value(b, nb, p):
    nb_x_max = np.nanmax(b)
    b_x_max = np.nanmax(nb)
    p_x_max = np.nanmax(p)
    if b_x_max > nb_x_max and b_x_max > p_x_max:
        x_max = b_x_max
    elif nb_x_max > b_x_max and nb_x_max > p_x_max:
        x_max = nb_x_max
    elif p_x_max > b_x_max and p_x_max > nb_x_max:
        x_max = p_x_max

    try:
        x_max = x_max+(x_max/10)
    except UnboundLocalError:
        return 50
    return x_max


def plot_smoothed_firing_rate_maps_per_trialtype(recording_folder, spike_data):
    spike_data["Avg_FR_in_space_beaconed"] = ""
    spike_data["Avg_FR_in_space_nonbeaconed"] = ""
    spike_data["Avg_FR_in_space_probe"] = ""

    print('I am plotting smoothed firing rate maps for trial types...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #bins=np.arange(0.5,199.5,1)
    bins=np.arange(0,200,1)
    #bins=np.arange(0.5,199.5,1)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)
        #avg_beaconed_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_harry_average_firing_rate_data(spike_data, cluster)
        #avg_beaconed_spike_rate=np.array(spike_data.loc[cluster, 'fr_binned_in_space'])

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1)+ '_beaconed.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_probe_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_probe_spike_rate-sd,avg_probe_spike_rate+sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.ylim(0)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_probe.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_nonbeaconed_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_nonbeaconed_spike_rate-sd,avg_nonbeaconed_spike_rate+sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_nbeaconed.png', dpi=200)
        plt.close()

        spike_data.at[cluster, 'Avg_FR_in_space_beaconed'] = list(avg_beaconed_spike_rate)# add data to dataframe
        spike_data.at[cluster, 'Avg_FR_in_space_nonbeaconed'] = list(avg_nonbeaconed_spike_rate)# add data to dataframe
        spike_data.at[cluster, 'Avg_FR_in_space_probe'] = list(avg_probe_spike_rate)# add data to dataframe

    return spike_data





def remake_trial_numbers(rewarded_beaconed_trial_numbers):
    unique_trials = np.unique(rewarded_beaconed_trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = rewarded_beaconed_trial_numbers[rewarded_beaconed_trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n +=1
    return new_trial_numbers, unique_trials


def remake_probe_trial_numbers(rewarded_beaconed_trial_numbers):
    unique_trials = np.unique(rewarded_beaconed_trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = rewarded_beaconed_trial_numbers[rewarded_beaconed_trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n +=10
    return new_trial_numbers, unique_trials


def plot_rewarded_spikes_on_track(recording_folder,spike_data):
    print('plotting spike rasters for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_plots2'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers = Python_PostSorting.Split_DataByReward.split_trials_by_reward(spike_data, cluster)
        trials, unique_trials = remake_trial_numbers(rewarded_beaconed_trial_numbers)
        """
        trials, unique_trials = remake_trial_numbers(rewarded_beaconed_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, trials, '|', color='Black', markersize=2.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        probe_trials, unique_trials = remake_probe_trial_numbers(rewarded_probe_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, rewarded_beaconed_trial_numbers, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_probe_position_cm, rewarded_probe_trial_numbers, '|', color='Blue', markersize=2.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded_p.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        nonbeaconed_trials, unique_trials = remake_probe_trial_numbers(rewarded_nonbeaconed_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, rewarded_beaconed_trial_numbers, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_nonbeaconed_position_cm, rewarded_nonbeaconed_trial_numbers, '|', color='Blue', markersize=2.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded_nb.png', dpi=200)
        plt.close()
        """
        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        nonbeaconed_trials, unique_trials = remake_probe_trial_numbers(rewarded_nonbeaconed_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, rewarded_beaconed_trial_numbers, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_nonbeaconed_position_cm, rewarded_nonbeaconed_trial_numbers, '|', color='Red', markersize=2.5)
        ax.plot(rewarded_probe_position_cm, rewarded_probe_trial_numbers, '|', color='Blue', markersize=2.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded_all.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        nonbeaconed_trials, unique_trials = remake_probe_trial_numbers(rewarded_nonbeaconed_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, rewarded_beaconed_trial_numbers, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_nonbeaconed_position_cm, rewarded_nonbeaconed_trial_numbers, '|', color='Red', markersize=2.5)
        ax.plot(rewarded_probe_position_cm, rewarded_probe_trial_numbers, '|', color='Blue', markersize=2.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded_all.png', dpi=200)
        plt.close()



def plot_smoothed_firing_rate_maps_for_rewarded_trials(recording_folder, spike_data):
    spike_data["Avg_rewarded_FR_in_space_beaconed"] = ""
    spike_data["Avg_rewarded_FR_in_space_nonbeaconed"] = ""
    spike_data["Avg_rewarded_FR_in_space_probe"] = ""
    print('I am plotting smoothed firing rate maps for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, average_beaconed_sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data_for_rewarded_trials(spike_data, cluster)
        bins=np.arange(0,200,1)
        #bins=np.arange(0.5,199.5,1)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_probe_spike_rate, '-', color='Blue' , alpha=0.7)
        ax.fill_between(bins, avg_probe_spike_rate-average_beaconed_sd,avg_probe_spike_rate+average_beaconed_sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded_p.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_nonbeaconed_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_nonbeaconed_spike_rate-average_beaconed_sd,avg_nonbeaconed_spike_rate+average_beaconed_sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.locator_params(axis = 'x', nbins=3)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded_nb.png', dpi=200)
        plt.close()

        spike_data.at[cluster, 'Avg_rewarded_FR_in_space_beaconed'] = list(avg_beaconed_spike_rate)# add data to dataframe
        spike_data.at[cluster, 'Avg_rewarded_FR_in_space_nonbeaconed'] = list(avg_nonbeaconed_spike_rate)# add data to dataframe
        spike_data.at[cluster, 'Avg_rewarded_FR_in_space_probe'] = list(avg_probe_spike_rate)# add data to dataframe

    return spike_data



def plot_smoothed_firing_rate_maps_for_rewarded_trials_outbound(recording_folder, spike_data):

    print('I am plotting smoothed firing rate maps for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, average_beaconed_sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data_for_rewarded_trials(spike_data, cluster)
        bins=np.arange(0,200,1)
        #bins=np.arange(0.5,199.5,1)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(30,90)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['0', '30', '60'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded_out.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_probe_spike_rate, '-', color='Blue' , alpha=0.7)
        ax.fill_between(bins, avg_probe_spike_rate-average_beaconed_sd,avg_probe_spike_rate+average_beaconed_sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '30', '60'])
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(30,90)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded_p_out.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_nonbeaconed_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_nonbeaconed_spike_rate-average_beaconed_sd,avg_nonbeaconed_spike_rate+average_beaconed_sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(30,90)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.locator_params(axis = 'x', nbins=3)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.set_xticklabels(['0', '30', '60'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_rewarded_nb_out.png', dpi=200)
        plt.close()

    return spike_data


