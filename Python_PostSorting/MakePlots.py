import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import numpy as np
import matplotlib.gridspec as gridspec
import Python_PostSorting.ExtractFiringData
import math
import Python_PostSorting.RewardFiring
import pandas as pd
from scipy import stats
from scipy import signal

"""

## cluster spatial firing properties
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate plots
-> spike number versus location

"""


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    convolved_rate = signal.convolve(rate, window, mode='same')/sum(window)
    return convolved_rate


def plot_beaconed_spikes_on_track(recording_folder,spike_data):
    print('plotting spike rasters...')
    save_path = recording_folder + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(4,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster)

        trials, unique_trials = remake_trial_numbers(beaconed_trial_number)

        ax.plot(beaconed_position_cm, trials, '|', color='Black', markersize=2)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0, 85)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def plot_beaconed_and_probe_spikes_on_track(recording_folder,spike_data):
    print('plotting spike rasters...')
    save_path = recording_folder + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(4,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster)

        ax.plot(beaconed_position_cm, beaconed_trial_number, '|', color='Black', markersize=2)
        ax.plot(nonbeaconed_position_cm, nonbeaconed_trial_number, '|', color='Blue', markersize=2)
        ax.plot(probe_position_cm, probe_trial_number, '|', color='Blue', markersize=1.5)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0, 95)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_probe.png', dpi=200)
        plt.close()



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
    print('I am plotting smoothed firing rate maps for trial types...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    bins=np.arange(0,200,1)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)
        #avg_beaconed_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_harry_average_firing_rate_data(spike_data, cluster)
        #avg_beaconed_spike_rate=np.array(spike_data.loc[cluster, 'fr_binned_in_space'])

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
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

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_probe_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_probe_spike_rate-sd,avg_probe_spike_rate+sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Firing rate (hz)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        x_max = np.nanmax(avg_beaconed_spike_rate)+5
        x_max_p = np.nanmax(avg_probe_spike_rate)+5
        if x_max_p > x_max:
            x_max = x_max_p
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.ylim(0, x_max)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_probe.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
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
        x_max = np.nanmax(avg_beaconed_spike_rate)+5
        x_max_p = np.nanmax(avg_nonbeaconed_spike_rate)+5
        if x_max_p > x_max:
            x_max = x_max_p
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_nbeaconed.png', dpi=200)
        plt.close()




"""

## cluster spatial firing properties on hit and miss trials
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate versus location plots
-> spike number versus location

the above seperated for hit and miss trials
hit trial = trial successfully rewarded
miss trial = trial not rewarded

"""


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
    save_path = recording_folder + '/Figures/rewarded_spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(4,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers = Python_PostSorting.RewardFiring.split_trials_by_reward(spike_data, cluster)

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

        spikes_on_track = plt.figure(figsize=(4,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        probe_trials, unique_trials = remake_probe_trial_numbers(rewarded_probe_trial_numbers)
        x_max = int(np.nanmax(trials))+1
        ax.plot(rewarded_beaconed_position_cm, trials, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_probe_position_cm, probe_trials, '|', color='Blue', markersize=2.5)
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


def plot_smoothed_firing_rate_maps_for_rewarded_trials(recording_folder, spike_data):
    print('I am plotting smoothed firing rate maps for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, average_beaconed_sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data_for_rewarded_trials(spike_data, cluster)
        bins=np.arange(0,200,1)
        avg_spikes_on_track = plt.figure(figsize=(4,3))
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

        bins=np.arange(0,200,1)
        avg_spikes_on_track = plt.figure(figsize=(4,3))
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

        bins=np.arange(0,200,1)
        avg_spikes_on_track = plt.figure(figsize=(4,3))
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




"""

## Instantaneous firing rate comparisons 

The following functions plots instantaneous firing rate aginast location and speed

Variables:
> distance
> speed
> firing rate

"""



def plot_instant_rates(recording_folder, spike_data):
    #print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates =  np.array(spike_data.loc[cluster_index].spike_rate_in_time[0])*10
        speed = np.array(spike_data.loc[cluster_index].spike_rate_in_time[1])
        position = np.array(spike_data.loc[cluster_index].spike_rate_in_time[2])
        rates, speed, position = remove_low_speeds(rates, speed, position )

        # plot speed versus rates
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(speed, rates, 'o', color='Black', markersize=1.5)
        plt.ylabel('Spike rate (hz)', fontsize=10, labelpad = 10)
        plt.xlabel('Speed (cm/s)', fontsize=10, labelpad = 10)
        x_max = np.nanmax(rates)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        #Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '.png', dpi=200)
        plt.close()

        # plot location versus rates
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position, rates, 'o', color='Black', markersize=1.5)
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.ylabel('Spike rate (hz)', fontsize=10, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=10, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        #Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '.png', dpi=200)
        plt.close()
    return spike_data


def remove_low_speeds(rates, speed, position ):
    data = np.vstack((rates, speed, position))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    rates = data_filtered[:,0]
    speed = data_filtered[:,1]
    position = data_filtered[:,2]
    return rates, speed, position


def plot_color_coded_instant_rates(recording_folder, spike_data):
    #print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0])*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        rates, speed, position = remove_low_speeds(rates, speed, position )

        try:
            window = signal.gaussian(5, std=2)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except ValueError:
                continue

        # plot speed versus rates
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        area = np.pi*1
        plt.scatter(speed, rates, s=1, c=position, cmap='BuPu_r')
        cbar = plt.colorbar(cmap='BuPu_r')
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Speed (cm/s)', fontsize=16, labelpad = 10)
        #x_max = np.nanmax(rates)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '_coded_new.png', dpi=200)
        plt.close()

        # plot location versus rates
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position, rates, s=1, c=speed)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_new.png', dpi=200)
        plt.close()
    return spike_data


def remove_low_speeds_and_segment_beaconed(rates, speed, position, types ):
    data = np.vstack((rates, speed, position, types))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    data_filtered = data_filtered[data_filtered[:,3] == 0,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,2] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    position_homebound = data_homebound[:,2]
    return rates_outbound , speed_outbound , position_outbound , rates_homebound, speed_homebound, position_homebound


def plot_color_coded_instant_rates_according_to_segment(recording_folder, spike_data):
    print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        #session_id = spike_data.at[cluster, "session_id"]
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real*10
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real
        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real

        # filter data
        try:
            window = signal.gaussian(2, std=3)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except ValueError:
                continue

        rates_o, speed_o, position_o,rates_h, speed_h, position_h = remove_low_speeds_and_segment_beaconed(rates, speed, position, types )

        # remove outliers
        rates =  pd.Series(rates_o)
        speed =  pd.Series(speed_o)
        position =  pd.Series(position_o)
        rates_o = rates[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        speed_o = speed[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        position_o = position[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        area = np.pi*1
        plt.scatter(speed_o, rates_o, s=1, c=position_o)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Speed (cm/s)', fontsize=16, labelpad = 10)
        #x_max = np.nanmax(rates)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        #ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        #ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #plt.text(0.7,0.7, "r-value" + str(round(r_value,2)) + "p-value" + str(round(p_value, 2)))
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '_coded_outbound.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        area = np.pi*1
        plt.scatter(speed_h, rates_h/3, s=area, c=position_h)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Speed (cm/s)', fontsize=16, labelpad = 10)
        x_max = np.nanmax(rates)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '_coded_homebound.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_o, rates_o, s=1, c=speed_o, cmap='BuPu_r') # jet
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(30, 90)
        ax.set_ylim(0)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_outbound.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_h, rates_h, s=1, c=speed_h, cmap='BuPu_r')
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(110, 170)
        ax.set_ylim(0)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_homebound.png', dpi=200)
        plt.close()
    return spike_data


def remove_low_speeds_and_segment_probe(rates, speed, position, types ):
    data = np.vstack((rates, speed, position, types))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    data_filtered = data_filtered[data_filtered[:,3] != 0,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,2] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    position_homebound = data_homebound[:,2]
    return rates_outbound , speed_outbound , position_outbound , rates_homebound, speed_homebound, position_homebound


def plot_color_coded_instant_rates_according_to_segment_probe(recording_folder, spike_data):
    print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real*10
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real
        #trials =  np.array(spike_data.iloc[cluster].spike_rate_in_time[3]).real
        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real
        # filter data
        try:
            window = signal.gaussian(5, std=2)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except (TypeError, ValueError):
                continue
        rates_o, speed_o, position_o,rates_h, speed_h, position_h = remove_low_speeds_and_segment_probe(rates, speed, position, types )
        area = np.pi*1
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_o, rates_o, s=area, c=speed_o, cmap='BuPu_r') # jet
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(30, 90)
        ax.set_ylim(0)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_outbound_probe.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_h, rates_h, s=area, c=speed_h, cmap='BuPu_r')
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(110, 170)
        ax.set_ylim(0)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_homebound_probe.png', dpi=200)
        plt.close()

    return spike_data



def remove_low_speeds_and_segment_nonbeaconed(rates, speed, position, types ):
    data = np.vstack((rates, speed, position, types))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    data_filtered = data_filtered[data_filtered[:,3] == 1,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,2] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    position_homebound = data_homebound[:,2]
    return rates_outbound , speed_outbound , position_outbound , rates_homebound, speed_homebound, position_homebound


def plot_color_coded_instant_rates_according_to_segment_nonbeaconed(recording_folder, spike_data):
    print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real*10
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real
        #trials =  np.array(spike_data.iloc[cluster].spike_rate_in_time[3]).real
        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real
        # filter data
        try:
            window = signal.gaussian(5, std=2)
            speed = signal.convolve(speed, window, mode='same')/ sum(window)/sum(window)
            rates = signal.convolve(rates, window, mode='same')/ sum(window)/sum(window)
        except (TypeError, ValueError):
                continue
        rates_o, speed_o, position_o,rates_h, speed_h, position_h = remove_low_speeds_and_segment_nonbeaconed(rates, speed, position, types )
        area = np.pi*1
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_o, rates_o, s=area, c=speed_o, cmap='BuPu_r') # jet
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(30, 90)
        ax.set_ylim(0)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_outbound_nonbeaconed.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_h, rates_h, s=area, c=speed_h, cmap='BuPu_r')
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(110, 170)
        ax.set_ylim(0)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_homebound_nonbeaconed.png', dpi=200)
        plt.close()
    return spike_data



def plot_color_trial_coded_instant_rates_according_to_segment(recording_folder, spike_data):
    print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0])*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3], dtype= np.int32)
        data=np.transpose(np.vstack((rates,speed, position, trials)))

        data_filtered = data[data[:,1] > 3,:]
        data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
        data_outbound = data_filtered[data_filtered[:,2] <= 90,:]

        rates_outbound = data_outbound[:,0]
        position_outbound = data_outbound[:,2]
        trials_outbound = data_outbound[:,3]

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        area = np.pi*1
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_outbound, rates_outbound, s=area, c=trials_outbound, cmap='BuPu_r') # jet
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlim(30, 90)
        ax.set_ylim(0)
        plt.ylabel('Firing rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_trial_coded_outbound.png', dpi=200)
        plt.close()

    return spike_data



### This plots raw data binned in time - a few trials as example data

def plot_tiny_raw_ind(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #load raw data binned in time
        rates =  np.array(spike_data.iloc[cluster].spikes_in_time[0])
        acceleration =  np.array(spike_data.iloc[cluster].spikes_in_time[3])
        speed =  np.array(spike_data.iloc[cluster].spikes_in_time[2])
        position=np.array(spike_data.iloc[cluster].spikes_in_time[1])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time[4], dtype= np.int32)

        ## stack and segment a few trials

        data = np.vstack((rates,acceleration, speed, position, trials))
        data = np.transpose(data)
        trial_data = data[350:750, :]

        # plot raw position
        avg_spikes_on_track = plt.figure(figsize=(7,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,3], '-', color='red', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=22,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_position.png', dpi=200)
        plt.close()

        # plot raw speed
        avg_spikes_on_track = plt.figure(figsize=(7,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,2], '-', color='Gold', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Speed (cm/s)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=22,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_speed.png', dpi=200)
        plt.close()


        # plot raw acceleration
        avg_spikes_on_track = plt.figure(figsize=(7,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,1], '-', color='Blue', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Acceleration (cm/s)2', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=22,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.set_xlim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_acceleration.png', dpi=200)
        plt.close()


        # plot raw spikes
        avg_spikes_on_track = plt.figure(figsize=(7,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_rates.png', dpi=200)
        plt.close()
    return spike_data


def plot_tiny_raw(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #load raw data binned in time
        rates =  np.array(spike_data.iloc[cluster].spikes_in_time[0])
        acceleration =  np.array(spike_data.iloc[cluster].spikes_in_time[3])
        speed =  np.array(spike_data.iloc[cluster].spikes_in_time[2])
        position=np.array(spike_data.iloc[cluster].spikes_in_time[1])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time[4], dtype= np.int32)

        ## stack and segment a few trials

        data = np.vstack((rates,acceleration, speed, position, trials))
        data = np.transpose(data)
        trial_data = data[350:750, :]

        # plot raw position
        avg_spikes_on_track = plt.figure(figsize=(15,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(4, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,3], '-', color='red', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

        # plot raw speed
        ax = avg_spikes_on_track.add_subplot(4, 1, 2)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,2], '-', color='Gold', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Speed (cm/s)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)


        # plot raw acceleration
        ax = avg_spikes_on_track.add_subplot(4, 1, 3)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,1], '-', color='Blue', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Acceleration (cm/s)2', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        #ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)

        # plot raw spikes
        ax = avg_spikes_on_track.add_subplot(4, 1, 4)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all.png', dpi=200)
        plt.close()
    return spike_data
