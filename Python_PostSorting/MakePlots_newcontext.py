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




def load_stop_data(spatial_data):
    locations = np.array(spatial_data.at[1,'stop_location_cm'])
    trials = np.array(spatial_data.at[1,'stop_trial_number'])
    return locations,trials


def split_stop_data_by_trial_type(spatial_data):
    locations,trials = load_stop_data(spatial_data)
    stop_data=np.transpose(np.vstack((locations, trials)))
    return stop_data



def plot_stops_on_track(recording_folder, spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    stops_on_track = plt.figure(figsize=(4,3))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed = split_stop_data_by_trial_type(spike_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='black', markersize=2)
    ax.plot(spike_data.at[1,"rewarded_locations"], spike_data.at[1,"rewarded_trials"], '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    #x_max = max(raw_position_data.trial_number)+0.5
    Python_PostSorting.plot_utility.style_vr_plot(ax, 200,0)
    plt.ylim(0,90)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(recording_folder + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()



def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')/sum(window)
    #filtered_time = signal.convolve(time, window, mode='same')
    #convolved_rate = (filtered/filtered_time)
    return convolved_rate


def plot_spikes_on_track(recording_folder,spike_data, prm, prefix):
    print('plotting spike rasters...')
    save_path = recording_folder + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        x_max = spike_data.loc[cluster, 'max_trial_number']+1
        spikes_on_track = plt.figure(figsize=(4,3.5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster)

        ax.plot(beaconed_position_cm, beaconed_trial_number, '|', color='Black', markersize=2)
        ax.plot(nonbeaconed_position_cm, nonbeaconed_trial_number, '|', color='Red', markersize=2)
        ax.plot(probe_position_cm, probe_trial_number, '|', color='Blue', markersize=2)

        plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


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


def plot_smoothed_firing_rate_maps(recording_folder, spike_data, prefix):
    print('I am plotting smoothed firing rate maps...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.fill_between(np.arange(0,200,1), avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.fill_between(np.arange(0,200,1), avg_nonbeaconed_spike_rate-sd,avg_nonbeaconed_spike_rate+sd, facecolor = 'Red', alpha = 0.2)
        #ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        x_max = np.nanmax(avg_beaconed_spike_rate) +5
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.xlim(0,200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_smoothed_firing_rate_maps_per_trialtype(recording_folder, spike_data):
    print('I am plotting smoothed firing rate maps for trial types...')
    save_path = recording_folder + '/Figures/spike_rate_smoothed/per_trialtype'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    bins=np.arange(0,200,1)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)
        avg_beaconed_spike_rate = avg_beaconed_spike_rate/3
        avg_nonbeaconed_spike_rate = avg_nonbeaconed_spike_rate/3
        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '50', '100'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
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
            labelsize=14,
            length=5,
            width=1.5)  # labels along the bottom edge are off

        ax.axvline(20, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.xlim(20,100)
        plt.ylim(0)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1)+ '_beaconed.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '50', '100'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
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
            labelsize=14,
            length=5,
            width=1.5)  # labels along the bottom edge are off

        ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.ylim(0)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1)+ '_beaconed_wholetrack.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-sd,avg_beaconed_spike_rate+sd, facecolor = 'Black', alpha = 0.2)
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.fill_between(bins, avg_nonbeaconed_spike_rate-sd,avg_nonbeaconed_spike_rate+sd, facecolor = 'Red', alpha = 0.2)
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.xlim(20,100)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_nbeaconed.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        x_max = np.nanmax(avg_probe_spike_rate)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '_probe.png', dpi=200)
        plt.close()



