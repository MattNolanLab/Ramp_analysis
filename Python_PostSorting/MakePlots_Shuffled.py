import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import numpy as np
import matplotlib.gridspec as gridspec
import Python_PostSorting.ExtractFiringData
import Python_PostSorting.ShuffleAnalysis

"""

## cluster spatial firing properties
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate plots
-> spike number versus location

"""



def extract_loc_and_trial_for_plots(spike_data, cluster):
    cluster_firings= Python_PostSorting.ExtractFiringData.extract_shuffled_firing_num_data(spike_data, cluster)
    locations = cluster_firings.loc[cluster_firings['firing_rate'] > 0, 'bins']
    trials = cluster_firings.loc[cluster_firings['firing_rate'] > 0, 'trial_number']
    return locations, trials


def plot_shuffled_spikes_on_track(recording_folder,spike_data, prm, prefix):
    print('plotting shuffled spike rasters...')
    save_path = recording_folder + '/Figures/shuffled_spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        x_max = max(np.array(spike_data.at[cluster, 'beaconed_trial_number']))+1
        spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        locations, trials = extract_loc_and_trial_for_plots(spike_data, cluster)
        #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=1)
        ax.plot(locations, trials, '|', color='Black', markersize=1.5)
        #ax.plot(spike_data.loc[cluster].nonbeaconed_position_cm, spike_data.loc[cluster].nonbeaconed_trial_number, '|', color='Red', markersize=1.5)
        #ax.plot(spike_data.loc[cluster].probe_position_cm, spike_data.loc[cluster].probe_trial_number, '|', color='Blue', markersize=1.5)

        plt.ylabel('Spikes on trials', fontsize=10, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=10, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(recording_folder + '/Figures/shuffled_spike_trajectories/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


def find_max_y_value(b, nb, p):
    nb_x_max = np.nanmax(np.array(np.nan_to_num(b), dtype=np.float16))
    b_x_max = np.nanmax(np.array(np.nan_to_num(nb), dtype=np.float16))
    p_x_max = np.nanmax(np.array(np.nan_to_num(p), dtype=np.float16))
    if b_x_max > nb_x_max and b_x_max > p_x_max:
        x_max = b_x_max
    elif nb_x_max > b_x_max and nb_x_max > p_x_max:
        x_max = nb_x_max
    elif p_x_max > b_x_max and p_x_max > nb_x_max:
        x_max = p_x_max
    else:
        x_max= b_x_max

    x_max = x_max+(x_max/10)
    return x_max


def plot_shuffled_firing_rate_maps(recording_folder, spike_data, prefix):
    print('I am plotting shuffled firing rate maps...')
    save_path = recording_folder + '/Figures/shuffled_spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        avg_spikes_on_track = plt.figure(figsize=(4,3))

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = Python_PostSorting.ExtractFiringData.extract_average_shuffled_firing_rate_data(spike_data, cluster)
        #avg_beaconed_spike_rate = PostSorting.ShuffleAnalysis.get_rolling_sum(avg_beaconed_spike_rate, 5)
        #avg_nonbeaconed_spike_rate = PostSorting.ShuffleAnalysis.get_rolling_sum(avg_nonbeaconed_spike_rate, 5)
        #avg_probe_spike_rate = PostSorting.ShuffleAnalysis.get_rolling_sum(avg_probe_spike_rate, 5)

        #avg_beaconed_spike_rate=np.array(np.nan_to_num(avg_beaconed_spike_rate), dtype=np.float16)
        #avg_nonbeaconed_spike_rate=np.array(np.nan_to_num(avg_nonbeaconed_spike_rate), dtype=np.float16)
        #avg_probe_spike_rate=np.array(np.nan_to_num(avg_probe_spike_rate), dtype=np.float16)

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=10, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=10, labelpad = 10)
        plt.xlim(0,200)
        #x_max = np.nanmax(avg_beaconed_spike_rate)

        x_max = find_max_y_value(avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(recording_folder + '/Figures/shuffled_spike_rate/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()

