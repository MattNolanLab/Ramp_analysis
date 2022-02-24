#firing rates plotted relative to first stop locations and / or relative to rewarded locations


import numpy as np
import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
from scipy import signal
import pandas as pd


"""

The following functions aim to recalculate position relative to reward position, and then plot spike trajectories against old and new location


"""

def run_reward_aligned_analysis(server_path,spike_data, prm):
    spike_data = calculate_spikes_on_track_relative_to_reward_time(spike_data)
    spike_data = plot_spikes_on_track_relative_to_reward_time(server_path, spike_data)

    #spike_data = plot_rewarded_spikes_on_track(server_path,spike_data)
    #spike_data = plot_rewarded_spikes_on_track_with_tt(server_path,spike_data)
    spike_data = calculate_spikes_on_track_relative_to_reward(spike_data)
    spike_data = calculate_stops_on_track_relative_to_reward(spike_data)
    #spike_data = plot_spikes_on_track_relative_to_reward(server_path, spike_data)
    #spike_data = calculate_positions_relative_to_reward(spike_data)
    #spike_data = plot_positions_relative_to_reward(spike_data)

    spike_data = calculate_spikes_per_trial(spike_data)
    spike_data = calculate_distance_from_reward(spike_data)
    spike_data = calculate_time_from_reward(spike_data)
    #spike_data = plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = plot_heatmap_by_trial(spike_data, prm)
    #spike_data = rewarded_firing_rate_by_trial(spike_data, prm)

    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.calculate_positions_relative_to_reward_uncued(spike_data)
    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.plot_positions_relative_to_reward_uncued(spike_data)
    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.plot_heatmap_by_trial_uncued(spike_data, prm)
    return spike_data


def remake_firing_times_to_per_trial(spike_data):
    print("remaking firing times based on trials...")
    spike_data["firing_times_per_trial"] = ""

    for cluster in range(len(spike_data)):
        time = np.array(spike_data.at[cluster, 'firing_times'])
        trials = np.array(spike_data.at[cluster, 'trial_number'])

        #ifnd the diff in trials
        data = np.vstack((time, trials))
        data=data.transpose()

        # bin data over position bins
        trial_numbers = np.unique(trials)
        new_time_array = np.zeros((0)); new_time_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,1] == trial,:]
            trial_start_time = trial_data[0,0]
            for rowcount, row in enumerate(trial_data+1):
                new_time = row[0] - trial_start_time
                new_time_array = np.append(new_time_array, new_time)

        data = np.vstack((new_time_array, trials))

        spike_data.at[cluster, 'firing_times_per_trial'] = new_time_array

    return spike_data


def calculate_spikes_per_trial(spike_data):
    spike_data["spikes_per_trial"] = ""
    for cluster in range(len(spike_data)):
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
        trials = np.array(spike_data.at[cluster, 'trial_number'])
        locations = np.array(spike_data.at[cluster, 'x_position_cm'])
        # split spikes by reward
        rewarded_position_cm = locations[np.isin(trials,rewarded_trials)]#take firing locations when on rewarded trials
        rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]#take firing trial numbers when on rewarded trials

        allspikes = np.nansum(rewarded_position_cm)
        trials = len(np.unique(rewarded_trial_numbers))
        spikes_on_trial = allspikes/trials

        spike_data.at[cluster, 'spikes_per_trial'] = spikes_on_trial
    return spike_data


def remake_trial_numbers(trial_numbers):
    unique_trials = np.unique(trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = trial_numbers[trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n += 1
    return new_trial_numbers, unique_trials, np.unique(new_trial_numbers)


def remake_all_trial_numbers(trial_numbers, stop_trial_numbers):
    unique_trials = np.unique(trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = trial_numbers[trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n += 1
    return new_trial_numbers, unique_trials, np.unique(new_trial_numbers)


def style_vr_plot(ax, hor):
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
    ax.axhline(hor, linewidth = 2.5, color = 'black') # bold line on the x axis
    return ax


def renumber_stop_trials(unique_trial_numbers, old_unique_trial_numbers, stop_trials):
    new_trial_array = []
    for rowcount, row in enumerate(stop_trials):
        new_trial = unique_trial_numbers[old_unique_trial_numbers[:] == row]
        new_trial_array = np.append(new_trial_array, new_trial )
    return new_trial_array


def renumber_stop_trials_based_on_renumbered(unique_trial_numbers, old_unique_trial_numbers, stop_trials):
    new_trial_array = np.zeros((stop_trials.shape[0]))
    for rowcount, row in enumerate(stop_trials):
        current_trial = row
        new_trial = unique_trial_numbers[old_unique_trial_numbers[:] == current_trial]
        new_trial_array[rowcount] = new_trial
    return new_trial_array


def plot_rewarded_spikes_on_track(recording_folder,spike_data):
    print('plotting spike rasters for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_trajectories_with_reward'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        try:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            #stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
            #stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
        except KeyError:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_position'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            #stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
            #stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
        rewarded_locations = rewarded_locations[~np.isnan(rewarded_locations)]


        # split spikes by reward
        spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)

        # split stops by reward
        #stop_rewarded_trials = stop_trials[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
        #stop_locations = stop_locations[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
        rewarded_locations = rewarded_locations[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials
        rewarded_trials = rewarded_trials[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials

        #renumber rewarded trials
        rewarded_trials, old_unique_trial_numbers, new_unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are
        #stop_rewarded_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, stop_rewarded_trials)
        spike_trial_numbers = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, spike_trial_numbers)


        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=2)
        #ax.plot(stop_locations, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1.5, alpha=0.5)
        ax.plot(spike_position_cm, spike_trial_numbers, '|', color='Black', markersize=2.5)
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
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_normal.png', dpi=200)
        plt.close()


        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=2)
        #ax.plot(stop_locations, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1, alpha=0.5)
        ax.plot(spike_position_cm, spike_trial_numbers, '|', color='Black', markersize=2.5)
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
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_normal_closeup.png', dpi=200)
        plt.close()

    return spike_data


def plot_rewarded_spikes_on_track_with_tt(recording_folder,spike_data):
    print('plotting spike rasters for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_trajectories_with_reward'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        try:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
        except KeyError:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
        rewarded_locations = rewarded_locations[~np.isnan(rewarded_locations)]

        spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)

        # split spikes by reward
        #spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)
        rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers = Python_PostSorting.RewardFiring.split_trials_by_reward(spike_data, cluster)

        # split stops by reward
        #stop_rewarded_trials = stop_trials[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
        #stop_locations = stop_locations[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
        rewarded_locations = rewarded_locations[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials
        rewarded_trials = rewarded_trials[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials

        #renumber rewarded trials
        rewarded_trials, old_unique_trial_numbers, new_unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are
        #stop_rewarded_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, stop_rewarded_trials)
        spike_trial_numbers = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_beaconed_trial_numbers)
        spike_trial_numbers_nb = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_nonbeaconed_trial_numbers)
        spike_trial_numbers_p = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_probe_trial_numbers)


        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=2)
        #ax.plot(stop_locations, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1.5, alpha=0.5)
        ax.plot(rewarded_beaconed_position_cm, spike_trial_numbers, '|', color='Black', markersize=2.5)
        ax.plot(rewarded_nonbeaconed_position_cm, spike_trial_numbers_nb, '|', color='red', markersize=2.5)
        ax.plot(rewarded_probe_position_cm, (spike_trial_numbers_p), '|', color='red', markersize=2.5)
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
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_normal_tt.png', dpi=200)
        plt.close()

    return spike_data


def split_spikes_by_reward(spike_data, cluster, rewarded_trials):
    trials = np.array(spike_data.at[cluster, 'trial_number'])
    locations = np.array(spike_data.at[cluster, 'x_position_cm'])
    rewarded_position_cm = locations[np.isin(trials,rewarded_trials)]#take firing locations when on rewarded trials
    rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]#take firing trial numbers when on rewarded trials
    return rewarded_position_cm, rewarded_trial_numbers


def split_spikes_by_reward_with_time(spike_data, cluster, rewarded_trials):
    time = np.array(spike_data.at[cluster, 'firing_times_per_trial'])
    trials = np.array(spike_data.at[cluster, 'trial_number'])
    locations = np.array(spike_data.at[cluster, 'x_position_cm'])
    rewarded_position_cm = locations[np.isin(trials,rewarded_trials)]#take firing locations when on rewarded trials
    rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]#take firing trial numbers when on rewarded trials
    rewarded_time = time[np.isin(trials,rewarded_trials)]#take firing locations when on rewarded trials
    return rewarded_position_cm, rewarded_trial_numbers, rewarded_time



def split_stops_by_reward(locations, trials, rewarded_trials):
    rewarded_position_cm = locations[np.isin(trials,rewarded_trials)] #take stop locations when on rewarded trials
    rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]     #take stop trial numbers when on rewarded trials
    data=np.vstack((rewarded_position_cm, rewarded_trial_numbers))
    data=np.transpose(data)
    return data



def calculate_distance_from_reward(spike_data):
    spike_data["distance_to_reward"] = ""

    for cluster in range(len(spike_data)):
        rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)

        # stack data
        data = np.vstack((spike_position_cm, spike_trial_numbers))
        data=data.transpose()

        # bin data over position bins
        trial_numbers = np.unique(spike_trial_numbers)
        new_position_array = np.zeros((0)); new_position_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]
                    upper_reward_area = reward_bin + 4
                    lower_reward_area = reward_bin - 4

                    for rowcount, row in enumerate(trial_data+1):
                        if row[0] > lower_reward_area and row[0] < upper_reward_area:
                            new_position = row[0] - reward_bin
                            new_position_array = np.append(new_position_array, new_position)

        spike_data.at[cluster, 'distance_to_reward'] = new_position_array

    return spike_data



def calculate_time_from_reward(spike_data):
    spike_data["time_to_reward"] = ""

    for cluster in range(len(spike_data)):
        rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        rewarded_times = np.array(spike_data.at[cluster, 'rewarded_times'], dtype=np.int16)
        spike_position_cm, spike_trial_numbers, spike_time = split_spikes_by_reward_with_time(spike_data, cluster, rewarded_trials)

        # stack data
        data = np.vstack((spike_position_cm, spike_trial_numbers, spike_time))
        data=data.transpose()

        # bin data over position bins
        trial_numbers = np.unique(spike_trial_numbers)
        new_position_array = np.zeros((0)); new_position_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_time = rewarded_times[np.isin(rewarded_trials,trial)]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_time) > 1:
                    reward_time = reward_time[0]
                    reward_bin = reward_bin[0]
                    upper_reward_area = reward_bin + 4
                    lower_reward_area = reward_bin - 4

                    for rowcount, row in enumerate(trial_data+1):
                        if row[0] > lower_reward_area and row[0] < upper_reward_area:
                            new_position = row[2] - reward_time
                            new_position_array = np.append(new_position_array, new_position)

        spike_data.at[cluster, 'time_to_reward'] = new_position_array
    return spike_data


def calculate_spikes_on_track_relative_to_reward(spike_data):
    spike_data["spike_trajectories_by_reward"] = ""

    for cluster in range(len(spike_data)):
        rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)

        # stack data
        data = np.vstack((spike_position_cm, spike_trial_numbers))
        data=data.transpose()

        # bin data over position bins
        trial_numbers = np.unique(spike_trial_numbers)
        new_position_array = np.zeros((0)); new_position_array[:] = np.nan
        new_trial_array = np.zeros((0)); new_trial_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]

                    for rowcount, row in enumerate(trial_data+1):
                        new_position = row[0] - reward_bin
                        new_position_array = np.append(new_position_array, new_position)
                        new_trial_array = np.append(new_trial_array, trial)

        data_new = np.vstack((new_position_array, new_trial_array))
        data_new=data_new.transpose()
        spike_data.at[cluster, 'spike_trajectories_by_reward'] = data_new

    return spike_data



def calculate_spikes_on_track_relative_to_reward_time(spike_data):
    spike_data["spike_trajectories_by_reward_time"] = ""

    for cluster in range(len(spike_data)):
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        rewarded_times = np.array(spike_data.at[cluster, 'rewarded_times'], dtype=np.int16)
        spike_position_cm, spike_trial_numbers, spike_times = split_spikes_by_reward_with_time(spike_data, cluster, rewarded_trials)

        # stack data
        data = np.vstack((spike_position_cm, spike_trial_numbers, spike_times))
        data=data.transpose()

        # bin data over position bins
        trial_numbers = np.unique(spike_trial_numbers)
        new_position_array = np.zeros((0)); new_position_array[:] = np.nan
        new_trial_array = np.zeros((0)); new_trial_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_times[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]

                for rowcount, row in enumerate(trial_data+1):
                    new_position = row[2] - reward_bin
                    new_position_array = np.append(new_position_array, new_position)
                    new_trial_array = np.append(new_trial_array, trial)

        data_new = np.vstack((new_position_array, new_trial_array))
        data_new=data_new.transpose()
        spike_data.at[cluster, 'spike_trajectories_by_reward_time'] = data_new

    return spike_data


def calculate_stops_on_track_relative_to_reward(spike_data):
    spike_data["stop_trajectories_by_reward"] = ""

    for cluster in range(len(spike_data)):
        try:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
        except KeyError:
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
        data = split_stops_by_reward(stop_locations, stop_trials, rewarded_trials)
        stop_trials = data[data[:,1] > 0, 1]

        # bin data over position bins
        trial_numbers = np.unique(stop_trials)
        new_position_array = np.zeros((0)); new_position_array[:] = np.nan
        new_trial_array = np.zeros((0)); new_trial_array[:] = np.nan
        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]

                for rowcount, row in enumerate(trial_data+1):
                    new_position = row[0] - reward_bin
                    new_position_array = np.append(new_position_array, new_position)
                    new_trial_array = np.append(new_trial_array, trial)

        data_new = np.vstack((new_position_array, new_trial_array))
        data_new=data_new.transpose()

        spike_data.at[cluster, 'stop_trajectories_by_reward'] = data_new

    return spike_data



def plot_spikes_on_track_relative_to_reward(recording_folder,spike_data):
    print('plotting spike rasters relative to reward...')
    save_path = recording_folder + '/Figures/spike_trajectories_with_reward'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        try:
            spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            spike_trajectories = np.array(spike_data.at[cluster, 'spike_trajectories_by_reward'])
            stop_trajectories = np.array(spike_data.at[cluster, 'stop_trajectories_by_reward'])
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)

            spike_position = spike_trajectories[:,0]
            spike_trial_numbers = spike_trajectories[:,1]

            stop_locations = stop_trajectories[stop_trajectories[:,1] > 0, 0]
            stop_trials = stop_trajectories[stop_trajectories[:,1] > 0, 1]

            # split stops by reward
            stop_rewarded_trials = stop_trials[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
            stop_locations = stop_locations[np.isin(stop_trials,spike_trial_numbers)]  # take all stops on spike trials
            rewarded_trials = rewarded_trials[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials

            #renumber rewarded trials
            rewarded_trials, old_unique_trial_numbers, new_unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are

            stop_rewarded_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, stop_rewarded_trials)
            spike_trial_numbers = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, spike_trial_numbers)

            #ax.plot(stop_locations, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1, alpha=0.5)
            ax.plot(spike_position, spike_trial_numbers, '|', color='Black', markersize=2.5)
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylabel('Trials', fontsize=18, labelpad = 0)
            plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
            plt.xlim(0,200)
            #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
            Python_PostSorting.plot_utility.style_vr_plot(ax)
            plt.ylim(0)
            plt.xlim(-100,100)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_reward.png', dpi=200)
            plt.close()


            spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(spike_position, spike_trial_numbers, '|', color='Black', markersize=2.5)
            ax.plot(stop_locations-0.5, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1.5, alpha=0.5)
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylabel('Spikes on trials', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylim(30, 50)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
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
            plt.xlim(-20,20)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_reward_closeup_stops.png', dpi=200)
            plt.close()

        except IndexError:
            continue

    return spike_data




def plot_spikes_on_track_relative_to_reward_time(recording_folder,spike_data):
    print('plotting spike rasters relative to reward...')
    save_path = recording_folder + '/Figures/spike_trajectories_with_reward'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        try:
            spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            spike_trajectories = np.array(spike_data.at[cluster, 'spike_trajectories_by_reward_time'])
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)

            spike_position = spike_trajectories[:,0]
            spike_trial_numbers = spike_trajectories[:,1]

            rewarded_trials = rewarded_trials[np.isin(rewarded_trials,spike_trial_numbers)]  # take all stops on spike trials

            #renumber rewarded trials
            rewarded_trials, old_unique_trial_numbers, new_unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are

            #stop_rewarded_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, stop_rewarded_trials)
            spike_trial_numbers = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, spike_trial_numbers)

            #ax.plot(stop_locations, stop_rewarded_trials, 'o', color='DodgerBlue', markersize=1, alpha=0.5)
            ax.plot(spike_position/1000, spike_trial_numbers, '|', color='Black', markersize=2.5)
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylabel('Trials', fontsize=18, labelpad = 0)
            plt.xlabel('Time (Sec)', fontsize=18, labelpad = 10)
            #plt.xlim(-2000,2000)
            #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
            #Python_PostSorting.plot_utility.style_vr_plot(ax)
            plt.ylim(0)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
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
            #plt.xlim(-100,100)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_reward_time.png', dpi=200)
            plt.close()


            spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(spike_position/1000, spike_trial_numbers, '|', color='Black', markersize=2.5)
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylabel('Trials', fontsize=18, labelpad = 0)
            plt.xlabel('Time (Sec)', fontsize=18, labelpad = 10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.axvline(0, linewidth = 1.5, color = 'red', alpha=0.5) # bold line on the y axis
            plt.ylim(30, 50)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
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
            #plt.xlim(-2000,2000)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_reward_closeup_time.png', dpi=200)
            plt.close()

        except IndexError:
            continue

    return spike_data




"""


The following functions aim to recalculate position relative to reward position, and then plot firing rates against old and new location


"""



def calculate_positions_relative_to_reward(spike_data):
    print("I am recalculating positions...")
    spike_data["spikes_in_time_reset_at_rewarded"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0])
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4], dtype= np.int32)
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
        rewarded_locations = np.array(spike_data.loc[cluster, 'rewarded_locations'])

        # stack data
        data = np.vstack((rates,position, trials, speed, types))
        data=data.transpose()
        data = data[data[:,3] >= 3,:]
        #data = data[data[:,4] == 0,:]
        rates = data[:,0]
        trials = data[:,2]

        data_new = np.zeros((1,3))
        if data.shape[0] > 0:
            # bin data over position bins
            #trial_numbers = np.arange(min(trials),max(trials), 1)
            trial_numbers = np.unique(data[:,2])
            new_position_array = np.zeros((0, 0 )); new_position_array[:,:] = np.nan

            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,2] == trial,:]
                reward_location = rewarded_locations[np.isin(rewarded_trials,trial)]

                for rowcount, row in enumerate(trial_data):
                    new_position = row[1] - reward_location
                    new_position_array = np.append(new_position_array, new_position)

            data_new = np.vstack((rates,new_position_array, trials))
            data_new=data_new.transpose()

        spike_data.at[cluster, 'spikes_in_time_reset_at_rewarded'] = pd.DataFrame(data_new)
    return spike_data



def calculate_positions_relative_to_reward_uncued(spike_data):
    print("I am recalculating positions...")
    spike_data["spikes_in_time_reset_at_rewarded_uncued"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0])
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4], dtype= np.int32)
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
        rewarded_locations = np.array(spike_data.loc[cluster, 'rewarded_locations'])
        window = signal.gaussian(2, std=2)
        #rates = signal.convolve(rates, window, mode='same')/ sum(window)

        # stack data
        data = np.vstack((rates,position, trials, speed, types))
        data=data.transpose()
        data = data[data[:,3] >= 3,:]
        data = data[data[:,4] != 0,:]
        rates = data[:,0]
        trials = data[:,2]

        data_new = np.zeros((1,3))
        if data.shape[0] > 0:
            # bin data over position bins
            #trial_numbers = np.arange(min(trials),max(trials), 1)
            trial_numbers = np.unique(data[:,2])
            new_position_array = np.zeros((0, 0 )); new_position_array[:,:] = np.nan

            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,2] == trial,:]
                reward_location = rewarded_locations[np.isin(rewarded_trials,trial)]

                for rowcount, row in enumerate(trial_data):
                    new_position = row[1] - reward_location
                    new_position_array = np.append(new_position_array, new_position)

            data_new = np.vstack((rates,new_position_array, trials))
            data_new=data_new.transpose()

        spike_data.at[cluster, 'spikes_in_time_reset_at_rewarded_uncued'] = pd.DataFrame(data_new)
    return spike_data





def plot_positions_relative_to_reward(spike_data):
    print("I am binning based on recalculated positions...")
    spike_data["FR_reset_at_reward_by_trial"] = ""
    spike_data["FR_reset_at_reward"] = ""
    spike_data["FR_sd_reset_at_reward"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'spikes_in_time_reset_at_rewarded'])

        # bin data over position bins
        bins = np.arange(-100,100,1)

        if (cluster_data.size) > 3:
            trial_numbers = np.unique(cluster_data[:,2])
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = cluster_data[cluster_data[:,2] == trial,:]
                if trial_data.shape[0] > 0:
                    trial_rates = trial_data[:,0]
                    trial_positions = trial_data[:,1]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(trial_rates, np.where(np.logical_and(trial_positions >= b, trial_positions < b+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            window = signal.gaussian(2, std=3)
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            x = np.nanmean(data_b, axis=1)
            #x = signal.convolve(x, window, mode='same')/ sum(window)
            x_sd = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'FR_reset_at_reward'] = list(x)# add data to dataframe
            spike_data.at[cluster, 'FR_sd_reset_at_reward'] = list(x_sd)
            spike_data.at[cluster, 'FR_reset_at_reward_by_trial'] = list(data_b)

    return spike_data


def calculate_trial_by_trial_peaks(spike_data):
    print("I am calculating trial by trial peaks...")
    spike_data["trial_peaks_max_reward"] = ""
    spike_data["trial_peak_locations_max_reward"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial'])
        peak_array = []
        peak_array_locs = []
        trials = (cluster_data.shape[1])-1
        for trialcount, trial in enumerate(range(trials)):
            trial_data = cluster_data[:,trialcount]
            region_of_interest = trial_data[65:110]
            peak_location = np.argmax(region_of_interest)+65
            peak_firingrate = np.max(region_of_interest)
            peak_array = np.append(peak_array, peak_firingrate )
            peak_array_locs = np.append(peak_array_locs, peak_location )

        spike_data.at[cluster, 'trial_peaks_max_reward'] = list(peak_array)# add data to dataframe
        spike_data.at[cluster, 'trial_peak_locations_max_reward'] = list(peak_array_locs)# add data to dataframe

    return spike_data


def calculate_trial_by_trial_minpeaks(spike_data):
    print("I am calculating trial by trial peaks...")
    spike_data["trial_peaks_min_reward"] = ""
    spike_data["trial_peak_locations_min_reward"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial'])
        peak_array = []
        peak_array_locs = []
        trials = (cluster_data.shape[1])-1
        for trialcount, trial in enumerate(range(trials)):
            trial_data = cluster_data[:,trialcount]
            region_of_interest = trial_data[65:110]
            peak_location = np.argmin(region_of_interest)+65
            peak_firingrate = np.min(region_of_interest)
            peak_array = np.append(peak_array, peak_firingrate )
            peak_array_locs = np.append(peak_array_locs, peak_location )

        spike_data.at[cluster, 'trial_peaks_min_reward'] = list(peak_array)# add data to dataframe
        spike_data.at[cluster, 'trial_peak_locations_min_reward'] = list(peak_array_locs)# add data to dataframe

    return spike_data


def plot_positions_relative_to_reward_uncued(spike_data):
    print("I am binning based on recalculated positions...")
    spike_data["FR_reset_at_reward_by_trial_uncued"] = ""


    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'spikes_in_time_reset_at_rewarded_uncued'])

        # bin data over position bins
        bins = np.arange(-100,100,1)

        if (cluster_data.size) > 3:
            trial_numbers = np.unique(cluster_data[:,2])
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = cluster_data[cluster_data[:,2] == trial,:]
                if trial_data.shape[0] > 0:
                    trial_rates = trial_data[:,0]
                    trial_positions = trial_data[:,1]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(trial_rates, np.where(np.logical_and(trial_positions >= b, trial_positions < b+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            window = signal.gaussian(2, std=2)
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            #spike_data.at[cluster, 'FR_reset_at_reward'] = list(x)# add data to dataframe
            #spike_data.at[cluster, 'FR_sd_reset_at_reward'] = list(x_sd)
            spike_data.at[cluster, 'FR_reset_at_reward_by_trial_uncued'] = list(data_b)

    return spike_data


def plot_rewarded_firing_rate(spike_data, prm):
    print("I am plotting firing rate relative to reward...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps_reset_at_reward'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(-100,100,1)
        rates=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        sd_rates=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        if rates.size > 1 :
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

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

            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            #ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)
            #ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
            plt.close()
    return spike_data





def plot_rewarded_firing_rate_by_trial(spike_data, prm):
    print("I am plotting firing rate relative to reward for some trials...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps_reset_at_reward_by_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(-100,100,1)
        rates=np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial'])
        #sd_rates=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        try:
            trial1 = rates[:,17]
            trial2 = rates[:,18]
            trial3 = rates[:,19]
            trial4 = rates[:,20]
            trial5 = rates[:,21]
            trial6 = rates[:,22]
        except IndexError:
            trial1 = rates[:,6]
            trial2 = rates[:,7]
            trial3 = rates[:,8]
            trial4 = rates[:,9]
            trial5 = rates[:,4]
            trial6 = rates[:,5]

        if rates.size > 1 :
            speed_histogram = plt.figure(figsize=(4,12))
            ax = speed_histogram.add_subplot(6, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial1, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            ax = speed_histogram.add_subplot(6, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial2, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            ax = speed_histogram.add_subplot(6, 1, 3)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial3, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            ax = speed_histogram.add_subplot(6, 1, 4)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial4, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            ax = speed_histogram.add_subplot(6, 1, 5)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial5, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            ax = speed_histogram.add_subplot(6, 1, 6)  # specify (nrows, ncols, axnum)
            ax.plot(position_array, trial6, '-', color='Black')
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            #plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(-100,100)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
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
            ax.axvline(0, linewidth = 1.5, color = 'black', alpha=0.5) # bold line on the y axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
            plt.close()
    return spike_data



import seaborn as sns




def plot_heatmap_by_trial(spike_data, prm):
    print("I am plotting firing rate relative to reward for all trials as a heatmap...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/heatmaps_per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates=np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial'])

        speed_histogram = plt.figure(figsize=(5,12))
        ax = sns.heatmap(np.transpose(rates))
        plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        plt.xlim(0,200)
        ax.axvline(100, linewidth = 1.5, color = 'black') # bold line on the y axis
        plt.locator_params(axis = 'x', nbins  = 4)
        plt.savefig(save_path + '/heatmap_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
        plt.close()

    return spike_data


def plot_heatmap_by_trial_uncued(spike_data, prm):
    print("I am plotting firing rate relative to reward for all trials as a heatmap...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/heatmaps_per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates=np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial_uncued'])

        try:
            speed_histogram = plt.figure(figsize=(5,12))
            ax = sns.heatmap(np.transpose(rates))
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.axvline(100, linewidth = 1.5, color = 'black') # bold line on the y axis
            plt.locator_params(axis = 'x', nbins  = 4)
            plt.savefig(save_path + '/heatmap_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_uncued.png', dpi=200)
            plt.close()
        except ValueError:
            continue

    return spike_data
