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
    #spike_data = plot_rewarded_spikes_on_track(server_path,spike_data)
    #spike_data = calculate_spikes_on_track_relative_to_reward(spike_data)
    #spike_data = calculate_stops_on_track_relative_to_reward(spike_data)
    #spike_data = plot_spikes_on_track_relative_to_reward(server_path, spike_data)
    spike_data = calculate_positions_relative_to_reward(spike_data)
    spike_data = plot_positions_relative_to_reward(spike_data)
    spike_data = calculate_trial_by_trial_peaks(spike_data)
    #spike_data = plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = plot_heatmap_by_trial(spike_data, prm)
    #spike_data = rewarded_firing_rate_by_trial(spike_data, prm)

    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.calculate_positions_relative_to_reward_uncued(spike_data)
    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.plot_positions_relative_to_reward_uncued(spike_data)
    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.plot_heatmap_by_trial_uncued(spike_data, prm)
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
        #new_trial = unique_trial_numbers[np.isin(old_unique_trial_numbers,row)]
        new_trial_array = np.append(new_trial_array, new_trial )

    return new_trial_array


def plot_rewarded_spikes_on_track(recording_folder,spike_data):
    print('plotting spike rasters for rewarded trials...')
    save_path = recording_folder + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        spikes_on_track = plt.figure(figsize=(4,3.5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)

        data = split_stops_by_reward(stop_locations, stop_trials, rewarded_trials)

        spike_position_cm, spike_trial_numbers = split_spikes_by_reward(spike_data, cluster, rewarded_trials)
        spike_trial_numbers, old_unique_trial_numbers, unique_trial_numbers = remake_trial_numbers(spike_trial_numbers) # this is for the sake of plotting so it doesnt show large gaps where failed trials are

        stop_locations = data[data[:,1] > 0, 0]
        stop_trials = data[data[:,1] > 0, 1]
        new_rewarded_trials, old_unique_trial_numbers, unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are

        stop_trials = renumber_stop_trials(unique_trial_numbers, old_unique_trial_numbers, stop_trials)  # this is for the sake of plotting as above

        rewarded_trials = unique_trial_numbers[np.isin(old_unique_trial_numbers,rewarded_trials)]  # this is for the sake of plotting as above

        ax.plot(spike_position_cm, spike_trial_numbers, '|', color='Black', markersize=1.5)
        ax.plot(rewarded_locations[:len(rewarded_trials)], rewarded_trials, '>', color='Red', markersize=2)
        ax.plot(stop_locations, stop_trials-1, 'o', color='DodgerBlue', markersize=1.5)

        plt.ylabel('Spikes on trials', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
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

        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        style_vr_plot(ax, 0)
        plt.ylim(30, 50)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded.png', dpi=200)
        plt.close()
    return spike_data


def split_spikes_by_reward(spike_data, cluster, rewarded_trials):
    trials = np.array(spike_data.at[cluster, 'trial_number'])
    locations = np.array(spike_data.at[cluster, 'x_position_cm'])
    rewarded_position_cm = locations[np.isin(trials,rewarded_trials)]#take firing locations when on rewarded trials
    rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]#take firing trial numbers when on rewarded trials
    return rewarded_position_cm, rewarded_trial_numbers


def split_stops_by_reward(locations, trials, rewarded_trials):
    rewarded_position_cm = locations[np.isin(trials,rewarded_trials)] #take stop locations when on rewarded trials
    rewarded_trial_numbers = trials[np.isin(trials,rewarded_trials)]     #take stop trial numbers when on rewarded trials
    data=np.vstack((rewarded_position_cm, rewarded_trial_numbers))
    data=np.transpose(data)
    return data


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
        new_position_array = np.zeros((0, 0 )); new_position_array[:,:] = np.nan

        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]

                for rowcount, row in enumerate(trial_data+1):
                    new_position = row[0] - reward_bin
                    new_position_array = np.append(new_position_array, new_position)

        data_new = np.vstack((new_position_array, spike_trial_numbers[:len(new_position_array)]))
        data_new=data_new.transpose()
        spike_data.at[cluster, 'spike_trajectories_by_reward'] = data_new

    return spike_data


def calculate_stops_on_track_relative_to_reward(spike_data):
    spike_data["stop_trajectories_by_reward"] = ""

    for cluster in range(len(spike_data)):
        rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
        data = split_stops_by_reward(stop_locations, stop_trials, rewarded_trials)
        stop_trials = data[data[:,1] > 0, 1]

        # bin data over position bins
        trial_numbers = np.unique(stop_trials)
        new_position_array = np.zeros((0, 0)); new_position_array[:,:] = np.nan

        for trialcount, trial in enumerate(trial_numbers):
            if trial > 0.1:
                trial_data = data[data[:,1] == trial,:]
                reward_bin = rewarded_locations[np.isin(rewarded_trials,trial)]
                if len(reward_bin) > 1:
                    reward_bin = reward_bin[0]

                for rowcount, row in enumerate(trial_data+1):
                    new_position = row[0] - reward_bin
                    new_position_array = np.append(new_position_array, new_position)

        data_new = np.vstack((new_position_array, stop_trials[:len(new_position_array)]))
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
            spikes_on_track = plt.figure(figsize=(4,3.5))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            spike_trajectories = np.array(spike_data.at[cluster, 'spike_trajectories_by_reward'])
            stop_trajectories = np.array(spike_data.at[cluster, 'stop_trajectories_by_reward'])
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)

            spike_position = spike_trajectories[:,0]
            spike_trial_numbers = spike_trajectories[:,1]

            stop_locations = stop_trajectories[stop_trajectories[:,1] > 0, 0]
            stop_trials = stop_trajectories[stop_trajectories[:,1] > 0, 1]
            data = split_stops_by_reward(stop_locations, stop_trials, rewarded_trials)

            spike_trial_numbers, old_unique_trial_numbers, unique_trial_numbers = remake_trial_numbers(spike_trial_numbers) # this is for the sake of plotting so it doesnt show large gaps where failed trials are

            stop_locations = data[data[:,1] > 0, 0]
            stop_trials = data[data[:,1] > 0, 1]
            new_rewarded_trials, old_unique_trial_numbers, unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are
            stop_trials = renumber_stop_trials(unique_trial_numbers, old_unique_trial_numbers, stop_trials)  # this is for the sake of plotting as above

            ax.plot(spike_position, spike_trial_numbers, '|', color='Black', markersize=1.5)
            ax.plot(stop_locations, stop_trials-1, 'o', color='DodgerBlue', markersize=1.5)

            plt.ylabel('Spikes on trials', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            #plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            #plt.ylim(30, 50)
            plt.ylim(17.5, 24.5)
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

            plt.locator_params(axis = 'y', nbins  = 4)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_track_firing_Cluster_' + str(cluster_index +1) + '_rewarded2.png', dpi=200)
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
        window = signal.gaussian(2, std=2)
        #rates = signal.convolve(rates, window, mode='same')/ sum(window)

        # stack data
        data = np.vstack((rates,position, trials, speed, types))
        data=data.transpose()
        data = data[data[:,3] >= 3,:]
        data = data[data[:,4] == 0,:]
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
    spike_data["trial_peaks_max_reward"] = ""
    spike_data["trial_peak_locations_max_reward"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'FR_reset_at_reward_by_trial'])
        peak_array = []
        peak_array_locs = []
        for colcount, col in enumerate(cluster_data):
            trial_data = cluster_data[:,colcount]
            region_of_interest = trial_data[60:110]
            peak_location = np.argmax(region_of_interest)+60
            peak_firingrate = np.max(region_of_interest)*10
            peak_array = np.append(peak_array, peak_firingrate )
            peak_array_locs = np.append(peak_array_locs, peak_location )

        spike_data.at[cluster, 'trial_peaks_max_reward'] = list(peak_array)# add data to dataframe
        spike_data.at[cluster, 'trial_peak_locations_max_reward'] = list(peak_array_locs)# add data to dataframe

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

            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
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
