
import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.plot_utility



def calculate_stops_from_speed(spike_data):
    print("calculating stops from speed...")
    spike_data["stops_cm_position"] = ""
    spike_data["stops_cm_trial"] = ""
    spike_data["stops_cm_trial_type"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        data = np.vstack((speed,position,types, trials))
        data=data.transpose()

        stop_positions = []
        stop_trials = []
        stop_types = []

        for rowcount, row in enumerate(data):
            if data[rowcount,0] < 11.7 :
                stop_positions = np.append(stop_positions, data[rowcount,1])
                stop_trials = np.append(stop_trials, data[rowcount,3])
                stop_types = np.append(stop_types, data[rowcount,2])

        stop_positions, stop_trials, stop_types = remove_extra_stops(2, stop_positions, stop_trials, stop_types)

        spike_data.at[cluster, 'stops_cm_position'] = list(stop_positions)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_trial'] = list(stop_trials)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_trial_type'] = list(stop_types)# add data to dataframe

    return spike_data



def calculate_stop_time_from_speed(spike_data):
    print("calculating stops from speed...")
    spike_data["stops_cm_position"] = ""
    spike_data["stops_cm_trial"] = ""
    spike_data["stops_cm_trial_type"] = ""
    spike_data["stops_cm_time"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        time = np.arange(100,len(trials)*100,100)
        data = np.vstack((speed,position,types, trials, time))
        data=data.transpose()

        stop_positions = []
        stop_trials = []
        stop_types = []
        stop_times = []

        for rowcount, row in enumerate(data):
            if data[rowcount,0] < 1.7 :
                stop_positions = np.append(stop_positions, data[rowcount,1])
                stop_trials = np.append(stop_trials, data[rowcount,3])
                stop_types = np.append(stop_types, data[rowcount,2])
                stop_times = np.append(stop_times, data[rowcount,4])

        stop_positions, stop_trials, stop_types = remove_extra_stops_with_time(2, stop_positions, stop_trials, stop_types, stop_times)

        spike_data.at[cluster, 'stops_cm_position'] = list(stop_positions)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_trial'] = list(stop_trials)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_trial_type'] = list(stop_types)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_time'] = list(stop_times)# add data to dataframe

    return spike_data


def remake_times_on_trials(time, trials):
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

    return new_time_array


def calculate_stops_from_200ms_speed(spike_data):
    print("calculating stops from speed...")
    spike_data["stop_location_cm"] = ""
    spike_data["stop_trial_number"] = ""
    spike_data["stop_trial_types"] = ""
    spike_data["stops_cm_time"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        time = np.arange(100,(len(trials)*100)+1,100)

        time = remake_times_on_trials(time, trials)

        data = np.vstack((speed,position,types, trials, time))
        data=data.transpose()

        stop_positions = []
        stop_trials = []
        stop_types = []
        stop_times = []

        for rowcount, row in enumerate(range(len(data)-1)):
            speed_200ms = (data[rowcount,0] + data[rowcount+1,0])/2
            if speed_200ms <= 0.7:
                stop_positions = np.append(stop_positions, data[rowcount,1])
                stop_trials = np.append(stop_trials, data[rowcount,3])
                stop_types = np.append(stop_types, data[rowcount,2])
                stop_times = np.append(stop_times, data[rowcount,4])

        stop_positions, stop_trials, stop_types, stop_times = remove_extra_stops_with_time(2, stop_positions, stop_trials, stop_types, stop_times)

        spike_data.at[cluster, 'stop_location_cm'] = list(stop_positions)# add data to dataframe
        spike_data.at[cluster, 'stop_trial_number'] = list(stop_trials)# add data to dataframe
        spike_data.at[cluster, 'stop_trial_types'] = list(stop_types)# add data to dataframe
        spike_data.at[cluster, 'stops_cm_time'] = list(stop_times)# add data to dataframe

    return spike_data


def remove_extra_stops(min_distance, stops, trials, types):
    new_stops = []
    new_trials = []
    new_types = []
    for rowcount, row in enumerate(range(len(stops)-1)):
        current_stop = stops[rowcount]
        current_trial = trials[rowcount]
        current_type = types[rowcount]
        next_stop = stops[rowcount + 1]
        if (next_stop - current_stop) > min_distance:
            new_stops.append(current_stop)
            new_trials.append(current_trial)
            new_types.append(current_type)

    return new_stops, new_trials, new_types


def remove_extra_stops_with_time(min_distance, stops, trials, types, time):
    new_stops = []
    new_trials = []
    new_types = []
    new_times = []
    for rowcount, row in enumerate(range(len(stops)-1)):
        current_stop = stops[rowcount]
        current_trial = trials[rowcount]
        current_type = types[rowcount]
        current_time = time[rowcount]
        next_stop = stops[rowcount + 1]
        if (next_stop - current_stop) > min_distance:
            new_stops.append(current_stop)
            new_trials.append(current_trial)
            new_types.append(current_type)
            new_times.append(current_time)

    return new_stops, new_trials, new_types, new_times





def calculate_rewards_from_stops(spike_data):
    print("calculating rewards from stops...")
    spike_data["rewarded_locations"] = ""
    spike_data["rewarded_trials"] = ""
    spike_data["rewarded_times"] = ""

    for cluster in range(len(spike_data)):
        stops = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
        trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
        time = np.array(spike_data.at[cluster, 'stops_cm_time'], dtype=np.int16)

        data = np.vstack((stops, trials, time))
        data=data.transpose()

        reward_positions = []
        reward_trials = []
        reward_times = []

        if len(np.unique(trials)) > 1:
            # bin data over position bins
            trial_numbers = np.arange(min(trials),max(trials), 1)
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,1] == trial,:]
                if trial_data.shape[0] > 0:
                    for rowcount, row in enumerate(trial_data):
                        if trial_data[rowcount,0] >= 88 and trial_data[rowcount,0] <= 110:
                            reward_positions = np.append(reward_positions, trial_data[rowcount,0])
                            reward_trials = np.append(reward_trials, trial_data[rowcount,1])
                            reward_times = np.append(reward_times, trial_data[rowcount,2])
                            break

        spike_data.at[cluster, 'rewarded_locations'] = list(reward_positions)# add data to dataframe
        spike_data.at[cluster, 'rewarded_trials'] = list(reward_trials)# add data to dataframe
        spike_data.at[cluster, 'rewarded_times'] = list(reward_times)# add data to dataframe

    return spike_data



def plot_stops_on_track_per_cluster(spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stops_on_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(4,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        stop_locations = np.array(spike_data.at[cluster, "stops_cm_position"])
        stop_trials = np.array(spike_data.at[cluster, "stops_cm_trial"])
        stop_trial_types = np.array(spike_data.at[cluster, "stops_cm_trial_type"])

        stop_data = np.vstack((stop_locations,stop_trials, stop_trial_types))
        stop_data = np.transpose(stop_data)
        probe_stops = stop_data[stop_data[:,2] != 0,:]
        probe_stop_locations = probe_stops[:,0]
        probe_stop_trials = probe_stops[:,1]

        #load reward data to plot
        #reward_locations = np.array(spike_data.at[cluster, "reward_position"])
        #reward_trials = np.array(spike_data.at[cluster, "reward_trial"])

        ax.plot(stop_locations, stop_trials, 'o', color='0.2', markersize=2)
        ax.plot(probe_stop_locations, probe_stop_trials, 'o', color='Blue', markersize=2)
        #ax.plot(reward_locations, reward_trials, '>', color='Red', markersize=3)
        plt.ylabel('Trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.ylim(0,105)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_raster_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def get_bin_size(x_position_cm):
    track_length = x_position_cm.max()
    start_of_track = x_position_cm.min()
    number_of_bins = 200
    bin_size_cm = (track_length - start_of_track)/number_of_bins
    bins = np.arange(start_of_track,track_length, 200)
    return bin_size_cm,number_of_bins, bins



def calculate_average_stops(spike_data):
    print('I am calculating avg stops...')
    spike_data["average_stops"] = ""
    spike_data["position_bins"] = ""
    spike_data["average_stops_probe"] = ""

    for cluster in range(len(spike_data)):
        stop_locations = np.array(spike_data.at[cluster, "stops_cm_position"])
        stop_locations = stop_locations[~np.isnan(stop_locations)] #need to deal with
        stop_trials = np.array(spike_data.at[cluster, "stops_cm_trial"])
        stop_trial_types = np.array(spike_data.at[cluster, "stops_cm_trial_type"])
        position_cm=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)

        bin_size_cm,number_of_bins, bins = get_bin_size(position_cm)
        number_of_trials = len(np.unique(stop_trials))
        stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
        for loc in range(int(number_of_bins)-1):
            stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
            stops_in_bins[loc] = stops_in_bin


        stop_data = np.vstack((stop_locations,stop_trials, stop_trial_types))
        stop_data = np.transpose(stop_data)
        probe_stops = stop_data[stop_data[:,2] != 0,:]
        probe_stop_locations = probe_stops[:,0]
        probe_stop_trials = probe_stops[:,1]
        number_of_probe_trials = len(np.unique(probe_stop_trials))

        stops_in_bins_probe = np.zeros((len(range(int(number_of_bins)))))
        for loc in range(int(number_of_bins)-1):
            stops_in_bin = len(probe_stop_locations[np.where(np.logical_and(probe_stop_locations > (loc), probe_stop_locations <= (loc+1)))])/number_of_probe_trials
            stops_in_bins_probe[loc] = stops_in_bin

        window = signal.gaussian(3, std=2)
        stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
        stops_in_bins_probe = signal.convolve(stops_in_bins_probe, window, mode='same')/ sum(window)

        spike_data.at[cluster, 'average_stops'] = pd.Series(stops_in_bins)
        spike_data.at[cluster, 'position_bins'] = pd.Series(range(int(number_of_bins)))
        spike_data.at[cluster, 'average_stops_probe'] = pd.Series(stops_in_bins_probe)

    return spike_data


def plot_stop_histogram(recording_folder, spike_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/stop_histogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(4,2))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position_bins = np.array(spike_data.at[cluster, "position_bins"])
        average_stops = np.array(spike_data.at[cluster, "average_stops"])
        average_stops_probe = np.array(spike_data.at[cluster, "average_stops_probe"])
        ax.plot(position_bins,average_stops, '-', color='Black')
        ax.plot(position_bins,average_stops_probe, '-', color='Blue')
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        x_max = np.nanmax(average_stops)+0.01
        plt.ylim(0, x_max)
        plt.ylabel('Stops (cm/s)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_hist' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()
    return spike_data





def calculate_average_spikes(spike_data):
    print('I am calculating avg stops...')
    spike_data["average_spikes"] = ""
    spike_data["position_bins"] = ""
    spike_data["average_spikes_probe"] = ""

    for cluster in range(len(spike_data)):
        stop_locations = np.array(spike_data.at[cluster, "x_position_cm"])
        stop_locations = stop_locations[~np.isnan(stop_locations)] #need to deal with
        stop_trials = np.array(spike_data.at[cluster, "trial_number"])
        stop_trial_types = np.array(spike_data.at[cluster, "trial_type"])
        position_cm=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)

        bin_size_cm,number_of_bins, bins = get_bin_size(position_cm)
        number_of_trials = len(np.unique(stop_trials))
        stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
        for loc in range(int(number_of_bins)-1):
            stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
            stops_in_bins[loc] = stops_in_bin
        window = signal.gaussian(3, std=2)

        try:
            stop_data = np.vstack((stop_locations,stop_trials, stop_trial_types))
            stop_data = np.transpose(stop_data)
            probe_stops = stop_data[stop_data[:,2] != 0,:]
            probe_stop_locations = probe_stops[:,0]
            probe_stop_trials = probe_stops[:,1]
            number_of_probe_trials = len(np.unique(probe_stop_trials))
            if number_of_probe_trials == 0:
                number_of_probe_trials = 1

            stops_in_bins_probe = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = len(probe_stop_locations[np.where(np.logical_and(probe_stop_locations > (loc), probe_stop_locations <= (loc+1)))])/number_of_probe_trials
                stops_in_bins_probe[loc] = stops_in_bin
            stops_in_bins_probe = signal.convolve(stops_in_bins_probe, window, mode='same')/ sum(window)
            stops_in_bins_probe = signal.convolve(stops_in_bins_probe, window, mode='same')/ sum(window)
            spike_data.at[cluster, 'average_spikes_probe'] = pd.Series(stops_in_bins_probe)
        except ValueError or ZeroDivisionError:
            continue

        stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
        stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)

        spike_data.at[cluster, 'average_spikes'] = pd.Series(stops_in_bins)
        spike_data.at[cluster, 'position_bins'] = pd.Series(range(int(number_of_bins)))

    return spike_data




