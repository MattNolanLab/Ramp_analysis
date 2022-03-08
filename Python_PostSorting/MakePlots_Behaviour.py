import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import numpy as np
import Python_PostSorting.ExtractFiringData
from scipy import signal



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



def remake_probe_trial_numbers(rewarded_beaconed_trial_numbers):
    unique_trials = np.unique(rewarded_beaconed_trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = rewarded_beaconed_trial_numbers[rewarded_beaconed_trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n +=5
    return new_trial_numbers, unique_trials


def calculate_average_stops(spike_data):
    spike_data["average_stops"] = ""
    spike_data["position_bins"] = ""
    for cluster in range(len(spike_data)):
        stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = len(np.unique(stop_trials)) # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            window = signal.gaussian(2, std=3)
            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)

            spike_data.at[cluster,'average_stops'] = list(stops_in_bins)
            spike_data.at[cluster,'position_bins'] = list(range(int(number_of_bins)))
        else:
            spike_data.at[cluster,'average_stops'] = np.nan
            spike_data.at[cluster,'position_bins'] = np.nan
    return spike_data


def calculate_average_nonbeaconed_stops(spike_data):
    spike_data["average_stops_nb"] = ""
    spike_data["position_bins_nb"] = ""
    spike_data["average_stops_p"] = ""
    spike_data["position_bins_p"] = ""
    for cluster in range(len(spike_data)):
        stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)


        stop_trial_types = calculate_stop_types(spike_data, cluster, stop_trials)
        beaconed,nonbeaconed,probe = split_stops_by_trial_type(stop_locations,stop_trials,stop_trial_types)

        stop_locations = nonbeaconed[:,0]
        stop_trials = nonbeaconed[:,1]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = len(np.unique(stop_trials)) # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            window = signal.gaussian(2, std=3)
            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)

            spike_data.at[cluster,'average_stops_nb'] = list(stops_in_bins)
            spike_data.at[cluster,'position_bins_nb'] = list(range(int(number_of_bins)))
        else:
            spike_data.at[cluster,'average_stops_nb'] = np.nan
            spike_data.at[cluster,'position_bins_nb'] = np.nan

        stop_locations = probe[:,0]
        stop_trials = probe[:,1]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = len(np.unique(stop_trials)) # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            window = signal.gaussian(2, std=3)
            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)

            spike_data.at[cluster,'average_stops_p'] = list(stops_in_bins)
            spike_data.at[cluster,'position_bins_p'] = list(range(int(number_of_bins)))
        else:
            spike_data.at[cluster,'average_stops_p'] = np.nan
            spike_data.at[cluster,'position_bins_p'] = np.nan

    return spike_data



def calculate_average_speed(spike_data):
    spike_data["average_speed"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)
        window = signal.gaussian(2, std=3)

        data = np.vstack((speed,position,types, trials))
        data=data.transpose()
        #data = data[data[:,0] > 3,:]
        data = data[data[:,2] == 0,:]

        stop_speed = data[:,0]
        stop_locations = data[:,1]
        stop_trials = data[:,3]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = np.nanmax(stop_trials) # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = np.nansum(stop_speed[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
            spike_data.at[cluster,'average_speed'] = list(stops_in_bins)
    return spike_data


def calculate_average_nonbeaconed_speed(spike_data):
    spike_data["average_speed_nb"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)
        window = signal.gaussian(2, std=3)

        data = np.vstack((speed,position,types, trials))
        data=data.transpose()
        #data = data[data[:,0] > 3,:]
        data = data[data[:,2] != 0,:]

        stop_speed = data[:,0]
        stop_locations = data[:,1]
        stop_trials = data[:,3]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = np.nanmax(stop_trials)/3 # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = np.nansum(stop_speed[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
            spike_data.at[cluster,'average_speed_nb'] = list(stops_in_bins)
    return spike_data



def plot_stop_histogram_per_cluster(spike_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stop_histogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(3.7,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        window = signal.gaussian(2, std=3)
        position_bins = np.array(spike_data.at[cluster, 'position_bins'])
        average_stops = np.array(spike_data.at[cluster, 'average_stops'])
        average_stops = signal.convolve(average_stops, window, mode='same')/ sum(window)
        ax.plot(position_bins,average_stops, '-', color='Black')
        average_stops_nb =  np.array(spike_data.at[cluster, 'average_stops_nb'])
        average_stops_nb = signal.convolve(average_stops_nb, window, mode='same')/ sum(window)
        ax.plot(position_bins,average_stops_nb, '-', color='Blue')
        plt.ylabel('Stops (cm)', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_raster_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def plot_speed_histogram(spike_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/speed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position_bins = np.arange(1,201,1)
        ax.plot(position_bins,np.array(spike_data.at[cluster, "average_speed"]), '-', color='Black')
        ax.plot(position_bins,np.array(spike_data.at[cluster, "average_speed_nb"]), '-', color='Blue')
        plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/speed_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def plot_speed(recording_folder, spike_data):
    print('I am plotting speed...')
    save_path = recording_folder + '/Figures/speed_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        position_bins = np.arange(0,200,1)
        avg_spikes_on_track = plt.figure(figsize=(4,3.5))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_bins,np.array(spike_data.at[cluster, "average_speed"]), '-', color='Black')

        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        ax.set_ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        ax.set_xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)

        x_max = np.nanmax(np.array(spike_data.at[cluster, "average_speed"]))
        plt.locator_params(axis = 'y', nbins = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def split_stops_by_trial_type(locations, trials, trial_type):
    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]==0),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed, nonbeaconed, probe


def calculate_stop_types(spike_data, cluster, stop_trials):
    stop_types = []
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
    data=np.transpose(np.vstack((trials, types)))

    for tcount, trial in enumerate(stop_trials):
        trial_data = data[data[:,0] == trial,:]
        try:
            type_in_trial = int(trial_data[0,1])
            stop_types = np.append(stop_types, type_in_trial)
        except IndexError:
            type_in_trial = 0
            stop_types = np.append(stop_types, type_in_trial)

    return stop_types


def split_stops_by_reward(beaconed,nonbeaconed,probe, rewarded_trials):
    #spike_data = add_columns_to_dataframe(spike_data)
    beaconed_position_cm = beaconed[:,0]
    beaconed_trial_number = beaconed[:,1]
    nonbeaconed_position_cm = nonbeaconed[:,0]
    nonbeaconed_trial_number = nonbeaconed[:,1]
    probe_position_cm = probe[:,0]
    probe_trial_number = probe[:,1]

    #take firing locations when on rewarded trials
    rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

    #take firing trial numbers when on rewarded trials
    rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]

    return rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers


def renumber_stop_trials_based_on_renumbered(unique_trial_numbers, old_unique_trial_numbers, stop_trials):
    new_trial_array = np.zeros((stop_trials.shape[0]))
    for rowcount, row in enumerate(stop_trials):
        current_trial = row
        new_trial = unique_trial_numbers[old_unique_trial_numbers[:] == current_trial]
        new_trial_array[rowcount] = new_trial
    return new_trial_array


def plot_stops_on_track_per_cluster(spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stops_on_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(3.7,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        try:
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_location_cm'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trial_number'], dtype=np.int16)
            #stop_trial_types = np.array(spike_data.at[cluster, "stop_trial_type"])
        except KeyError:
            rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
            rewarded_locations = np.array(spike_data.at[cluster, 'rewarded_locations'], dtype=np.int16)
            stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
            #stop_trial_types = np.array(spike_data.at[cluster, "stop_trial_type"])

        stop_locations = stop_locations[stop_trials != 0]
        stop_trials = stop_trials[stop_trials != 0]

        stop_trial_types = calculate_stop_types(spike_data, cluster, stop_trials)
        beaconed,nonbeaconed,probe = split_stops_by_trial_type(stop_locations,stop_trials,stop_trial_types)
        rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers = split_stops_by_reward(beaconed,nonbeaconed,probe, rewarded_trials)

        rewarded_trials, old_unique_trial_numbers, new_unique_trial_numbers = remake_trial_numbers(rewarded_trials) # this is for the sake of plotting so it doesnt show large gaps where failed trials are
        #stop_rewarded_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, stop_rewarded_trials)
        beaconed_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_beaconed_trial_numbers)
        nonbeaconed_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_nonbeaconed_trial_numbers)
        probe_trials = renumber_stop_trials_based_on_renumbered(new_unique_trial_numbers, old_unique_trial_numbers, rewarded_probe_trial_numbers)

        #beaconed_trials, unique_trials = remake_trial_numbers(rewarded_beaconed_trial_numbers)
        #nonbeaconed_trials, unique_trials = remake_probe_trial_numbers(rewarded_nonbeaconed_trial_numbers)
        #probe_trials, unique_trials = remake_probe_trial_numbers(rewarded_probe_trial_numbers)
        #rewarded_trials, unique_trials = remake_trial_numbers(rewarded_trials)


        ax.plot(rewarded_beaconed_position_cm, beaconed_trials, 'o', color='0.5', markersize=2)
        ax.plot(rewarded_nonbeaconed_position_cm, nonbeaconed_trials, 'o', color='blue', markersize=2)
        ax.plot(rewarded_probe_position_cm, probe_trials, 'o', color='blue', markersize=2)
        #ax.plot(rewarded_locations, rewarded_trials, '<', color='red', markersize=3)
        plt.ylabel('Stops on trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.ylim(0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_raster_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()
