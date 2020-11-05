import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import Python_PostSorting.Speed_Analysis
import numpy as np
import matplotlib.gridspec as gridspec
import Python_PostSorting.ExtractFiringData
import math


def load_stop_data(spatial_data):
    locations = np.array(spatial_data.at[1,'stop_location_cm'])
    trials = np.array(spatial_data.at[1,'stop_trial_number'])
    trial_type = np.array(spatial_data.at[1,'stop_trial_type'])
    return locations,trials,trial_type


def split_stop_data_by_trial_type(spatial_data):
    locations,trials,trial_type = load_stop_data(spatial_data)
    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]!=1),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed[:,1], nonbeaconed[:,1], probe[:,1], beaconed[:,0], nonbeaconed[:,0], probe[:,0]



def plot_stops_on_track(recording_folder, spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    stops_on_track = plt.figure(figsize=(4,3))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(spike_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='black', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='black', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='black', markersize=2)
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


def split_stops_by_reward(spike_data):
    #spike_data = add_columns_to_dataframe(spike_data)
    beaconed_trial_number,nonbeaconed_trial_number,probe_trial_number,beaconed_position_cm,nonbeaconed_position_cm,probe_position_cm = split_stop_data_by_trial_type(spike_data)

    rewarded_trials = np.array(spike_data.at[1,'rewarded_trials'])

    #take firing locations when on rewarded trials
    rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

    #take firing trial numbers when on rewarded trials
    rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]

    return rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers


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



def plot_rewarded_stops_on_track(recording_folder, spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(4,3))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers = split_stops_by_reward(spike_data)
    trials, unique_trials = remake_trial_numbers(rewarded_beaconed_trial_numbers)

    ax.plot(rewarded_beaconed_position_cm, trials, 'o', color='Black', markersize=2)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    #x_max = max(raw_position_data.trial_number)+0.5
    Python_PostSorting.plot_utility.style_vr_plot(ax, 200,0)
    plt.ylim(0,80)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(recording_folder + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()



def plot_stop_histogram(recording_folder, processed_position_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"])
    average_stops = np.array(processed_position_data["average_stops"])
    ax.plot(position_bins,average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.average_stops)+0.1
    Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(recording_folder + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
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
        plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        x_max = max(np.array(spike_data.at[cluster, "average_speed"]))+0.5
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/speed_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def plot_speed_by_trial_histogram(spike_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/speed/per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position_bins = np.arange(1,201,1)
        cluster_speed, max_trial_number = Python_PostSorting.Speed_Analysis.extract_speed_data(spike_data, cluster)
        ax.plot(position_bins,np.array(spike_data.at[cluster, "average_speed"]), '-', color='Black')
        plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        x_max = max(np.array(spike_data.at[cluster, "average_speed"]))+0.5
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/speed_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def plot_firing_rate_and_speed(recording_folder, spike_data):
    print('I am plotting smoothed firing rate maps with speed...')
    save_path = recording_folder + '/Figures/spike_rate_speed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)

        avg_spikes_on_track = plt.figure(figsize=(5,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        position_bins = np.arange(0,200,1)
        color = 'black'
        ax2.set_ylabel('Speed (cm/s)', color=color, fontsize=14, labelpad = 10)  # we already handled the x-label with ax1
        ax2.plot(position_bins,np.array(spike_data.at[cluster, "average_speed"]), '-', color='red')
        ax2.tick_params(axis='y', labelcolor=color)
        Python_PostSorting.plot_utility.style_vr_twin_plot(ax2, np.max(np.array(spike_data.at[cluster, "average_speed"])), 0)

        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        ax.set_ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        ax.set_xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)

        x_max = np.nanmax(avg_beaconed_spike_rate)
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.4, left = 0.2, right = 0.8, top = 0.92)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_smoothed_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
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



def plot_stops_on_track_per_cluster(spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stops_on_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(4,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        rewarded_stop_locations = np.array(spike_data.at[cluster, "rewarded_locations"])
        rewarded_trials = np.array(spike_data.at[cluster, "rewarded_trials"])

        stop_locations = np.array(spike_data.at[cluster, "stop_location_cm"])
        stop_trials = np.array(spike_data.at[cluster, "stop_trial_number"])

        #beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(stop_locations)

        ax.plot(stop_locations, stop_trials, 'o', color='0.5', markersize=2)
        #ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
        #ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
        ax.plot(rewarded_stop_locations, rewarded_trials, '>', color='Red', markersize=3)
        plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
        plt.xlim(0,200)
        plt.ylim(0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        #x_max = max(raw_position_data.trial_number)+0.5
        Python_PostSorting.plot_utility.style_vr_plot(ax, 70, 0)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_raster_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()
