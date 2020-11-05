import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import math
from scipy import signal
import Python_PostSorting.ConvolveRates_FFT

def extract_speed_data(spike_data, cluster):
    speed = np.array(spike_data.at[cluster, "binned_speed_ms_per_trial"])
    max_trial_number=int(speed.shape[0]/200)+1
    trials = np.repeat(np.arange(1,max_trial_number), 200)

    cluster_speed = pd.DataFrame({ 'speed' :  speed, 'trial_number' :  trials})
    return cluster_speed


def efficient_extract_of_speed(spike_data, cluster_index):
    cluster_speed = pd.DataFrame({ 'speed' :  spike_data.at[cluster_index, "binned_speed_ms_per_trial"], 'trial_number' :  spike_data.loc[cluster_index].spike_rate_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials[2]})
    return cluster_speed


def split_firing_data_by_trial_type(cluster_speed):
    beaconed_cluster_speed = cluster_speed.where(cluster_speed["trial_type"] ==0)
    nbeaconed_cluster_speed = cluster_speed.where(cluster_speed["trial_type"] ==1)
    probe_cluster_speed = cluster_speed.where(cluster_speed["trial_type"] ==2)
    return beaconed_cluster_speed, nbeaconed_cluster_speed, probe_cluster_speed


def split_speed_by_reward(spike_data, cluster_index, cluster_speed):
    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)
    speed = np.array(cluster_speed['speed'], dtype=np.int16)
    trials = np.array(cluster_speed['trial_number'], dtype=np.int16)

    rewarded_speed = speed[np.isin(trials,rewarded_trials)]
    rewarded_trials = trials[np.isin(trials,rewarded_trials)]

    #rewarded_cluster_speed = cluster_speed.where(cluster_speed["trial_type"] ==0)
    #nonrewarded_cluster_speed = cluster_speed.where(cluster_speed["trial_type"] ==1)
    return rewarded_speed, rewarded_trials


def reshape_and_average_over_trials(beaconed_cluster_speed, max_trial_number):
    beaconed_reshaped_hist = np.reshape(beaconed_cluster_speed, (int(max_trial_number), 200))
    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    return np.array(average_beaconed_spike_rate, dtype=np.float16)


def calculate_average_speed(spike_data):
    print("calculating average speed")
    spike_data["average_speed"] = ""
    for cluster in range(len(spike_data)):
        cluster_speed = efficient_extract_of_speed(spike_data, cluster)
        #beaconed_cluster_speed = split_firing_data_by_trial_type(cluster_speed)
        rewarded_speed, rewarded_trials = split_speed_by_reward(spike_data, cluster, cluster_speed)
        unique_trials=np.unique(rewarded_trials)
        average_beaconed_speed= reshape_and_average_over_trials(rewarded_speed, unique_trials.shape[0])
        #spike_data = add_speed_data_to_dataframe(cluster, average_beaconed_speed,spike_data)
        spike_data.at[cluster, "average_speed"] = list(average_beaconed_speed)
    return spike_data


### --------------------------------------------------------------------------------------- ###


## histogram of speeds

def generate_speed_histogram(spike_data, recording_folder):
    print('I am calculating speed histogram...')
    spike_data["speed_histogram"] = ""
    save_path = recording_folder + 'Figures/Speed_histogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real/1000
        speed = speed[speed > 0]
        try:
            posrange = np.linspace(0, 100, num=100)
            values = np.array([[posrange[0], posrange[-1]]])
            H, bins = np.histogram(speed, bins=(posrange), range=values)
            plt.plot(bins[1:], H)
            #plt.hist(speed, density=True, bins=50)
            #plt.xlim(-5, 100)
            plt.ylabel('Probability')
            plt.xlabel('Data')
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_histogram_Cluster_' + str(cluster +1) + '_1' + '.png', dpi=200)
            plt.close()

            spike_data.at[cluster,"speed_histogram"] = [H, bins]
        except ValueError:
            continue
    return spike_data


### --------------------------------------------------------------------------------------- ###


def fix_position(position):
    position_fixed = np.zeros((position.shape[0]))
    for rowcount, row in enumerate(position[1:]):
        position_diff = position[rowcount] - position[rowcount-1]
        next_position_diff = position[rowcount]+1 - rowcount
        if position_diff < -110:
            position_fixed[rowcount] = 0
        else:
            position_fixed[rowcount] = position[rowcount]
    return position_fixed


def cumulative_position(position):
    position_fixed = np.zeros((position.shape[0]))
    total_position=0
    for rowcount, row in enumerate(position[1:]):
        position_diff = position[rowcount] - position[rowcount-1]
        if position_diff < -20:
            total_position+=200
        cum_position = position[rowcount] + total_position
        position_fixed[rowcount] = cum_position
    return position_fixed


def calculate_speed_from_position(spike_data, recording_folder):
    print('I am calculating speed from position...')
    save_path = recording_folder + 'Figures/Speed_from_Position'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_path = recording_folder + 'Figures/Speed_from_Position'

    spike_data["speed"] = ""
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real, dtype=np.float32)/1000
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real, dtype=np.float32)
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real, dtype=np.float32)*10
        try:
            position = fix_position(position)
            cum_position = cumulative_position(position)
            pos_diff = np.diff(cum_position)
            pos_diff = pos_diff*4
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)

            plt.scatter(pos_diff, rates[1:], marker='o', s=1)
            plt.scatter(speed, rates, s=1)
            plt.xlim(-50, 60)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_Cluster_' + str(cluster +1) + '_3' + '.png', dpi=200)
            plt.close()

        except IndexError:
            print(session_id)

    return spike_data





### --------------------------------------------------------------------------------------- ###



### FOR R ####

def package_speed_data_for_r(spike_data):
    for cluster_index in range(len(spike_data)):
        spike_data["speed_data"] = ""
        speed = np.array(spike_data.at[cluster_index, "binned_speed_ms_per_trial"])
        trials=np.array(spike_data.loc[cluster_index].spike_num_on_trials[1], dtype= np.int32)
        types=np.array(spike_data.loc[cluster_index].spike_num_on_trials[2], dtype= np.int32)

        sr=[]
        sr.append(np.array(speed))
        sr.append(np.array(trials))
        sr.append(np.array(types))
        spike_data.at[cluster_index, 'speed_data'] = list(sr)
    return spike_data


def package_data_for_r(spike_data):
    spike_data["R_LM_data"] = ""
    for cluster_index in range(len(spike_data)):
        speed = np.array(spike_data.at[cluster_index, "binned_speed_ms_per_trial"])
        trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials[1], dtype= np.int32)
        types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials[2], dtype= np.int32)
        rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials[0])
        time = np.array(spike_data.at[cluster_index, "binned_time_ms_per_trial"])

        sr=[]
        sr.append(np.array(rate))
        sr.append(np.array(speed))
        sr.append(np.array(time))
        sr.append(np.array(trials))
        sr.append(np.array(types))
        spike_data.at[cluster_index, 'R_LM_data'] = list(sr)
    return spike_data






### ----------------------------------------------------------------------------------------- ###


def extract_time_binned_speed(spike_data, prm):
    spike_data["Speed_averaged"] = ""
    for cluster in range(len(spike_data)):
        speed = np.array(spike_data.at[cluster, "speed_rate_in_time"])
        position = np.array(spike_data.at[cluster, "position_rate_in_time"])

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape))
        binned_speed_sd = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            speed_in_position = np.take(speed, np.where(np.logical_and(position >= rowcount, position <= rowcount+1)))
            average_speed = np.nanmean(speed_in_position)
            sd_speed = np.nanstd(speed_in_position)
            binned_speed[rowcount] = average_speed/1000
            binned_speed_sd[rowcount] = sd_speed
        spike_data.at[cluster, 'Speed_averaged'] = list(binned_speed)


        ##print('plotting speed histogram...', cluster)
        save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/speed'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        binned_speed = convolve_with_scipy(binned_speed)
        binned_speed_sd = convolve_with_scipy(binned_speed_sd)
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,2))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed/1000, '-', color='Black')
        ax.fill_between(position_array, binned_speed/1000-binned_speed_sd/1000,binned_speed/1000+binned_speed_sd/1000, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)

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
        ax.axhline(-5, linewidth = 2.5, color = 'black') # bold line on the x axis
        ax.set_ylim(-5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_speed_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

    return spike_data


def convolve_with_scipy(rate):
    window = signal.gaussian(20, std=3)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')
    #filtered_time = signal.convolve(time, window, mode='same')
    #convolved_rate = (filtered/filtered_time)
    return (convolved_rate/10)











### --------------------------------------------------------------------------------------- ###

 ### Bin speed in space


def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def make_location_bins(shuffled_cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(shuffled_cluster_firings[:,1])
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def bin_speed_over_space(speed, position, trials, bin_array, max_trial):
    data = np.transpose(np.vstack((speed,position, trials)))
    bins = np.arange(0,200,1)
    accel_over_trials = np.zeros((len(bins), max_trial))
    for tcount, trial in enumerate(np.arange(1,max_trial+1,1)):
        trial_data = data[data[:,2] == trial,:]# get data only for each trial
        for bcount, bin in enumerate(bins):
            accel_above_position = trial_data[trial_data[:,1] >= bin ,:]# get data only for bins +
            accel_in_position = accel_above_position[accel_above_position[:,1] <= bin+1 ,:]# get data only for bin
            mean_accel = np.nanmean(accel_in_position[:,0])
            accel_over_trials[bcount, tcount] = mean_accel
    return accel_over_trials


def calculate_speed_binned_in_space(recording_folder, spike_data):
    print('I am calculating speed over space...')
    spike_data["speed_rate_on_trials"] = ""
    for cluster in range(len(spike_data)):
        cluster_firings = extract_firing_info(spike_data, cluster)
        bin_array = make_location_bins(cluster_firings)

        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        trials = np.array(spike_data.iloc[cluster].spike_rate_in_time[4], dtype= np.int32)
        max_trial = np.max(trials)+1

        binned_speed = bin_speed_over_space(speed, position, trials, bin_array, max_trial)

        #plot_avg_speed(recording_folder, spike_data, cluster, binned_acceleration)
        #plot_trial_acceleration(recording_folder, spike_data, cluster, binned_acceleration, trials, position, speed)

        spike_data = add_speed_to_dataframe(cluster, binned_speed, spike_data)
    return spike_data



def add_speed_to_dataframe(cluster_index, binned_acceleration, spike_data):
    spike_data.at[cluster_index, 'speed_rate_on_trials'] = binned_acceleration

    return spike_data

