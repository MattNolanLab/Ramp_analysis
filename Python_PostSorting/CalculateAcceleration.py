import os
import matplotlib.pylab as plt
import Python_PostSorting.ConvolveRates_FFT
import numpy as np
import cmath
import pandas as pd
from scipy import signal



### --------------------------------------------------------------------------------------------------- ###

### CALCULATE ACCELERATION OVER TIME



def generate_acceleration(spike_data, recording_folder):
    print('I am calculating acceleration...')
    spike_data["spikes_in_time"] = ""
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real*10 #*4 to convert from 250 ms sampling to Hz
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real # *4 to convert from 250 ms sampling to Hz
        trials =  np.array(spike_data.iloc[cluster].spike_rate_in_time[3]).real
        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real

        # filter data
        try:
            window = signal.gaussian(2, std=3)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except (ValueError, TypeError):
                continue

        # remove outliers
        rates_o =  pd.Series(rates)
        speed_o =  pd.Series(speed)
        position_o =  pd.Series(position)
        trials_o =  pd.Series(trials)
        types_o =  pd.Series(types)

        rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        trials = trials_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        types = types_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers

        if speed.size > 1:
            #calculate acceleration
            acceleration = np.diff(np.array(speed))
            acceleration = np.hstack((0, acceleration))

            #plot_acceleration(recording_folder, spike_data, cluster_index, speed, acceleration)
            #plot_instant_acceleration(recording_folder, spike_data, cluster_index, rates, position, speed, acceleration)
            plot_instant_acceleration_by_segment(recording_folder, spike_data, cluster, cluster_index, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration))
        else:
            acceleration = np.zeros((speed.size))
        spike_data = store_acceleration(spike_data, cluster, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration), np.asarray(trials), np.asarray(types))
    return spike_data


def plot_acceleration(recording_folder, spike_data, cluster, speed, acceleration):
    plt.plot(speed[:1000])
    plt.plot(acceleration[:1000])
    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_accell_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return


def remove_low_speeds(rates, speed, position, acceleration ):
    data = np.vstack((rates, speed, position, acceleration))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    rates = data_filtered[:,0]
    speed = data_filtered[:,1]
    position = data_filtered[:,2]
    acceleration = data_filtered[:,3]
    return rates, speed, position, acceleration


def plot_instant_acceleration(recording_folder, spike_data, cluster, rates, position, speed, acceleration):
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    rates, speed, position, acceleration = remove_low_speeds(rates, speed, position,acceleration )
    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(acceleration, rates, 'o', color='Black', markersize=1.5)
    plt.ylabel('Spike rate (hz)', fontsize=10, labelpad = 10)
    plt.xlabel('Speed (cm/s)', fontsize=10, labelpad = 10)
    ax.locator_params(axis = 'x', nbins=3)
    plt.locator_params(axis = 'y', nbins  = 4)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster +1) + '_acceleration' + '.png', dpi=200)
    plt.close()
    return


def remove_low_speeds_and_segment(rates, speed, position, acceleration ):
    data = np.vstack((rates, speed, position, acceleration))
    data=data.transpose()
    data_filtered = data[data[:,1] >= 3,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,2] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]
    acceleration_outbound = data_outbound[:,3]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    position_homebound = data_homebound[:,2]
    acceleration_homebound = data_homebound[:,3]

    return rates_outbound , speed_outbound , position_outbound , acceleration_outbound, rates_homebound, speed_homebound, position_homebound, acceleration_homebound


def plot_instant_acceleration_by_segment(recording_folder, spike_data, cluster, cluster_index, rates, position, speed, acceleration):
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    rates_o, speed_o, position_o, acceleration_o, rates_h, speed_h, position_h, acceleration_h = remove_low_speeds_and_segment(rates, speed, position, acceleration )

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(acceleration_o, rates_o, s=area, c=position_o)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Acceleration $(cm/s^2)$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-20, 25)
    #ax.set_ylim(0)
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
    #ax.axvline(-100, linewidth = 1.5, color = 'black') # bold line on the y axis
    #ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_acceleration' + '_coded_outbound.png', dpi=200)
    plt.close()

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(acceleration_h, rates_h, s=area, c=position_h)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Acceleration $(cm/s)^2$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-100, 150)
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
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_acceleration' + '_coded_homebound.png', dpi=200)
    plt.close()
    return


def store_acceleration(spike_data,cluster_index, rates, position, speed, acceleration,  trials, types):
    sn=[]
    sn.append(rates) # rate
    sn.append(position) # speed
    sn.append(speed) # position
    sn.append(acceleration) # acceleration
    sn.append(trials) # trials
    sn.append(types) # types
    spike_data.at[cluster_index, 'spikes_in_time'] = list(sn)
    return spike_data






### --------------------------------------------------------------------------------------------------- ###

### ACCELERATION OVER SPACE



def extract_firing_info(spike_data, cluster_index):
    rate=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[0])
    trials=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    cluster_firings=np.transpose(np.vstack((rate,trials, types)))
    return cluster_firings


def make_location_bins(shuffled_cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(shuffled_cluster_firings[:,1])
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def bin_acceleration_over_space(acceleration, position, trials, bin_array, max_trial):
    data = np.transpose(np.vstack((acceleration,position, trials)))
    bins = np.arange(0,200,1)
    accel_over_trials = np.zeros((len(bins), max_trial+1))
    for tcount, trial in enumerate(np.arange(1,max_trial,1)):
        trial_data = data[data[:,2] == trial,:]# get data only for each trial
        for bcount, bin in enumerate(bins):
            accel_above_position = trial_data[trial_data[:,1] >= bin ,:]# get data only for bins +
            accel_in_position = accel_above_position[accel_above_position[:,1] < bin+1 ,:]# get data only for bin
            mean_accel = np.nanmean(accel_in_position[:,0])
            accel_over_trials[bcount, tcount] = mean_accel
    return accel_over_trials



def plot_avg_acceleration(recording_folder, spike_data, cluster, acceleration):
    spike_data["avg_acceleration"] = ""
    shapes = int(acceleration.size/200) * 200
    acceleration = acceleration[:shapes]
    reshaped_hist = np.reshape(acceleration, (int(acceleration.size/200),200))
    speed = np.array(spike_data.at[cluster, "Speed_averaged"])

    plt.plot(speed, color="Red")
    plt.plot(np.nanmean(reshaped_hist, axis=0), color="Black")
    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_Avg_accel_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return


def plot_trial_acceleration(recording_folder, spike_data, cluster, acceleration):
    speed=np.array(spike_data.loc[cluster].speed_rate_on_trials[0], dtype= np.int32)
    trials=np.array(spike_data.loc[cluster].spike_rate_on_trials_smoothed[1], dtype= np.int32)
    types=np.array(spike_data.loc[cluster].spike_rate_on_trials_smoothed[2], dtype= np.int32)
    bins=np.arange(1,201,1)
    max_trial = np.max(trials)+1
    position= np.tile(bins,int(max_trial))

    data = np.vstack((trials, speed, position, acceleration))
    trial_data = data[data[:,0] == 10,:] # get data only for one trial

    plt.plot(speed, color="Red")
    plt.plot(trial_data[:,2], trial_data[:,3], color="Black")

    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_Trial_accel_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return


def calculate_acceleration_binned_in_space(recording_folder, spike_data):
    print('I am calculating acceleration over space...')
    spike_data["acceleration_rate_on_trials"] = ""
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        cluster_firings = extract_firing_info(spike_data, cluster)
        bin_array = make_location_bins(cluster_firings)

        acceleration =  np.array(spike_data.loc[cluster_index].spikes_in_time[3])
        speed =  np.array(spike_data.loc[cluster_index].spikes_in_time[2])
        position=np.array(spike_data.loc[cluster_index].spikes_in_time[1])
        trials=np.array(spike_data.loc[cluster_index].spikes_in_time[4], dtype= np.int32)
        max_trial = np.max(trials)

        binned_acceleration = bin_acceleration_over_space(acceleration, position, trials, bin_array, max_trial)

        plot_avg_acceleration(recording_folder, spike_data, cluster, binned_acceleration)
        plot_trial_acceleration(recording_folder, spike_data, cluster, binned_acceleration)

        spike_data = add_rate_data_to_dataframe(cluster, cluster_firings, binned_acceleration, spike_data, bin_array)
    return spike_data



def add_rate_data_to_dataframe(cluster_index, cluster_firings, binned_acceleration, spike_data, bin_array):
    sr=[]
    sr.append(np.array(binned_acceleration))
    sr.append(np.array(cluster_firings[:,1], dtype=np.int32))
    sr.append(np.array(cluster_firings[:,2], dtype=np.int32))
    sr.append(bin_array)

    spike_data.at[cluster_index, 'acceleration_rate_on_trials'] = list(sr)

    return spike_data

