import numpy as np
import pandas as pd
import Python_PostSorting.Create2DHistogram
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.ConvolveRates_FFT

def create_reward_histogram(spike_data, cluster, max_trial):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_positions = np.array(spike_data.loc[cluster, 'rewarded_locations'])
    bins = np.arange(0,(200)+1,1)
    trialrange = np.arange(1,(max_trial+1),1)
    reward_histogram = Python_PostSorting.Create2DHistogram.create_2dhistogram(rewarded_trials, rewarded_positions, bins, trialrange)
    return reward_histogram


def reshape_reward_histogram(reward_histogram, spike_data, cluster):
    reshaped_reward_histogram = np.reshape(reward_histogram, (reward_histogram.shape[0]*reward_histogram.shape[1]))
    spike_data.at[cluster, 'reward_histogram'] = list(reshaped_reward_histogram)
    return spike_data


def find_rewarded_trials(reward_histogram):
    trial_indicator = np.sum(reward_histogram, axis=1)
    return trial_indicator


def fill_in_binned_trial_indicator(trial_indicator):
    binned_trial_indicator=[]
    for row in trial_indicator:
        trial_reward_indicator = [row]
        whole_trial_as_indicator = np.repeat(trial_reward_indicator, 200)
        binned_trial_indicator = np.append(binned_trial_indicator, whole_trial_as_indicator)
    return binned_trial_indicator


def generate_reward_indicator(spike_data):
    print("generating reward indicator...")
    spike_data["binned_trial_indicator"] = ""
    spike_data["reward_histogram"] = ""

    for cluster in range(len(spike_data)):
        trials=np.max(np.array(spike_data.loc[cluster].spike_rate_on_trials_smoothed[1], dtype= np.int32))
        reward_histogram = create_reward_histogram(spike_data, cluster, trials)
        spike_data = reshape_reward_histogram(reward_histogram, spike_data, cluster)
        trial_indicator = find_rewarded_trials(reward_histogram)
        binned_trial_indicator = fill_in_binned_trial_indicator(trial_indicator)
        spike_data.at[cluster,'binned_trial_indicator'] = list(binned_trial_indicator)
    return spike_data


def package_reward_data_for_r(spike_data):
    print("packaging data for R...")
    spike_data["R_Reward_data"] = ""
    for cluster_index in range(len(spike_data)):
        reward = np.array(spike_data.at[cluster_index, "binned_trial_indicator"], dtype= np.int32)
        trials=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype= np.int32)
        types=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2], dtype= np.int32)
        rate=np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])

        if reward.shape[0] == trials.shape[0]:
            sr=[]
            sr.append(np.array(rate))
            sr.append(np.array(reward))
            sr.append(np.array(trials))
            sr.append(np.array(types))
            spike_data.at[cluster_index, 'R_Reward_data'] = list(sr)
        else:
            print("Arrays are not the same shape")
    return spike_data


def add_columns_to_dataframe(spike_data):
    spike_data["rewarded_beaconed_position_cm"] = ""
    spike_data["rewarded_nonbeaconed_position_cm"] = ""
    spike_data["rewarded_probe_position_cm"] = ""
    spike_data["rewarded_beaconed_trial_numbers"] = ""
    spike_data["rewarded_nonbeaconed_trial_numbers"] = ""
    spike_data["rewarded_probe_trial_numbers"] = ""
    spike_data["failed_beaconed_position_cm"] = ""
    spike_data["failed_nonbeaconed_position_cm"] = ""
    spike_data["failed_probe_position_cm"] = ""
    spike_data["failed_beaconed_trial_numbers"] = ""
    spike_data["failed_nonbeaconed_trial_numbers"] = ""
    spike_data["failed_probe_trial_numbers"] = ""
    return spike_data


def split_trials_by_reward(spike_data, cluster_index):
    #spike_data = add_columns_to_dataframe(spike_data)
    beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster_index)

    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)

    #take firing locations when on rewarded trials
    rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

    #take firing trial numbers when on rewarded trials
    rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]

    return rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers


def split_trials_by_failure(spike_data, cluster_index):
    #spike_data = add_columns_to_dataframe(spike_data)
    beaconed_position_cm, nonbeaconed_position_cm, probe_position_cm, beaconed_trial_number, nonbeaconed_trial_number, probe_trial_number = Python_PostSorting.ExtractFiringData.split_firing_by_trial_type(spike_data, cluster_index)

    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)

    #take firing locations when on rewarded trials
    failed_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials, invert=True)]
    failed_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials, invert=True)]
    failed_probe_position_cm = probe_position_cm[~np.isin(probe_trial_number,rewarded_trials, invert=True)]

    #take firing trial numbers when on rewarded trials
    failed_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials, invert=True)]
    failed_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials, invert=True)]
    failed_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials, invert=True)]

    return failed_beaconed_position_cm, failed_nonbeaconed_position_cm, failed_probe_position_cm, failed_beaconed_trial_numbers, failed_nonbeaconed_trial_numbers, failed_probe_trial_numbers



def remove_low_speeds(rates, speed, position,trials, types ):
    data = np.vstack((rates, speed, position, trials, types))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    return data_filtered



def split_time_data_by_reward(spike_data, prm):
    spike_data["spikes_in_time_rewarded"] = ""
    spike_data["spikes_in_time_failed"] = ""
    spike_data["spikes_in_time_rewarded_b"] = ""
    spike_data["spikes_in_time_failed_b"] = ""
    spike_data["spikes_in_time_rewarded_p"] = ""
    spike_data["spikes_in_time_failed_p"] = ""
    spike_data["spikes_in_time_rewarded_nb"] = ""
    spike_data["spikes_in_time_failed_nb"] = ""
    spike_data["averaged_rewarded_b"] = ""
    spike_data["averaged_failed_b"] = ""
    spike_data["averaged_rewarded_nb"] = ""
    spike_data["averaged_failed_nb"] = ""
    spike_data["averaged_rewarded_p"] = ""
    spike_data["averaged_failed_p"] = ""

    for cluster in range(len(spike_data)):
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        data = remove_low_speeds(rates, speed, position, trials, types )

        ## for all trials
        rates = data[:,0]
        speed = data[:,1]
        position = data[:,2]
        trials = data[:,3]
        types = data[:,4]

        rewarded_rates = rates[np.isin(trials,rewarded_trials)]
        rewarded_speed = speed[np.isin(trials,rewarded_trials)]
        rewarded_position = position[np.isin(trials,rewarded_trials)]
        reward_trials = trials[np.isin(trials,rewarded_trials)]
        reward_types = types[np.isin(trials,rewarded_trials)]
        failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
        failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
        failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
        failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
        failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

        spike_data = drop_all_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for beaconed trials
        data_filtered = data[data[:,4] == 0,:] # filter data for beaconed trials
        rates = data_filtered[:,0]
        speed = data_filtered[:,1]
        position = data_filtered[:,2]
        trials = data_filtered[:,3]
        types = data_filtered[:,4]

        rewarded_rates = rates[np.isin(trials,rewarded_trials)]
        rewarded_speed = speed[np.isin(trials,rewarded_trials)]
        rewarded_position = position[np.isin(trials,rewarded_trials)]
        reward_trials = trials[np.isin(trials,rewarded_trials)]
        reward_types = types[np.isin(trials,rewarded_trials)]
        failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
        failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
        failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
        failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
        failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

        spike_data = drop_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe trials
        data_filtered = data[data[:,4] == 2,:] # filter data for probe trials & nonbeaconed
        rates = data_filtered[:,0]
        speed = data_filtered[:,1]
        position = data_filtered[:,2]
        trials = data_filtered[:,3]
        types = data_filtered[:,4]

        rewarded_rates = rates[np.isin(trials,rewarded_trials)]
        rewarded_speed = speed[np.isin(trials,rewarded_trials)]
        rewarded_position = position[np.isin(trials,rewarded_trials)]
        reward_trials = trials[np.isin(trials,rewarded_trials)]
        reward_types = types[np.isin(trials,rewarded_trials)]
        failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
        failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
        failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
        failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
        failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

        spike_data = drop_probe_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe & nonbeaconed trials
        data_filtered = data[data[:,4] != 0,:] # filter data for nonbeaconed trials
        rates = data_filtered[:,0]
        speed = data_filtered[:,1]
        position = data_filtered[:,2]
        trials = data_filtered[:,3]
        types = data_filtered[:,4]

        rewarded_rates = rates[np.isin(trials,rewarded_trials)]
        rewarded_speed = speed[np.isin(trials,rewarded_trials)]
        rewarded_position = position[np.isin(trials,rewarded_trials)]
        reward_trials = trials[np.isin(trials,rewarded_trials)]
        reward_types = types[np.isin(trials,rewarded_trials)]
        failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
        failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
        failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
        failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
        failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

        spike_data = drop_nb_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)
        spike_data = extract_time_binned_firing_rate_rewarded(spike_data, cluster, prm)
        #spike_data = extract_time_binned_firing_rate_failed(spike_data, cluster, prm)
    return spike_data



def drop_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_b'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_failed_b'] = list(sn)
    return spike_data


def drop_all_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_rewarded'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_failed'] = list(sn)
    return spike_data


def drop_probe_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_p'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_failed_p'] = list(sn)
    return spike_data


def drop_nb_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_rewarded_nb'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_failed_nb'] = list(sn)
    return spike_data



def beaconed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path):
    cluster_index = spike_data.cluster_id.values[cluster] - 1
    speed_histogram = plt.figure(figsize=(4,3))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(position_array,binned_speed, '-', color='Black')
    ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)

    plt.ylabel('Rates (Hz)', fontsize=12, labelpad = 10)
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
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    ax.set_ylim(0)
    plt.locator_params(axis = 'x', nbins  = 3)
    ax.set_xticklabels(['-30', '70', '170'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
    plt.close()


def probe_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path):
    cluster_index = spike_data.cluster_id.values[cluster] - 1
    speed_histogram = plt.figure(figsize=(4,3))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(position_array,binned_speed, '-', color='Black')
    ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)

    plt.ylabel('Rates (Hz)', fontsize=12, labelpad = 10)
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
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    ax.set_ylim(0)
    plt.locator_params(axis = 'x', nbins  = 3)
    ax.set_xticklabels(['-30', '70', '170'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_probe.png', dpi=200)
    plt.close()


def beaconed_failed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path):
    cluster_index = spike_data.cluster_id.values[cluster] - 1
    speed_histogram = plt.figure(figsize=(4,3))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(position_array,binned_speed, '-', color='Black')
    ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)

    plt.ylabel('Rates (Hz)', fontsize=12, labelpad = 10)
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
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    ax.set_ylim(0)
    plt.locator_params(axis = 'x', nbins  = 3)
    ax.set_xticklabels(['-30', '70', '170'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'failed.png', dpi=200)
    plt.close()


def extract_time_binned_firing_rate_rewarded(spike_data,cluster, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    speed=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[1])
    rates=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[0])
    position=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[2])
    #trials=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[3], dtype= np.int32)
    types=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[4], dtype= np.int32)

    try:
        rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
        #speed = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(speed)
    except TypeError:
        print("Error")

    data = np.vstack((rates,speed,position,types))
    data=data.transpose()

    data_filtered = data[data[:,3] == 0,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed
    binned_speed = convolve_with_scipy(binned_speed)
    #binned_speed_sd = convolve_with_scipy(binned_speed_sd)
    spike_data.at[cluster, 'averaged_rewarded_b'] = list(binned_speed)

    beaconed_failed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path)

    data_filtered = data[data[:,3] == 2,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed

    data_b = pd.Series(binned_speed,dtype=None, copy=False)
    data_b = data_b.interpolate(method='linear', order=2)
    data_b = np.asarray(data_b)
    #binned_speed = convolve_with_scipy(binned_speed)
    spike_data.at[cluster, 'averaged_rewarded_p'] = list(data_b)

    data_filtered = data[data[:,3] != 0,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed
    binned_speed = convolve_with_scipy(binned_speed)
    spike_data.at[cluster, 'averaged_rewarded_nb'] = list(binned_speed)
    return spike_data


def extract_time_binned_firing_rate_failed(spike_data, cluster, prm):
    speed=np.array(spike_data.iloc[cluster].spikes_in_time_failed[1])
    rates=np.array(spike_data.iloc[cluster].spikes_in_time_failed[0])
    position=np.array(spike_data.iloc[cluster].spikes_in_time_failed[2])
    types=np.array(spike_data.iloc[cluster].spikes_in_time_failed[4], dtype= np.int32)

    rates = convolve_with_scipy(rates)

    data = np.vstack((rates,speed,position,types))
    data=data.transpose()

    data_filtered = data[data[:,3] == 0,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed
    binned_speed = convolve_with_scipy(binned_speed)
    #binned_speed_sd = convolve_with_scipy(binned_speed_sd)
    spike_data.at[cluster, 'averaged_failed_b'] = list(binned_speed)

    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    beaconed_failed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path)

    data_filtered = data[data[:,3] == 2,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed
    #binned_speed = convolve_with_scipy(binned_speed)
    #binned_speed_sd = convolve_with_scipy(binned_speed_sd)
    spike_data.at[cluster, 'averaged_failed_p'] = list(binned_speed)

    data_filtered = data[data[:,3] != 0,:]
    rates = data_filtered[:,0]
    position = data_filtered[:,2]

    position_array = np.arange(1,201,1)
    binned_speed = np.zeros((position_array.shape))
    binned_speed_sd = np.zeros((position_array.shape))
    for rowcount, row in enumerate(position_array):
        speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
        average_speed = np.nanmean(speed_in_position)
        sd_speed = np.nanstd(speed_in_position)
        binned_speed[rowcount] = average_speed
        binned_speed_sd[rowcount] = sd_speed
    #binned_speed = convolve_with_scipy(binned_speed)
    #binned_speed_sd = convolve_with_scipy(binned_speed_sd)
    spike_data.at[cluster, 'averaged_failed_nb'] = list(binned_speed)
    return spike_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


