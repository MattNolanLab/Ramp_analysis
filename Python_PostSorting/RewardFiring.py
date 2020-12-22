import numpy as np
import pandas as pd
import Python_PostSorting.Create2DHistogram
import os
import matplotlib.pylab as plt
from scipy import signal
from scipy import stats


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
    data_filtered = data[data[:,1] >= 3,:]
    return data_filtered


def add_columns(spike_data):
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
    return spike_data


def extract_data_from_frame(spike_data, cluster):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
    data = remove_low_speeds(rates, speed, position, trials, types )
    return rewarded_trials, data


def split_trials(data, rewarded_trials):
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

    return rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types


def split_time_data_by_reward(spike_data, prm):
    spike_data = add_columns(spike_data)

    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)

        ## for all trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data, rewarded_trials)
        spike_data = drop_alldata_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for beaconed trials
        data_filtered = data[data[:,4] == 0,:] # filter data for beaconed trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe trials
        data_filtered = data[data[:,4] == 2,:] # filter data for probe trials & nonbeaconed
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_probe_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        ## for probe & nonbeaconed trials
        data_filtered = data[data[:,4] != 0,:] # filter data for nonbeaconed trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)
        spike_data = drop_nb_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)
        #spike_data = extract_time_binned_firing_rate_rewarded(spike_data, cluster, prm)


        rewarded_locations = np.array(spike_data.loc[cluster, 'rewarded_locations'])
        rewarded_locations = rewarded_locations[~np.isnan(rewarded_locations)]
        locations = np.array(np.append(rewarded_locations, rewarded_locations[0:14]))
        spike_data.at[cluster,"rewarded_locations"] = pd.Series(locations)
        rewarded_locations = np.array(spike_data.loc[cluster, 'rewarded_locations'])
        #spike_data = extract_time_binned_firing_rate_failed(spike_data, cluster, prm)
    return spike_data


def drop_alldata_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


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
        window = signal.gaussian(3, std=5)
        rates = signal.convolve(rates, window, mode='same')/ sum(window)/sum(window)
    except (TypeError, ValueError):
        print("")

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
    spike_data.at[cluster, 'averaged_rewarded_b'] = list(binned_speed)

    #beaconed_failed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path)
    #beaconed_plot(spike_data,cluster,  position_array, binned_speed, binned_speed_sd, save_path)

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
    spike_data.at[cluster, 'averaged_rewarded_p'] = list(np.asarray(data_b))

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
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def find_avg_speed_in_reward_zone(data):
    trialids = np.array(np.unique(np.array(data[:,3])))
    trial_id = np.zeros((trialids.shape[0]))
    for trialcount, trial in enumerate(trialids):
        trial_data = data[data[:,3] == trial,:]# get data only for each trial
        data_in_position = trial_data[trial_data[:,2] > 80,:]
        data_in_position = data_in_position[data_in_position[:,2] < 110,:]
        trial_id[trialcount] = np.nanmean(data_in_position[:,1])
    return trial_id, trialids


def find_confidence_interval(trial_id):
    mean, sigma = np.mean(trial_id), np.std(trial_id)
    interval = stats.norm.interval(0.90, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    #plt.plot(trial_id)
    #plt.show()
    return upper, lower


def catagorise_failed_trials(speed_array_failed, trialids_failed, upper, lower):
    trial_id = np.zeros((speed_array_failed.shape[0]))
    trial_id_runthrough = []
    trial_id_try = []

    for rowcount, row in enumerate(speed_array_failed):
        speed = speed_array_failed[rowcount]
        trial = trialids_failed[rowcount]

        if (speed <= upper and speed >= lower):
            trial_id_try = np.append(trial_id_try, trial)
        else :
            trial_id_runthrough = np.append(trial_id_runthrough, trial)
        trial_id[rowcount] == id
    return trial_id_try, trial_id_runthrough


"""
    rates = data[:,0]
    speed = data[:,1]
    position = data[:,2]
    trials = data[:,3]
    types = data[:,4]
"""


def split_time_data_by_speed(spike_data, prm):
    spike_data["run_through_trialid"] = ""
    spike_data["try_trialid"] = ""
    spike_data["spikes_in_time_try_b"] = ""
    spike_data["spikes_in_time_runthru_b"] = ""

    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)

        ## for beaconed trials
        data_filtered = data[data[:,4] == 0,:] # filter data for beaconed trials
        rates, speed , position, trials, types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data_filtered, rewarded_trials)

        data = np.vstack((rates, speed, position, trials, types))
        data=data.transpose()

        speed_array_rewarded, trialids_rewarded = find_avg_speed_in_reward_zone(data)
        upper, lower = find_confidence_interval(speed_array_rewarded)

        data = np.vstack((failed_rates, failed_speed, failed_position, failed_trials , failed_types))
        data=data.transpose()

        speed_array_failed, trialids_failed = find_avg_speed_in_reward_zone(data)
        trial_id, trial_id2 = catagorise_failed_trials(speed_array_failed, trialids_failed, upper, lower)

        spike_data.at[cluster,"run_through_trialid"] = pd.Series(trial_id2)
        spike_data.at[cluster,"try_trialid"] = pd.Series(trial_id)

        spike_data = split_and_save_data(spike_data)

    return spike_data



def split_and_save_data(spike_data):
    for cluster in range(len(spike_data)):
        try_trials = np.array(spike_data.loc[cluster, 'try_trialid'])
        runthru_trials = np.array(spike_data.loc[cluster, 'run_through_trialid'])

        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        rewarded_rates = rates[np.isin(trials,try_trials)]
        rewarded_speed = speed[np.isin(trials,try_trials)]
        rewarded_position = position[np.isin(trials,try_trials)]
        reward_trials = trials[np.isin(trials,try_trials)]
        reward_types = types[np.isin(trials,try_trials)]
        failed_rates = rates[np.isin(trials,runthru_trials)]
        failed_speed = speed[np.isin(trials,runthru_trials)]
        failed_position = position[np.isin(trials,runthru_trials)]
        failed_trials = trials[np.isin(trials,runthru_trials)]
        failed_types = types[np.isin(trials,runthru_trials)]

        spike_data = drop_runthru_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)
    return spike_data



def drop_runthru_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):

    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_try_b'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_runthru_b'] = list(sn)
    return spike_data



def extract_time_binned_firing_rate_runthru(spike_data):
    spike_data["Rates_averaged_runthru_b"] = ""
    spike_data["Rates_sd_runthru_b"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[4], dtype= np.int32)
        window = signal.gaussian(2, std=2)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()

        if len(np.unique(trials)) > 1:
            # bin data over position bins
            bins = np.arange(0,200,1)
            trial_numbers = np.arange(min(trials),max(trials), 1)
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                if trial_data.shape[0] > 0:
                    t_rates = trial_data[:,0]
                    t_pos = trial_data[:,2]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            x = np.nanmean(data_b, axis=1)
            x_sd = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'Rates_averaged_runthru_b'] = list(x)# add data to dataframe
            spike_data.at[cluster, 'Rates_sd_runthru_b'] = list(x_sd)
        else:
            spike_data.at[cluster, 'Rates_averaged_runthru_b'] = np.nan
            spike_data.at[cluster, 'Rates_sd_runthru_b'] = np.nan
            #print("no data")
    return spike_data



def extract_time_binned_firing_rate_try(spike_data):
    spike_data["Rates_averaged_try_b"] = ""
    spike_data["Rates_sd_try_b"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[4], dtype= np.int32)
        window = signal.gaussian(2, std=2)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        if len(np.unique(trials)) > 1:
            # bin data over position bins
            bins = np.arange(0,200,1)
            trial_numbers = np.arange(min(trials),max(trials), 1)
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                if trial_data.shape[0] > 0:
                    t_rates = trial_data[:,0]
                    t_pos = trial_data[:,2]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            x = np.nanmean(data_b, axis=1)
            x_sd = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'Rates_averaged_try_b'] = list(x)# add data to dataframe
            spike_data.at[cluster, 'Rates_sd_try_b'] = list(x_sd)
        else:
            spike_data.at[cluster, 'Rates_averaged_try_b'] = np.nan
            spike_data.at[cluster, 'Rates_sd_try_b'] = np.nan
            #print("no data")
    return spike_data

