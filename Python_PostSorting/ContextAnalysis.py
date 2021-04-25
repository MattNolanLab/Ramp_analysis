import numpy as np
import pandas as pd
import Python_PostSorting.Create2DHistogram
import matplotlib.pylab as plt
from scipy import signal
from scipy import stats
import os





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

        ax.axhspan(11,20, facecolor='k', linewidth =0, alpha=.15) # black box
        ax.axhspan(31,40, facecolor='k', linewidth =0, alpha=.15) # black box
        ax.axhspan(51,60, facecolor='k', linewidth =0, alpha=.15) # black box
        ax.axhspan(71,80, facecolor='k', linewidth =0, alpha=.15) # black box
        ax.axhspan(91,90, facecolor='k', linewidth =0, alpha=.15) # black box

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


def load_stop_data(spatial_data):
    locations = np.array(spatial_data.at[1,'stop_location_cm'])
    trials = np.array(spatial_data.at[1,'stop_trial_number'])
    return locations,trials


def split_stop_data_by_trial_type(spatial_data):
    locations,trials = load_stop_data(spatial_data)
    stop_data=np.transpose(np.vstack((locations, trials)))
    return stop_data



def plot_stops_on_track(recording_folder, spike_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    stops_on_track = plt.figure(figsize=(4,3))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed = split_stop_data_by_trial_type(spike_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='black', markersize=2)
    ax.plot(spike_data.at[1,"rewarded_locations"], spike_data.at[1,"rewarded_trials"], '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    #x_max = max(raw_position_data.trial_number)+0.5
    plt.ylim(0)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(recording_folder + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()



def split_trials_by_reward(spike_data, cluster_index):
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


def add_columns(spike_data):
    spike_data["spikes_in_time_c1"] = ""
    spike_data["spikes_in_time_c2"] = ""
    return spike_data


def make_context_trial_array():
    context1 = np.array((1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,41,42,43,44,45,46,47,48,49,50,61,62,63,64,65,66,67,68,69,70,81,82,83,84,85,86,87,88,89,90))
    context2 = np.array((11,12,13,14,15,16,17,18,19,20,31,32,33,34,35,36,37,38,39,40,51,52,53,54,55,56,57,58,59,60,71,72,73,74,75,76,77,78,79,80,91,92,93,94,95,96,97,98,99,100))
    return context1,context2


def extract_data_from_frame(spike_data, cluster):
    context1,context2 = make_context_trial_array()
    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

    window = signal.gaussian(2, std=3)
    speed = signal.convolve(speed, window, mode='same')/ sum(window)
    data = np.vstack((rates, speed, position, trials, types))
    data=data.transpose()
    return context1, data


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


def split_time_data_by_context(spike_data, prm):
    spike_data = add_columns(spike_data)

    for cluster in range(len(spike_data)):
        context1, data = extract_data_from_frame(spike_data, cluster)

        ## for all trials
        rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types = split_trials(data, context1)
        spike_data = drop_context1_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

    return spike_data


def drop_context1_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):
    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # types
    spike_data.at[cluster_index, 'spikes_in_time_c1'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # types
    spike_data.at[cluster_index, 'spikes_in_time_c2'] = list(sn)
    return spike_data




def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate





def extract_time_binned_firing_rate_context1(spike_data):
    spike_data["Rates_averaged_c1"] = ""
    spike_data["Rates_sd_c1"] = ""


    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_c1[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_c1[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_c1[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_c1[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_c1[4], dtype= np.int32)
        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        # bin data over position bins
        bins = np.arange(0,200,1)
        trial_numbers = np.arange(min(trials),max(trials), 1)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                type_in_position = int(trial_data[0,3])
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                    average_rates = np.nanmean(rate_in_position)
                    if type_in_position == 0 :
                        binned_data[bcount, tcount, 0] = average_rates
                    if (type_in_position == 1 or type_in_position == 2) :
                        binned_data[bcount, tcount, 1] = average_rates
                    if type_in_position == 2 :
                        binned_data[bcount, tcount, 2] = average_rates

        #remove nans interpolate
        data_b = pd.DataFrame(binned_data[:,:,0], dtype=None, copy=False)
        data_b = data_b.dropna(axis = 1, how = "all")
        data_b.reset_index(drop=True, inplace=True)
        data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = np.asarray(data_b)
        x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Rates_averaged_c1'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'Rates_sd_c1'] = list(x_sd)

    return spike_data


def plot_c1_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_c1'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_c1'])

        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

        plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
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
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off

        ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 4)
        ax.set_xticklabels(['10', '30', '50'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'context1.png', dpi=200)
        plt.close()

    return spike_data





def extract_time_binned_firing_rate_context2(spike_data):
    spike_data["Rates_averaged_c2"] = ""
    spike_data["Rates_sd_c2"] = ""


    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_c2[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_c2[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_c2[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_c2[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_c2[4], dtype= np.int32)
        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        # bin data over position bins
        bins = np.arange(0,200,1)
        trial_numbers = np.arange(min(trials),max(trials), 1)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                type_in_position = int(trial_data[0,3])
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                    average_rates = np.nanmean(rate_in_position)
                    if type_in_position == 0 :
                        binned_data[bcount, tcount, 0] = average_rates
                    if (type_in_position == 1 or type_in_position == 2) :
                        binned_data[bcount, tcount, 1] = average_rates
                    if type_in_position == 2 :
                        binned_data[bcount, tcount, 2] = average_rates

        #remove nans interpolate
        data_b = pd.DataFrame(binned_data[:,:,0], dtype=None, copy=False)
        data_b = data_b.dropna(axis = 1, how = "all")
        data_b.reset_index(drop=True, inplace=True)
        data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = np.asarray(data_b)
        x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Rates_averaged_c2'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'Rates_sd_c2'] = list(x_sd)

    return spike_data


def plot_c2_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_c2'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_c2'])

        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

        plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
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
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off

        ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 4)
        ax.set_xticklabels(['10', '30', '50'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'context2.png', dpi=200)
        plt.close()

    return spike_data

