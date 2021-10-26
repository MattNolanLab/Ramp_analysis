import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.plot_utility
import seaborn as sns


def extract_time_binned_firing_rate_rewarded(spike_data):
    #spike_data["Rates_averaged_rewarded_by_trial"] = ""
    #spike_data["Rates_averaged_rewarded_by_trial_uncued"] = ""
    spike_data["Avg_FR_beaconed_rewarded"] = ""
    spike_data["SD_FR_beaconed_rewarded"] = ""
    spike_data["Avg_FR_nonbeaconed_rewarded"] = ""
    spike_data["SD_FR_nonbeaconed_rewarded"] = ""
    spike_data["Avg_FR_probe_rewarded"] = ""
    spike_data["SD_FR_probe_rewarded"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4], dtype= np.int32)
        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 0.3,:]
        data = data[data[:,3] == 0,:]

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
        x_b = np.nanmean(data_b, axis=1)
        x_sd_b = np.nanstd(data_b, axis=1)
        #spike_data.at[cluster, 'Rates_averaged_rewarded_by_trial'] = list(data_b)# add data to dataframe
        spike_data.at[cluster, 'Avg_FR_beaconed_rewarded'] = list(x_b)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_beaconed_rewarded'] = list(x_sd_b)

        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 0.3,:]
        data = data[data[:,3] == 2,:]

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
        spike_data.at[cluster, 'Avg_FR_probe_rewarded'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_probe_rewarded'] = list(x_sd)

        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 0.3,:]
        data = data[data[:,3] != 0,:]

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
        spike_data.at[cluster, 'Avg_FR_nonbeaconed_rewarded'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_nonbeaconed_rewarded'] = list(x_sd)
        #spike_data.at[cluster, 'Rates_averaged_rewarded_by_trial_uncued'] = list(data_b)# add data to dataframe
    return spike_data



def calculate_trial_by_trial_peaks(spike_data):
    print("I am calculating trial by trial peaks...")
    spike_data["trial_peaks_max"] = ""
    spike_data["trial_peak_locations_max"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_by_trial'])
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

        spike_data.at[cluster, 'trial_peaks_max'] = list(peak_array)# add data to dataframe
        spike_data.at[cluster, 'trial_peak_locations_max'] = list(peak_array_locs)# add data to dataframe

    return spike_data


def calculate_trial_by_trial_minpeaks(spike_data):
    print("I am calculating trial by trial peaks...")
    spike_data["trial_peaks_min"] = ""
    spike_data["trial_peak_locations_min"] = ""

    for cluster in range(len(spike_data)):
        cluster_data = np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_by_trial'])
        peak_array = []
        peak_array_locs = []
        trials = (cluster_data.shape[1])-1
        for trialcount, trial in enumerate(range(trials)):
            trial_data = cluster_data[:,trialcount]
            region_of_interest = trial_data[65:100]
            peak_location = np.argmin(region_of_interest)+65
            peak_firingrate = np.min(region_of_interest)
            peak_array = np.append(peak_array, peak_firingrate )
            peak_array_locs = np.append(peak_array_locs, peak_location )

        spike_data.at[cluster, 'trial_peaks_min'] = list(peak_array)# add data to dataframe
        spike_data.at[cluster, 'trial_peak_locations_min'] = list(peak_array_locs)# add data to dataframe

    return spike_data


def plot_heatmap_by_trial(spike_data, prm):
    print("I am plotting firing rate for some trials...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/heatmaps_per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_by_trial'])

        speed_histogram = plt.figure(figsize=(5,12))
        ax = sns.heatmap(np.transpose(rates))
        plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        plt.xlim(0,200)
        ax.axvline(90, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axvline(110, linewidth = 1.5, color = 'black') # bold line on the y axis
        plt.locator_params(axis = 'x', nbins  = 4)
        plt.savefig(save_path + '/heatmap_normal_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
        plt.close()

    return spike_data


def plot_heatmap_by_trial_uncued(spike_data, prm):
    print("I am plotting firing rate for some trials...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/heatmaps_per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_by_trial_uncued'])
        try:
            speed_histogram = plt.figure(figsize=(5,12))
            ax = sns.heatmap(np.transpose(rates))
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.axvline(90, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axvline(110, linewidth = 1.5, color = 'black') # bold line on the y axis
            plt.locator_params(axis = 'x', nbins  = 4)
            plt.savefig(save_path + '/heatmap_normal_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_uncued.png', dpi=200)
            plt.close()
        except ValueError:
            continue

    return spike_data


def plot_rewarded_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed_rewarded'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
        plt.close()

    return spike_data



def plot_rewarded_firing_rate_probe(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'Avg_FR_probe_rewarded'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'SD_FR_probe_rewarded'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,rates_p, '-', color='Blue')
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'Blue', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 4)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-5', '20', '45', '70'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_probe.png', dpi=200)
        plt.close()

    return spike_data


def plot_rewarded_firing_rate_probe_outbound(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'Avg_FR_probe_rewarded'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'SD_FR_probe_rewarded'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,rates_p, '-', color='Red')
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'Red', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(19,101)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 4)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-5', '20', '45', '70'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_probe_outbound.png', dpi=200)
        plt.close()

    return spike_data


def plot_rewarded_firing_rate_nonbeaconed_outbound(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'Avg_FR_nonbeaconed_rewarded'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,rates_p, '-', color='Red')
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'Red', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(20,100)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 4)
        plt.locator_params(axis = 'y', nbins  = 3)
        #ax.set_xticklabels(['0', '25', '50'])
        plt.xticks([30, 55, 80], ['0', '25', '50'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_nb_outbound.png', dpi=200)
        plt.close()

    return spike_data


## Rewarded and failed trials
## Beaconed trials and probe trials

def extract_time_binned_firing_rate(spike_data, prm):
    spike_data["Rates_averaged_rewarded_by_trial"] = ""
    spike_data["Avg_FR_beaconed"] = ""
    spike_data["SD_FR_beaconed"] = ""
    spike_data["Avg_FR_nonbeaconed"] = ""
    spike_data["SD_FR_nonbeaconed"] = ""
    spike_data["Avg_FR_probe"] = ""
    spike_data["SD_FR_probe"] = ""

    for cluster in range(len(spike_data)):
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]
        data = data[data[:,3] == 0,:]

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
        #data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = data_b.interpolate(method='pad')
        data_b = np.asarray(data_b)
        x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x_b = np.nanmean(data_b, axis=1)
        x_sd_b = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_by_trial'] = list(data_b)# add data to dataframe
        spike_data.at[cluster, 'Avg_FR_beaconed'] = list(x_b)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_beaconed'] = list(x_sd_b)

        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]
        data = data[data[:,3] == 2,:]

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
        #data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = data_b.interpolate(method='pad')
        data_b = np.asarray(data_b)
        x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Avg_FR_probe'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_probe'] = list(x_sd)


        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]
        data = data[data[:,3] != 0,:]

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
        #data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = data_b.interpolate(method='pad')
        data_b = np.asarray(data_b)
        x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Avg_FR_nonbeaconed'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'SD_FR_nonbeaconed'] = list(x_sd)
    return spike_data







def plot_firing_rate_probe(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed'])
        rates_p=np.array(spike_data.loc[cluster, 'Avg_FR_probe'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'SD_FR_probe'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,rates_p, '-', color='Blue')
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'Blue', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_probe.png', dpi=200)
        plt.close()

    return spike_data







def plot_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Avg_FR_timebinned'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_beaconed'])

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black')
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_beaconed.png', dpi=200)
        plt.close()

    return spike_data



def find_peak(df):
    region_of_interest = df[60:110]
    max_loc = np.argmax(region_of_interest)
    max_rate = np.max(region_of_interest)
    return max_loc+60,max_rate


def find_min_peak(df):
    region_of_interest = df[60:110]
    max_loc = np.argmin(region_of_interest)
    max_rate = np.min(region_of_interest)
    return max_loc+60,max_rate




def plot_aligned_data(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        peak_max_norm, amp_max_norm = find_peak(rates)
        peak_max_p, amp_max_p = find_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm+5, '*', color='Black', markersize=4)

        ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        ax.plot(peak_max_p,amp_max_p+5, '*', color='royalblue', markersize=4)
        ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned.png', dpi=200)
        plt.close()

    return spike_data




def plot_aligned_data_with_min(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        peak_max_norm, amp_max_norm = find_min_peak(rates)
        peak_max_p, amp_max_p = find_min_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm-5, '*', color='Black', markersize=4)

        ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        ax.plot(peak_max_p,amp_max_p-5, '*', color='royalblue', markersize=4)
        ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_min.png', dpi=200)
        plt.close()

    return spike_data





def plot_aligned_fs_data(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_FS'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_FS'])

        peak_max_norm, amp_max_norm = find_peak(rates)
        peak_max_p, amp_max_p = find_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm+5, '*', color='Black', markersize=4)

        ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        ax.plot(peak_max_p,amp_max_p+5, '*', color='royalblue', markersize=4)
        ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_max_FS.png', dpi=200)
        plt.close()

    return spike_data




def plot_aligned_fs_data_with_min(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_FS'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_FS'])

        peak_max_norm, amp_max_norm = find_min_peak(rates)
        peak_max_p, amp_max_p = find_min_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm-5, '*', color='Black', markersize=4)

        ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        ax.plot(peak_max_p,amp_max_p-5, '*', color='royalblue', markersize=4)
        ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_min_FS.png', dpi=200)
        plt.close()

    return spike_data




def plot_aligned_norm_data_alone_with_peak(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        #rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        #sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        peak_max_norm, amp_max_norm = find_peak(rates)
        #peak_max_p, amp_max_p = find_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm+5, '*', color='Black', markersize=4)

        #ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        #ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        #ax.plot(peak_max_p,amp_max_p-5, '*', color='royalblue', markersize=4)
        #ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_norm.png', dpi=200)
        plt.close()

    return spike_data





def plot_aligned_reward_data_alone_with_peak(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        #rates=np.array(spike_data.loc[cluster, 'Avg_FR_beaconed_rewarded'])
        #sd_rates=np.array(spike_data.loc[cluster, 'SD_FR_nonbeaconed_rewarded'])
        rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        #peak_max_norm, amp_max_norm = find_peak(rates)
        peak_max_p, amp_max_p = find_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        #ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        #ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        #ax.plot(peak_max_norm,amp_max_norm-5, '*', color='Black', markersize=4)

        ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        ax.plot(peak_max_p,amp_max_p+5, '*', color='royalblue', markersize=4)
        ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_reward.png', dpi=200)
        plt.close()

    return spike_data





def plot_aligned_fs_data_alone_with_peak(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Aligned_examples'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'FR_reset_at_FS'])
        sd_rates=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_FS'])
        #rates_p=np.array(spike_data.loc[cluster, 'FR_reset_at_reward'])
        #sd_rates_p=np.array(spike_data.loc[cluster, 'FR_sd_reset_at_reward'])

        peak_max_norm, amp_max_norm = find_peak(rates)
        #peak_max_p, amp_max_p = find_peak(rates_p)

        speed_histogram = plt.figure(figsize=(3.7,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,rates, '-', color='Black', alpha=0.6, linewidth=1.5)
        ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.1)
        ax.plot(peak_max_norm,amp_max_norm+5, '*', color='Black', markersize=4)

        #ax.plot(position_array,rates_p, '-', color='royalblue', alpha=0.9, linewidth=1.5)
        #ax.fill_between(position_array, rates_p-sd_rates_p,rates_p+sd_rates_p, facecolor = 'royalblue', alpha = 0.1)
        #ax.plot(peak_max_p,amp_max_p-5, '*', color='royalblue', markersize=4)
        #ax.axvline(100, linewidth = 1, color = 'Red') # bold line on the y axis

        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_aligned_fs.png', dpi=200)
        plt.close()

    return spike_data


