import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.plot_utility


"""

## Calculate space binned firing rate

The following functions load rewarded firing rate data binned in time (100 ms) an bins the data into 199 location bins and X number of trials

- load rewarded spike rates in time 
- bin in space (rows = bins, columns = trials)
- average across trials
- store in dataframe


"""

def extract_time_binned_firing_rate_rewarded(spike_data):
    spike_data["Rates_averaged_rewarded_b"] = ""
    spike_data["Rates_averaged_rewarded_nb"] = ""
    spike_data["Rates_averaged_rewarded_p"] = ""
    spike_data["Rates_sd_rewarded_b"] = ""
    spike_data["Rates_sd_rewarded_nb"] = ""
    spike_data["Rates_sd_rewarded_p"] = ""
    spike_data["Rates_bytrial_rewarded_b"] = ""
    spike_data["Rates_bytrial_rewarded_nb"] = ""
    spike_data["Rates_bytrial_rewarded_p"] = ""


    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:] # remove speed < 3 cm/s

        # bin data over position bins
        bins = np.arange(0.5,199.5,1)
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
        spike_data.at[cluster, 'Rates_bytrial_rewarded_b'] = list(data_b)
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_b'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'Rates_sd_rewarded_b'] = list(x_sd)

        #just probe trials
        data_p = pd.DataFrame(binned_data[:,:,2], dtype=None, copy=False)
        data_p = data_p.dropna(axis = 1, how = "all")
        data_p.reset_index(drop=True, inplace=True)
        data_p = data_p.interpolate(method='linear', limit=None, limit_direction='both')
        data_p = np.asarray(data_p)
        x = np.reshape(data_p, (data_p.shape[0]*data_p.shape[1]))
        spike_data.at[cluster, 'Rates_bytrial_rewarded_p'] = list(data_p)
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_p = np.reshape(x, (data_p.shape[0], data_p.shape[1]))
        x = np.nanmean(data_p, axis=1)
        x_sd = np.nanstd(data_p, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_p'] = list(x)
        spike_data.at[cluster, 'Rates_sd_rewarded_p'] = list(x_sd)


        data_nb = pd.DataFrame(binned_data[:,:,1], dtype=None, copy=False)
        data_nb = data_nb.dropna(axis = 1, how = "all")
        data_nb.reset_index(drop=True, inplace=True)
        data_nb = data_nb.interpolate(method='linear', limit=None, limit_direction='both')
        data_nb = np.asarray(data_nb)
        x = np.reshape(data_nb, (data_nb.shape[0]*data_nb.shape[1]))
        spike_data.at[cluster, 'Rates_bytrial_rewarded_nb'] = list(data_nb)
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_nb = np.reshape(x, (data_nb.shape[0], data_nb.shape[1]))
        x = np.nanmean(data_nb, axis=1)
        x_sd = np.nanstd(data_nb, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_nb'] = list(x)
        spike_data.at[cluster, 'Rates_sd_rewarded_nb'] = list(x_sd)

    return spike_data



"""

## Calculate space binned firing rate

The following functions load firing rate data binned in time (100 ms) an bins the data into 199 location bins and X number of trials

- load spike rates in time 
- bin in space (rows = bins, columns = trials)
- average across trials
- store in dataframe


"""


def extract_time_binned_firing_rate(spike_data):
    spike_data["Rates_bytrial"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:] # remove speed < 3 cm/s

        # bin data over position bins
        bins = np.arange(0.5,199.5,1)
        trial_numbers = np.arange(min(trials),max(trials), 1)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        beaconed_trials=[];nonbeaconed_trials=[];probe_trials=[]
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
                        beaconed_trials = np.append(beaconed_trials,trial)
                    if (type_in_position == 1 or type_in_position == 2) :
                        binned_data[bcount, tcount, 1] = average_rates
                        nonbeaconed_trials = np.append(nonbeaconed_trials,trial)
                    if type_in_position == 2 :
                        binned_data[bcount, tcount, 2] = average_rates
                        probe_trials = np.append(probe_trials,trial)


        #remove nans interpolate
        data_b = pd.DataFrame(binned_data[:,:,0], dtype=None, copy=False)
        data_b = data_b.dropna(axis = 1, how = "all")
        data_b.reset_index(drop=True, inplace=True)
        data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
        data_b = np.asarray(data_b)
        beaconed = np.reshape(data_b, (data_b.shape[1]*data_b.shape[0]), order='F')
        types = np.array(np.tile(int(0), len(beaconed)))
        beaconed_data = np.transpose(np.vstack((beaconed,beaconed_trials,types )))

        #just probe trials
        data_p = pd.DataFrame(binned_data[:,:,2], dtype=None, copy=False)
        data_p = data_p.dropna(axis = 1, how = "all")
        data_p.reset_index(drop=True, inplace=True)
        data_p = data_p.interpolate(method='linear', limit=None, limit_direction='both')
        data_p = np.asarray(data_p)
        probe = np.reshape(data_p, (data_p.shape[1]*data_p.shape[0]), order='F')
        types = np.array(np.tile(int(2), len(probe)))
        probe_data = np.transpose(np.vstack((probe,probe_trials,types )))

        data_nb = pd.DataFrame(binned_data[:,:,1], dtype=None, copy=False)
        data_nb = data_nb.dropna(axis = 1, how = "all")
        data_nb.reset_index(drop=True, inplace=True)
        data_nb = data_nb.interpolate(method='linear', limit=None, limit_direction='both')
        data_nb = np.asarray(data_nb)
        nonbeaconed = np.reshape(data_nb, (data_nb.shape[1]*data_nb.shape[0]), order='F')
        types = np.array(np.tile(int(1), len(nonbeaconed)))

        nonbeaconed_data = np.transpose(np.vstack((nonbeaconed,nonbeaconed_trials,types )))

        try:
            data = np.transpose(np.vstack((list(beaconed_data), list(probe_data), list(nonbeaconed_data))))
        except ValueError:
            data = np.transpose(np.vstack((list(beaconed_data), list(nonbeaconed_data))))


        spike_data.at[cluster, 'Rates_bytrial'] = list(data)

    return spike_data






"""

## The following functions plot averaged space binned firing rate

- plot beaconed firing rate
- plot nonbeaconed firing rate
- plot probe firing rate
"""


def plot_rewarded_rates(spike_data, prm):
    #spike_data = plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = plot_rewarded_nb_firing_rate(spike_data, prm)
    spike_data = plot_rewarded_p_firing_rate(spike_data, prm)

    return spike_data




def plot_rewarded_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_b'])

        if rates.size > 5:
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
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
            plt.close()

    return spike_data


def plot_rewarded_nb_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_b'])

        rates_nb=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_nb'])
        sd_rates_nb=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_nb'])


        if rates.size > 1:
            speed_histogram = plt.figure(figsize=(3.7,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            ax.plot(position_array,rates_nb, '-', color='Red')
            ax.fill_between(position_array, rates_nb-sd_rates_nb,rates_nb+sd_rates_nb, facecolor = 'Red', alpha = 0.2)

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
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_nb.png', dpi=200)
            plt.close()
    return spike_data



def plot_rewarded_p_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Rate_Maps2'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_b'])

        rates_nb=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_nb'])
        sd_rates_nb=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_nb'])

        rates_p=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_p'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_p'])

        if rates.size > 1:
            speed_histogram = plt.figure(figsize=(3.7,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            ax.plot(position_array,rates_nb, '-', color='Red')
            ax.fill_between(position_array, rates_nb-sd_rates_nb,rates_nb+sd_rates_nb, facecolor = 'Red', alpha = 0.2)

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
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_p.png', dpi=200)
            plt.close()
    return spike_data


