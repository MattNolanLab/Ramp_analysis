import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.plot_utility


def extract_time_binned_firing_rate_rewarded(spike_data):
    spike_data["Rates_averaged_rewarded_b"] = ""
    spike_data["Rates_averaged_rewarded_nb"] = ""
    spike_data["Rates_averaged_rewarded_p"] = ""
    spike_data["Rates_sd_rewarded_b"] = ""
    spike_data["Rates_sd_rewarded_nb"] = ""
    spike_data["Rates_sd_rewarded_p"] = ""


    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)
        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        # bin data over position bins
        bins = np.arange(0.5,199.5,1)
        trial_numbers = np.unique(trials)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                type_in_position = int(trial_data[0,3])
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= b, t_pos < b+1)))
                    average_rates = np.nanmean(rate_in_position)
                    if type_in_position == 0 :
                        binned_data[bcount, tcount, 0] = average_rates
                    if type_in_position == 2 :
                        binned_data[bcount, tcount, 2] = average_rates
                    if (type_in_position == 1 or type_in_position == 2) :
                        binned_data[bcount, tcount, 1] = average_rates


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
        spike_data.at[cluster, 'Rates_averaged_rewarded_b'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'Rates_sd_rewarded_b'] = list(x_sd)

        #just probe trials
        data_p = pd.DataFrame(binned_data[:,:,2], dtype=None, copy=False)
        data_p = data_p.dropna(axis = 1, how = "all")
        data_p.reset_index(drop=True, inplace=True)
        data_p = data_p.interpolate(method='linear', limit=None, limit_direction='both')
        data_p = np.asarray(data_p)
        x = np.reshape(data_p, (data_p.shape[0]*data_p.shape[1]))
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
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_nb = np.reshape(x, (data_nb.shape[0], data_nb.shape[1]))
        x = np.nanmean(data_nb, axis=1)
        x_sd = np.nanstd(data_nb, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_nb'] = list(x)
        spike_data.at[cluster, 'Rates_sd_rewarded_nb'] = list(x_sd)
    return spike_data




def extract_time_binned_firing_rate_allspeeds(spike_data):
    spike_data["Rates_averaged_rewarded_b"] = ""
    spike_data["Rates_averaged_rewarded_nb"] = ""
    spike_data["Rates_averaged_rewarded_p"] = ""
    spike_data["Rates_sd_rewarded_b"] = ""
    spike_data["Rates_sd_rewarded_nb"] = ""
    spike_data["Rates_sd_rewarded_p"] = ""


    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)


        window = signal.gaussian(2, std=3)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        # bin data over position bins
        bins = np.arange(0.5,199.5,1)
        trial_numbers = np.unique(trials)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                type_in_position = int(trial_data[0,3])
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= b, t_pos < b+1)))
                    average_rates = np.nanmean(rate_in_position)
                    if type_in_position == 0 :
                        binned_data[bcount, tcount, 0] = average_rates
                    if type_in_position == 2 :
                        binned_data[bcount, tcount, 2] = average_rates
                    if (type_in_position == 1) :
                        binned_data[bcount, tcount, 1] = average_rates


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
        spike_data.at[cluster, 'Rates_averaged_rewarded_b'] = list(x)# add data to dataframe
        spike_data.at[cluster, 'Rates_sd_rewarded_b'] = list(x_sd)

        #just probe trials
        data_p = pd.DataFrame(binned_data[:,:,2], dtype=None, copy=False)
        data_p = data_p.dropna(axis = 1, how = "all")
        data_p.reset_index(drop=True, inplace=True)
        data_p = data_p.interpolate(method='linear', limit=None, limit_direction='both')
        data_p = np.asarray(data_p)
        x = np.reshape(data_p, (data_p.shape[0]*data_p.shape[1]))
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
        x = signal.convolve(x, window, mode='same')/ sum(window)
        data_nb = np.reshape(x, (data_nb.shape[0], data_nb.shape[1]))
        x = np.nanmean(data_nb, axis=1)
        x_sd = np.nanstd(data_nb, axis=1)
        spike_data.at[cluster, 'Rates_averaged_rewarded_nb'] = list(x)
        spike_data.at[cluster, 'Rates_sd_rewarded_nb'] = list(x_sd)
    return spike_data



def extract_time_binned_firing_rate_unsmoothed(spike_data, prm):
    spike_data["data_set"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        #data = data[data[:,1] >= 3,:]

        # bin data over position bins
        max_position = np.nanmax(position)
        min_position = np.nanmin(position)
        bin_size =  (max_position - min_position)/199
        bins = np.arange(min_position, max_position, step=bin_size)
        trial_numbers = np.unique(trials)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data[data[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= b, t_pos < b+bin_size)))
                    average_rates = np.nanmean(rate_in_position)
                    binned_data[bcount, tcount, 0] = average_rates


        #remove nans interpolate
        data_b = pd.DataFrame(binned_data[:,:,0], dtype=None, copy=False)
        data_b = data_b.dropna(axis = 1, how = "all")
        data_b.reset_index(drop=True, inplace=True)

        data_b = np.asarray(data_b)
        rates = np.transpose(data_b)
        rates = rates.flatten()
        rates = pd.Series(rates)
        rates = rates.interpolate(method='linear')
        rates = np.asarray(rates)
        save_data = np.reshape(rates, (data_b.shape[0], data_b.shape[1]), order='F')

        x = np.nanmean(data_b, axis=1)
        x_sd = np.nanstd(data_b, axis=1)

        spike_data.at[cluster, 'data_set'] = list(save_data)

        save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        if rates.size > 5:
            cluster_index = spike_data.cluster_id.values[cluster] - 1

            speed_histogram = plt.figure(figsize=(3.7,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(bins,x, '-', color='Black')
            ax.fill_between(bins, x-x_sd,x+x_sd, facecolor = 'Black', alpha = 0.2)
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
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.set_xticklabels(['-30', '70', '170'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'unsmoothed2.png', dpi=200)
            plt.close()

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


def plot_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_b'])

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
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_all.png', dpi=200)
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



def plot_nb_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_b'])

        rates_nb=np.array(spike_data.loc[cluster, 'Rates_averaged_nb'])
        sd_rates_nb=np.array(spike_data.loc[cluster, 'Rates_sd_nb'])


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
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_all_nb.png', dpi=200)
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



def plot_p_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Firing_Rate_Maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,199,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_b'])

        rates_nb=np.array(spike_data.loc[cluster, 'Rates_averaged_nb'])
        sd_rates_nb=np.array(spike_data.loc[cluster, 'Rates_sd_nb'])

        rates_p=np.array(spike_data.loc[cluster, 'Rates_averaged_p'])
        sd_rates_p=np.array(spike_data.loc[cluster, 'Rates_sd_p'])

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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_all_p.png', dpi=200)
        plt.close()

    return spike_data



def plot_rewarded_rates(spike_data, prm):
    #spike_data = plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = plot_rewarded_nb_firing_rate(spike_data, prm)
    spike_data = plot_rewarded_p_firing_rate(spike_data, prm)

    #spike_data = plot_firing_rate(spike_data, prm)
    #spike_data = plot_nb_firing_rate(spike_data, prm)
    #spike_data = plot_p_firing_rate(spike_data, prm)
    return spike_data





def plot_tiny_raw2(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates=np.array(spike_data.loc[cluster, 'data_set'])
        data_b = np.asarray(rates)
        rates1 = np.transpose(data_b)
        rates = rates1.flatten()

        bins = 199
        trial_data = rates[:bins*10]
        # plot raw
        avg_spikes_on_track = plt.figure(figsize=(15,15)) # width, height?
        ax = avg_spikes_on_track.add_subplot(10, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
        ymax= np.nanmax(rates)
        ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Hz', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

        trial_data = rates[bins*10:bins*20]
        ax = avg_spikes_on_track.add_subplot(10, 1, 2)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
        ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Hz', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])

        try:
            trial_data = rates[bins*20:bins*30]
            ax = avg_spikes_on_track.add_subplot(10, 1, 3)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*40:bins*50]
            ax = avg_spikes_on_track.add_subplot(10, 1, 4)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*50:bins*60]
            ax = avg_spikes_on_track.add_subplot(10, 1, 5)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*60:bins*70]
            ax = avg_spikes_on_track.add_subplot(10, 1, 6)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*70:bins*80]
            ax = avg_spikes_on_track.add_subplot(10, 1, 7)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*80:bins*90]
            ax = avg_spikes_on_track.add_subplot(10, 1, 8)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        try:
            trial_data = rates[bins*90:bins*100]
            ax = avg_spikes_on_track.add_subplot(10, 1, 9)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data, 'o', color='red', alpha=0.5, markersize=0.5)
            ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        except(IndexError, ValueError):
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
            plt.close()
            continue

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid_allspeed.png', dpi=200)
        plt.close()

    return spike_data




def plot_tiny_raw(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates1=np.array(spike_data.loc[cluster, 'data_set'])
        rates1 = np.transpose(rates1)
        rates = rates1.flatten()

        position = np.arange(0,len(rates),1)
        data_b = np.asarray(rates[:len(position)])
        #rates = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))

        bins = 199
        avg_spikes_on_track = plt.figure(figsize=(20,15)) # width, height?
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position,data_b, 'o', color='red', alpha=0.5, markersize=0.5)
        ymax= np.nanmax(rates)
        #ax.vlines([199,398,597, 796, 995, 1194,1393,1592,1791], 0,ymax, color='black', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Hz', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_grid.png', dpi=200)
        plt.close()
    return spike_data


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

