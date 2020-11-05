import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from scipy import signal
import Python_PostSorting.plot_utility

def convolve_with_scipy(rate):
    window = signal.gaussian(3, std=5)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def extract_time_binned_firing_rate_rewarded(spike_data, prm):
    spike_data["Rates_averaged_rewarded_b"] = ""
    spike_data["Rates_averaged_rewarded_nb"] = ""
    spike_data["Rates_averaged_rewarded_p"] = ""
    spike_data["Rates_sd_rewarded_b"] = ""
    spike_data["Rates_sd_rewarded_nb"] = ""
    spike_data["Rates_sd_rewarded_p"] = ""
    spike_data["Firing_rate_rewarded_b"] = ""
    spike_data["Firing_rate_rewarded_nb"] = ""
    spike_data["Firing_rate_rewarded_p"] = ""

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        speed=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[4], dtype= np.int32)

        try:
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
            #speed = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(speed)
        except TypeError:
            continue

        # stack data and filter for speeds < 3 cm/s
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data_speed_filtered = data[data[:,1] >= 3,:]
        try:
            # bin data over position bins
            bins = np.linspace(start=np.nanmin(position),stop=np.nanmax(position),num=200)
            bin_unit = (np.nanmax(position) - np.nanmin(position))/200
            trial_numbers = np.arange(np.nanmin(trials),np.nanmax(trials), 1)
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
            binned_data_sd = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data_sd[:, :, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data_speed_filtered[data_speed_filtered[:,4] == trial,:]
                if trial_data.shape[0] > 0:
                    t_rates = trial_data[:,0]
                    t_pos = trial_data[:,2]
                    type_in_position = int(trial_data[0,3])
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= b, t_pos < b+bin_unit)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount, type_in_position] = average_rates

            data_b = pd.DataFrame(binned_data[:,:,0],dtype=None, copy=False)
            data_b = data_b.interpolate(method='linear', order=2)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = np.asarray(data_b)
            data_b2 = np.reshape(data_b, data_b.shape[0]*data_b.shape[1])
            data_b2 = convolve_with_scipy(data_b2) # convolve with guassian kernal
            data_b = np.reshape(data_b2, (data_b.shape[0], data_b.shape[1]))
            avg_b_binned_data = np.nanmean(data_b, axis=1)
            sd_b_binned_data = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'Rates_averaged_rewarded_b'] = list(avg_b_binned_data) # add data to dataframe
            spike_data.at[cluster, 'Rates_sd_rewarded_b'] = list(sd_b_binned_data)
            spike_data.at[cluster, 'Firing_rate_rewarded_b'] = data_b

            binned_data_both = np.nanmean(binned_data[:,:,1:], axis=2)
            data_nb = pd.DataFrame(binned_data_both,dtype=None, copy=False)
            data_nb = data_nb.interpolate(method='linear', order=2)
            data_nb = data_nb.dropna(axis = 1, how = "all")
            data_nb.reset_index(drop=True, inplace=True)
            data_nb = np.asarray(data_nb)
            data_nb2 = np.reshape(data_nb, data_nb.shape[0]*data_nb.shape[1])
            try:
                data_nb2 = convolve_with_scipy(data_nb2) # convolve with guassian kernal
            except ValueError:
                print("convolve error")
            data_nb = np.reshape(data_nb2, (data_nb.shape[0], data_nb.shape[1]))
            avg_both_binned_data = np.nanmean(data_nb, axis=1)
            sd_both_binned_data = np.nanstd(data_nb, axis=1)
            spike_data.at[cluster, 'Rates_averaged_rewarded_nb'] = list(avg_both_binned_data)
            spike_data.at[cluster, 'Rates_sd_rewarded_nb'] = list(sd_both_binned_data)
            spike_data.at[cluster, 'Firing_rate_rewarded_nb'] = data_nb

            #just probe trials
            data_p = pd.DataFrame(binned_data[:,:,2],dtype=None, copy=False)
            data_p = data_p.interpolate(method='linear', order=2)
            data_p = data_p.dropna(axis = 1, how = "all")
            data_p.reset_index(drop=True, inplace=True)
            data_p = np.asarray(data_p)
            data_p2 = np.reshape(data_p, data_p.shape[0]*data_p.shape[1])
            try:
                data_p2 = convolve_with_scipy(data_p2) # convolve with guassian kernal
            except ValueError:
                print("convolve error")
            data_p = np.reshape(data_p2, (data_p.shape[0], data_p.shape[1]))
            avg_p_binned_data = np.nanmean(data_p, axis=1)
            sd_p_binned_data = np.nanstd(data_p, axis=1)
            spike_data.at[cluster, 'Rates_averaged_rewarded_p'] = list(sd_p_binned_data)
            spike_data.at[cluster, 'Rates_sd_rewarded_p'] = list(avg_p_binned_data)
            spike_data.at[cluster, 'Firing_rate_rewarded_p'] = data_p
        except ValueError:
            continue
    return spike_data



def plot_rewarded_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_b'])

        try:
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

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
            plt.locator_params(axis = 'x', nbins  = 4)
            ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded.png', dpi=200)
            plt.close()
        except ValueError:
            continue
    return spike_data



def plot_rewarded_nb_firing_rate(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates_rewarded'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)
        rates=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_b'])
        sd_rates=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_b'])

        rates_nb=np.array(spike_data.loc[cluster, 'Rates_averaged_rewarded_nb'])
        sd_rates_nb=np.array(spike_data.loc[cluster, 'Rates_sd_rewarded_nb'])

        try:
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            ax.plot(position_array,rates_nb, '-', color='Red')
            ax.fill_between(position_array, rates_nb-sd_rates_nb,rates_nb+sd_rates_nb, facecolor = 'Red', alpha = 0.2)

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
            plt.locator_params(axis = 'x', nbins  = 4)
            ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'rewarded_nb.png', dpi=200)
            plt.close()
        except ValueError:
            continue

    return spike_data



def plot_rewarded_rates(spike_data, prm):
    spike_data = plot_rewarded_firing_rate(spike_data, prm)
    spike_data = plot_rewarded_nb_firing_rate(spike_data, prm)
    return spike_data
