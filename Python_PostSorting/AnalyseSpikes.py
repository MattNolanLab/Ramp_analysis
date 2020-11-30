import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import math
from scipy import signal



def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def bin_data(data):
    bins = np.arange(0,200,1)
    bin_means = (np.histogram(data, bins, weights=data)[0] /
             np.histogram(data, bins)[0])
    return bin_means


def add_data_to_frame(spike_data):
    spike_data["Rates_averaged"] = ""
    spike_data["Rates_averaged_nb"] = ""
    spike_data["Rates_averaged_p"] = ""

    spike_data["Firing_rate_b"] = ""
    spike_data["Firing_rate_nb"] = ""
    spike_data["Firing_rate_p"] = ""

    return spike_data



def extract_time_binned_firing_rate_overtrial_per_trialtype(spike_data, prm):

    spike_data = add_data_to_frame(spike_data)

    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        # extract firing info for each cluster
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        try:
            rates = convolve_with_scipy(rates) # convolve spikes
        except TypeError:
            continue

        # stack data and filter for speeds < 3 cm/s
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data_speed_filtered = data[data[:,1] >= 3,:]

        # bin data over position bins
        bins = np.arange(0,200,1)
        trial_numbers = np.arange(min(trials),max(trials), 1)
        binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data[:, :, :] = np.nan
        binned_data_sd = np.zeros((bins.shape[0], trial_numbers.shape[0], 3)); binned_data_sd[:, :, :] = np.nan
        for tcount, trial in enumerate(trial_numbers):
            trial_data = data_speed_filtered[data_speed_filtered[:,4] == trial,:]
            if trial_data.shape[0] > 0:
                t_rates = trial_data[:,0]
                t_pos = trial_data[:,2]
                type_in_position = int(trial_data[0,3])
                for bcount, b in enumerate(bins):
                    rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                    average_rates = np.nanmean(rate_in_position)
                    binned_data[bcount, tcount, type_in_position] = average_rates

        # average for beaconed
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
        # add data to dataframe
        spike_data.at[cluster, 'Rates_averaged'] = list(avg_b_binned_data)
        spike_data.at[cluster, 'Firing_rate_b'] = data_b


        ### plot average spike rate for beaconed trials across whole track
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_b_binned_data, '-', color='Black')
        ax.fill_between(bins, avg_b_binned_data-sd_b_binned_data,avg_b_binned_data+sd_b_binned_data, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_beaconed_bytrial.png', dpi=200)
        plt.close()

        ## same as above for just the outbound zone
        avg_b_binned_data2 = avg_b_binned_data[20:100]
        bins_out = bins[20:100]
        sd_b_binned_data2 = sd_b_binned_data[20:100]

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins_out,avg_b_binned_data2, '-', color='Black')
        ax.fill_between(bins_out, avg_b_binned_data2-sd_b_binned_data2,avg_b_binned_data2+sd_b_binned_data2, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(20,100)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.axvspan(90, 100, facecolor='DarkGreen', alpha=.15, linewidth =0)
        ax.axvspan(20, 30, facecolor='k', linewidth =0, alpha=.15) # black box
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
        ax.set_xticklabels(['10', '20', '70'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_beaconed_bytrial_outbound.png', dpi=200)
        plt.close()

        data_p = pd.DataFrame(binned_data[:,:,2],dtype=None, copy=False)
        data_p = data_p.interpolate(method='linear', order=2)
        data_p = data_p.dropna(axis = 1, how = "all")
        data_p.reset_index(drop=True, inplace=True)
        data_p = np.asarray(data_p)

        avg_p_binned_data = np.nanmean(data_p, axis=1)
        sd_p_binned_data = np.nanstd(data_p, axis=1)
        spike_data.at[cluster, 'Rates_averaged_p'] = list(avg_p_binned_data)
        spike_data.at[cluster, 'Firing_rate_p'] = data_p

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_p_binned_data, '-', color='Black')
        ax.fill_between(bins, avg_p_binned_data-sd_p_binned_data,avg_p_binned_data+sd_p_binned_data, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_probe_bytrial.png', dpi=200)
        plt.close()

        # plot beaconed and probe together
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_b_binned_data, '-', color='Black')
        ax.fill_between(bins, avg_b_binned_data-sd_b_binned_data,avg_b_binned_data+sd_b_binned_data, facecolor = 'Black', alpha = 0.2)
        ax.plot(bins,avg_p_binned_data, '-', color='Blue')
        ax.fill_between(bins, avg_p_binned_data-sd_p_binned_data,avg_p_binned_data+sd_p_binned_data, facecolor = 'Blue', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_overlay_bytrial.png', dpi=200)
        plt.close()

        ## average over non-beaconed and probe trials
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
            continue
        data_nb = np.reshape(data_nb2, (data_nb.shape[0], data_nb.shape[1]))
        avg_both_binned_data = np.nanmean(data_nb, axis=1)
        sd_both_binned_data = np.nanstd(data_nb, axis=1)
        spike_data.at[cluster, 'Rates_averaged_nb'] = list(avg_both_binned_data)
        spike_data.at[cluster, 'Firing_rate_nb'] = data_nb

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_b_binned_data, '-', color='Black')
        ax.fill_between(bins, avg_b_binned_data-sd_b_binned_data,avg_b_binned_data+sd_b_binned_data, facecolor = 'Black', alpha = 0.2)
        ax.plot(bins,avg_both_binned_data, '-', color='Crimson')
        ax.fill_between(bins, avg_both_binned_data-sd_both_binned_data,avg_both_binned_data+sd_both_binned_data, facecolor = 'Crimson', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(20,100)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_both_bytrial_outbound.png', dpi=200)
        plt.close()

    print("finished plotting whole track rates for trial types")
    return spike_data
