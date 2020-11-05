import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import math
from scipy import signal
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.linear_model import LinearRegression

### ----------------------------------------------------------------------------------------- ###


def extract_time_binned_firing_rate(spike_data, prm):
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0])*10
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4], dtype= np.int32)

        try:
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
        except TypeError:
            continue

        data = np.vstack((rates,speed,position,types))
        data=data.transpose()
        data_filtered = data[data[:,1] > 3,:]

        data_filtered = data_filtered[data_filtered[:,3] == 0,:]
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
        binned_speed_sd = convolve_with_scipy(binned_speed_sd)

        ##print('plotting speed histogram...', cluster)
        save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Position (cm)', fontsize=16, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
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
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

        #probe
        data_filtered = data[data[:,1] > 3,:]
        data_filtered = data_filtered[data_filtered[:,3] == 2,:]
        rates = data_filtered[:,0]
        position = data_filtered[:,2]

        position_array = np.arange(1,201,1)
        binned_speed_nb = np.zeros((position_array.shape))
        binned_speed_sd_nb = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            speed_in_position = np.take(rates, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
            average_speed = np.nanmean(speed_in_position)
            sd_speed = np.nanstd(speed_in_position)
            binned_speed_nb[rowcount] = average_speed
            binned_speed_sd_nb[rowcount] = sd_speed

        ##print('plotting speed histogram...', cluster)
        save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        binned_speed_nb = convolve_with_scipy(binned_speed_nb)
        binned_speed_sd_nb = convolve_with_scipy(binned_speed_sd_nb)

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,binned_speed_nb, '-', color='blue', alpha=0.5)
        ax.fill_between(position_array, binned_speed_nb-binned_speed_sd_nb,binned_speed_nb+binned_speed_sd_nb, facecolor = 'blue', alpha = 0.2)

        plt.ylabel('Rates (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Position (cm)', fontsize=16, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
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

        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_wholetrack.png', dpi=200)
        plt.close()
    return spike_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    #plt.plot(window)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def extract_time_binned_firing_rate_per_trialtype(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    spike_data["Rates_averaged"] = ""
    spike_data["Rates_averaged_nb"] = ""
    spike_data["Rates_averaged_p"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        data = np.vstack((rates,speed,position,types))
        data=data.transpose()
        data_speed_filtered = data[data[:,1] > 3,:]

        #filter for beaconed trials
        data_filtered = data_speed_filtered[data_speed_filtered[:,3] == 0,:]
        rates_b = data_filtered[:,0]
        position_b = data_filtered[:,2]

        #plot raw - without convolving
        position_array = np.arange(1,201,1)
        binned_speed_raw = np.zeros((position_array.shape)); binned_speed_raw[:] = np.nan
        binned_speed_raw_sd = np.zeros((position_array.shape)); binned_speed_raw_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_b, np.where(np.logical_and(position_b >= rowcount, position_b < rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed_raw[rowcount] = average_rates
            binned_speed_raw_sd[rowcount] = sd_speed
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed_raw, '-', color='Black')
        ax.fill_between(position_array, binned_speed_raw-binned_speed_raw_sd,binned_speed_raw+binned_speed_raw_sd, facecolor = 'Black', alpha = 0.2)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_beaconed_raw.png', dpi=200)
        plt.close()

        #just beaconed trials
        try:
            rates_b = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates_b)
        except TypeError:
            continue
        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape)); binned_speed[:] = np.nan
        binned_speed_sd = np.zeros((position_array.shape)); binned_speed_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_b, np.where(np.logical_and(position_b >= rowcount, position_b < rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed[rowcount] = average_rates
            binned_speed_sd[rowcount] = sd_speed
        spike_data.at[cluster, 'Rates_averaged'] = list(binned_speed)

        #just probe trials
        data_filtered = data_speed_filtered[data_speed_filtered[:,3] == 2,:]
        rates_p = data_filtered[:,0]
        position_p = data_filtered[:,2]
        try:
            rates_p = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates_p)
        except TypeError:
            continue
        binned_speed_p = np.zeros((position_array.shape)); binned_speed_p[:] = np.nan
        binned_speed_p_sd = np.zeros((position_array.shape)); binned_speed_p_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_p, np.where(np.logical_and(position_p >= rowcount, position_p < rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed_p[rowcount] = average_rates
            binned_speed_p_sd[rowcount] = sd_speed
        spike_data.at[cluster, 'Rates_averaged_p'] = list(binned_speed_p)

        #both non-beaconed and probe trials
        data_filtered = data_speed_filtered[data_speed_filtered[:,3] != 0,:]
        rates_both = data_filtered[:,0]
        position_both = data_filtered[:,2]
        try:
            rates_both = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates_both)
        except TypeError:
            continue
        binned_speed_both = np.zeros((position_array.shape)); binned_speed_both[:] = np.nan
        binned_speed_both_sd = np.zeros((position_array.shape)); binned_speed_both_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_both, np.where(np.logical_and(position_both >= rowcount, position_both < rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed_both[rowcount] = average_rates
            binned_speed_both_sd[rowcount] = sd_speed
        spike_data.at[cluster, 'Rates_averaged_nb'] = list(binned_speed_both)

        binned_speed = convolve_with_scipy(binned_speed)
        binned_speed_sd = convolve_with_scipy(binned_speed_sd)
        binned_speed_p = convolve_with_scipy(binned_speed_p)
        binned_speed_p_sd = convolve_with_scipy(binned_speed_p_sd)
        binned_speed_both = convolve_with_scipy(binned_speed_both)
        binned_speed_both_sd = convolve_with_scipy(binned_speed_both_sd)

        #beaconed trials
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_beaconed.png', dpi=200)
        plt.close()

        binned_speed_p = pd.Series(binned_speed_p).interpolate(method='linear', order=1)
        binned_speed_p_sd = pd.Series(binned_speed_p_sd).interpolate(method='linear', order=1)
        #probe trials
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed_p, '-', color='Blue')
        ax.fill_between(position_array, binned_speed_p-binned_speed_p_sd,binned_speed_p+binned_speed_p_sd, facecolor = 'Blue', alpha = 0.1)
        ax.plot(position_array,binned_speed, '-', color='Black', alpha=0.5)
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
            labelsize=12,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_probe.png', dpi=200)
        plt.close()

        #probe and nonbeaconed trials
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed_both, '-', color='Black')
        ax.fill_between(position_array, binned_speed_both-binned_speed_both_sd,binned_speed_both+binned_speed_both_sd, facecolor = 'Black', alpha = 0.2)
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
            labelsize=12,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_both.png', dpi=200)
        plt.close()

    print("finished plotting whole track rates for trial types")
    return spike_data


def extract_time_binned_firing_rate_per_trialtype_shuffled(spike_data, prm):
    spike_data["Shuffled_Rates_averaged"] = ""
    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].shuffled_spikes_in_time[0].real)*10
        position=np.array(spike_data.iloc[cluster].shuffled_spikes_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].shuffled_spikes_in_time[4].real, dtype= np.int32)

        data = np.vstack((rates,position, types))
        data=data.transpose()
        data_filtered = data[data[:,2] == 0,:]
        rates_b = data_filtered[:,0]
        position_b = data_filtered[:,1]

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape))
        binned_speed_sd = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_b, np.where(np.logical_and(position_b >= rowcount, position_b <= rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed[rowcount] = average_rates
            binned_speed_sd[rowcount] = sd_speed
        spike_data.at[cluster, 'Shuffled_Rates_averaged'] = list(binned_speed)
    return spike_data


def remove_speed_outliers(rates,speed, position, types, trials):
    # remove outliers
    rates_o =  pd.Series(rates)
    speed_o =  pd.Series(speed)
    position_o =  pd.Series(position)
    types_o =  pd.Series(types)
    trials_o =  pd.Series(trials)

    rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
    speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
    position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
    types = types_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
    trials = trials_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers

    return rates,speed, position, types, trials


def extract_time_binned_firing_rate_per_trialtype_outbound(spike_data, prm):
    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        # filter data
        try:
           rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
        except TypeError:
                continue

        data = np.vstack((rates, position, types, speed))
        data=data.transpose()
        data_filtered = data[data[:,2] == 0,:]
        data_filtered = data_filtered[data_filtered[:,3] > 3,:]
        rates_b = data_filtered[:,0]
        position_b = data_filtered[:,1]

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape)); binned_speed[:] = np.nan
        binned_speed_sd = np.zeros((position_array.shape)); binned_speed_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_b, np.where(np.logical_and(position_b >= rowcount, position_b <= rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed[rowcount] = average_rates
            binned_speed_sd[rowcount] = sd_speed

        data_filtered = data[data[:,2] != 0,:]
        data_filtered = data_filtered[data_filtered[:,3] > 3,:]
        rates_nb = data_filtered[:,0]
        position_nb = data_filtered[:,1]

        binned_speed_nb = np.zeros((position_array.shape)); binned_speed_nb[:] = np.nan
        binned_speed_nb_sd = np.zeros((position_array.shape)); binned_speed_nb_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_nb, np.where(np.logical_and(position_nb >= rowcount, position_nb <= rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed_nb[rowcount] = average_rates
            binned_speed_nb_sd[rowcount] = sd_speed

        ##print('plotting histogram...', cluster)
        save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        binned_speed = convolve_with_scipy(binned_speed)
        binned_speed_sd = convolve_with_scipy(binned_speed_sd)
        binned_speed_nb = convolve_with_scipy(binned_speed_nb)
        binned_speed_nb_sd = convolve_with_scipy(binned_speed_nb_sd)

        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #plot beaconed and non beaconed on same plot
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,binned_speed_nb, '-', color='Red')
        ax.fill_between(position_array, binned_speed_nb-binned_speed_nb_sd,binned_speed_nb+binned_speed_nb_sd, facecolor = 'Red', alpha = 0.2)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
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
        plt.xlim(20,100)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_nonbeaconed_outbound.png', dpi=200)
        plt.close()

        binned_speed = binned_speed[20:100]
        position_array = position_array[20:100]
        binned_speed_sd = binned_speed_sd[20:100]

        #plot beaconed alone
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.axvspan(90, 100, facecolor='DarkGreen', alpha=.15, linewidth =0)
        ax.axvspan(0, 10, facecolor='k', linewidth =0, alpha=.15) # black box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
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
        ax.set_ylim(0)
        #plt.xlim(20,100)
        plt.locator_params(axis = 'x', nbins  = 3)
        #ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_outbound.png', dpi=200)
        plt.close()
    return spike_data


def extract_time_binned_firing_rate_per_trialtype_probe(spike_data, prm):
    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        #trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        try:
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
        except TypeError:
            continue

        data = np.vstack((rates, position, types, speed))
        data=data.transpose()
        data_filtered = data[data[:,2] == 0,:]
        data_filtered = data_filtered[data_filtered[:,3] > 3,:]

        rates_b = data_filtered[:,0]
        position_b = data_filtered[:,1]

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape)); binned_speed[:] = np.nan
        binned_speed_sd = np.zeros((position_array.shape)); binned_speed_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_b, np.where(np.logical_and(position_b >= rowcount, position_b <= rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed[rowcount] = average_rates
            binned_speed_sd[rowcount] = sd_speed

        data_filtered = data[data[:,2] != 0,:]
        data_filtered = data_filtered[data_filtered[:,3] > 3,:]
        rates_nb = data_filtered[:,0]
        position_nb = data_filtered[:,1]
        binned_speed_nb = np.zeros((position_array.shape)); binned_speed_nb[:] = np.nan
        binned_speed_nb_sd = np.zeros((position_array.shape)); binned_speed_nb_sd[:] = np.nan
        for rowcount, row in enumerate(position_array):
            rate_in_position = np.take(rates_nb, np.where(np.logical_and(position_nb >= rowcount, position_nb <= rowcount+1)))
            average_rates = np.nanmean(rate_in_position)
            sd_speed = np.nanstd(rate_in_position)
            binned_speed_nb[rowcount] = average_rates
            binned_speed_nb_sd[rowcount] = sd_speed

        ##print('plotting speed histogram...', cluster)
        save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        binned_speed = convolve_with_scipy(binned_speed)
        binned_speed_sd = convolve_with_scipy(binned_speed_sd)
        binned_speed_nb = convolve_with_scipy(binned_speed_nb)
        binned_speed_nb_sd = convolve_with_scipy(binned_speed_nb_sd)

        binned_speed_nb = pd.Series(binned_speed_nb).interpolate(method='linear', order=1)
        binned_speed_nb_sd = pd.Series(binned_speed_nb_sd).interpolate(method='linear', order=1)

        cluster_index = spike_data.cluster_id.values[cluster] - 1
        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_array,binned_speed, '-', color='Black')
        ax.fill_between(position_array, binned_speed-binned_speed_sd,binned_speed+binned_speed_sd, facecolor = 'Black', alpha = 0.2)
        ax.plot(position_array,binned_speed_nb, '-', color='Red')
        ax.fill_between(position_array, binned_speed_nb-binned_speed_nb_sd,binned_speed_nb+binned_speed_nb_sd, facecolor = 'Red', alpha = 0.2)
        plt.ylabel('Rates (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
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
        plt.xlim(20,100)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_probe_outbound.png', dpi=200)
        plt.close()

    return spike_data




"""
        #plot outbound fit
        #slope, intercept, r_value, p_value, std_err = stats.linregress(position_array[30:90],binned_speed[30:90])
        #line = slope*position_array[30:90]+intercept
        #ax.plot(position_array[30:90], line, '-', color = 'red', linewidth=2)
        #plot homebound fit
        #slope, intercept, r_value, p_value, std_err = stats.linregress(position_array[110:170],binned_speed[110:170])
        #line = slope*position_array[110:170]+intercept
        #ax.plot(position_array[110:170], line, '-', color = 'red', linewidth=2)

"""

def bin_data(data):
    bins = np.arange(0,200,1)
    bin_means = (np.histogram(data, bins, weights=data)[0] /
             np.histogram(data, bins)[0])
    return bin_means


def extract_time_binned_firing_rate_overtrial_per_trialtype(spike_data, prm):
    spike_data["Rates_averaged"] = ""
    spike_data["Rates_averaged_nb"] = ""
    spike_data["Rates_averaged_p"] = ""
    save_path = prm.get_local_recording_folder_path() + '/Figures/Average_Rates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        try:
            rates = Python_PostSorting.ConvolveRates_FFT.convolve_binned_spikes(rates)
        except TypeError:
            continue

        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data_speed_filtered = data[data[:,1] >= 3,:]

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

        avg_b_binned_data = np.nanmean(binned_data[:,:,0], axis=1)
        sd_b_binned_data = np.nanstd(binned_data[:,:,0], axis=1)
        avg_b_binned_data = convolve_with_scipy(avg_b_binned_data)
        #sd_b_binned_data = convolve_with_scipy(sd_b_binned_data)
        spike_data.at[cluster, 'Rates_averaged'] = list(avg_b_binned_data)

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
        ax.axvspan(0, 10, facecolor='k', linewidth =0, alpha=.15) # black box
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
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_beaconed_bytrial_outbound.png', dpi=200)
        plt.close()

        avg_p_binned_data = np.nanmean(binned_data[:,:,2], axis=1)
        sd_p_binned_data = np.nanstd(binned_data[:,:,2], axis=1)

        avg_p_binned_data = pd.Series(avg_p_binned_data).interpolate(method='linear', order=1)
        sd_p_binned_data = pd.Series(sd_p_binned_data).interpolate(method='linear', order=1)
        avg_p_binned_data = convolve_with_scipy(avg_p_binned_data)
        #sd_p_binned_data = convolve_with_scipy(sd_p_binned_data)
        spike_data.at[cluster, 'Rates_averaged_p'] = list(avg_p_binned_data)

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


        avg_both_binned_data = np.nanmean(np.nanmean(binned_data[:,:,1:], axis=2), axis=1)
        sd_both_binned_data = np.nanstd(np.nanmean(binned_data[:,:,1:], axis=2), axis=1)

        avg_both_binned_data = pd.Series(avg_both_binned_data).interpolate(method='linear', order=1)
        sd_both_binned_data = pd.Series(sd_both_binned_data).interpolate(method='linear', order=1)

        avg_both_binned_data = convolve_with_scipy(avg_both_binned_data)
        #sd_both_binned_data = convolve_with_scipy(sd_both_binned_data)
        spike_data.at[cluster, 'Rates_averaged_nb'] = list(avg_both_binned_data)

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
        plt.locator_params(axis = 'x', nbins  = 3)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_whole_both_bytrial_outbound.png', dpi=200)
        plt.close()

    print("finished plotting whole track rates for trial types")
    return spike_data
