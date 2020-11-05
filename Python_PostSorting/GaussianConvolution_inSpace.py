import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import Python_PostSorting.ShuffleAnalysis
import Python_PostSorting.ExtractFiringData
import math
import elephant as elephant
import os
from scipy import signal
from scipy import stats


def extract_firing_num_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({'firing_rate':  spike_data.iloc[cluster_index].spike_rate_on_trials[0], 'trial_number':  spike_data.iloc[cluster_index].spike_rate_on_trials[1], 'trial_type':  spike_data.iloc[cluster_index].spike_rate_on_trials[2]})
    return cluster_firings


def get_bin_size():
    bin_size_cm = 5
    return bin_size_cm


def get_number_of_bins(cluster_firings, bin_array):
    bin_size_cm = get_bin_size()
    number_of_bins_x = int(bin_array.max())
    length_of_arena_y = cluster_firings.trial_number.max()
    number_of_bins_y = int(length_of_arena_y)
    return number_of_bins_x, number_of_bins_y, bin_size_cm


def round_down(num, divisor):
	return num - (num%divisor)


def average_convolved_rate_over_trials(convolved_rate):
    trials=int(round_down((convolved_rate.shape[0]/200),1))
    array_shape=trials*200
    reshaped_hist = np.reshape(convolved_rate[:array_shape], (int(trials), 200))
    hist = np.nanmean(reshaped_hist, axis=0)
    hist_sd = np.std(reshaped_hist, axis=0)
    return hist, hist_sd


def convolve_with_elephant(rate, time):
    convolved_num = elephant.statistics.fftkernel(rate, 0.5) # convolve spike number
    convolved_time = elephant.statistics.fftkernel(time, 0.5) # convolve dwell time
    convolved_rate= (convolved_num/convolved_time) # calculate firing rate
    return convolved_rate


def convolve_with_scipy(rate):
    window = signal.gaussian(5, std=3)
    #plt.plot(window)
    try:
        convolved_rate = signal.convolve(rate, window, mode='same')
    except ValueError:
        convolved_rate=np.zeros((rate.shape[0]))
    #filtered_time = signal.convolve(time, window, mode='same')
    #convolved_rate = (filtered/filtered_time)
    #mean_rate = np.nanmean(convolved_rate)
    #convolved_rate[convolved_rate>mean_rate*3] = mean_rate
    return (convolved_rate/10)


def gaussian_convolve_firing(cluster_firings):
    rate = np.array(cluster_firings["firing_rate"])
    #time = np.array(cluster_firings["time"])
    #time = np.asarray(time*1000, dtype=np.int16) # convert time to ms

    convolved_rate = convolve_with_scipy(rate)
    #convolved_rate = convolve_with_elephant(rate, time)
    return convolved_rate


def quick_plot(convolved_rate):
    reshaped_hist = np.reshape(convolved_rate, (int(124), 200))
    hist = np.nanmean(reshaped_hist, axis=0)
    plt.plot(hist)


def make_location_bins(cluster_firings):
    bins=np.arange(1,201,1)
    max_trial = np.max(np.array(cluster_firings["trial_number"]))
    bin_array= np.tile(bins,int(max_trial))
    return bin_array


def quick_plots(b, nb, spike_data, cluster, server_path):
    rate = np.array(b["firing_rate"])
    rate2 = np.array(nb["firing_rate"])
    save_path = server_path + '/Figures/spike_num'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_index = spike_data.cluster_id.values[cluster] - 1

    trials=int(round_down((rate.shape[0]/200),1))
    array_shape=trials*200
    reshaped_hist = np.reshape(rate[:array_shape], (int(trials), 200))
    hist = np.nanmean(reshaped_hist, axis=0)
    hist_sd = stats.sem(reshaped_hist, axis=0)

    trials=int(round_down((rate2.shape[0]/200),1))
    array_shape=trials*200
    reshaped_hist = np.reshape(rate2[:array_shape], (int(trials), 200))
    hist1 = np.nanmean(reshaped_hist, axis=0)
    hist1_sd = stats.sem(reshaped_hist, axis=0)

    bins=np.arange(0,200,1)
    plot = plt.figure(figsize=(4,3))
    ax = plot.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(bins, hist, '-', color='Black')
    ax.plot(bins, hist1, '-', color='Red')
    #plt.show()
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_spike_map_Cluster_' + str(cluster_index +1) + '_b.png', dpi=200)
    plt.close()
    return


def split_firing_data_by_trial_type(cluster_firings, spike_data, cluster, server_path):
    beaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==0)
    nbeaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==1)
    probe_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==2)
    beaconed_cluster_firings.dropna(axis=0, inplace=True)
    nbeaconed_cluster_firings.dropna(axis=0, inplace=True)
    probe_cluster_firings.dropna(axis=0, inplace=True)

    quick_plots(beaconed_cluster_firings, nbeaconed_cluster_firings, spike_data, cluster, server_path)
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def make_convolved_firing_field_maps(server_path, spike_data):
    print('I am calculating the average firing rate ...')
    spike_data["convolved_firing_maps"] = ""

    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        session_id=spike_data.at[cluster, "session_id"]
        #mean_firing_rate=spike_data.at[cluster, "mean_firing_rate"]
        cluster_firings = extract_firing_num_data(spike_data, cluster)
        beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings, spike_data, cluster, server_path)
        convolved_rate_b = gaussian_convolve_firing(beaconed_cluster_firings)
        convolved_rate_nb = gaussian_convolve_firing(nbeaconed_cluster_firings)
        convolved_rate_p = gaussian_convolve_firing(probe_cluster_firings)

        average_convolved_rate_b, sd_b = average_convolved_rate_over_trials(convolved_rate_b)
        average_convolved_rate_nb, sd_nb = average_convolved_rate_over_trials(convolved_rate_nb)
        average_convolved_rate_p, sd_p = average_convolved_rate_over_trials(convolved_rate_p)

        #spike_data = add_data_to_dataframe(spike_data, cluster, cluster_firings, convolved_rate)
        plot_convolved_rate(server_path, spike_data, cluster, session_id, average_convolved_rate_b, average_convolved_rate_nb, average_convolved_rate_p, sd_b, sd_nb, sd_p)
        #plot_convolved_rate_with_fits(server_path, spike_data, fits, cluster, session_id, average_convolved_rate_b, average_convolved_rate_nb, average_convolved_rate_p, sd_b, sd_nb, sd_p)
        #plot_convolved_rate_per_segment(server_path, spike_data, cluster, average_convolved_rate_b, average_convolved_rate_nb, average_convolved_rate_p, sd_b, sd_nb, sd_p)
    print('-------------------------------------------------------------')
    print('firing field maps processed')
    print('-------------------------------------------------------------')
    return spike_data


def add_data_to_dataframe(spike_data, cluster_index, cluster_firings,convolved_rate):
    sr_smooth=[]
    sr_smooth.append(convolved_rate)
    sr_smooth.append(np.array(cluster_firings['trial_number']))
    sr_smooth.append(np.array(cluster_firings['trial_type']))
    spike_data.at[cluster_index, 'convolved_firing_maps'] = list(sr_smooth)
    return spike_data


def plot_convolved_rate(recording_folder, spike_data, cluster, session_id, hist1, hist2, hist3, sd1, sd2, sd3):
    #print('I am plotting smoothed firing rate maps...')
    bins=np.arange(0,200,1)
    save_path = recording_folder + '/Figures/spike_rate_convolved'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_index = spike_data.cluster_id.values[cluster] - 1

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(bins, hist1, '-', color='Black')
    ax.fill_between(bins, hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.2)

    ax.plot(bins, hist2, '-', color='Red')
    ax.fill_between(bins, hist2-sd2,hist2+sd2, facecolor = 'Red', alpha = 0.2)

    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    try:
        x_max = (np.nanmax(hist1) + np.nanmax(sd1))
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    except ValueError:
        print('Axis limits cannot be NaN or Inf')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_b.png', dpi=200)
    plt.close()

    """
    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(hist2, '-', color='Black')
    ax.fill_between(bins, hist2-sd2,hist2+sd2, facecolor = 'Black', alpha = 0.3)
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    x_max = (np.nanmax(hist2) + np.nanmax(sd2))
    plt.locator_params(axis = 'y', nbins  = 4)
    Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_nb.png', dpi=200)
    plt.close()


    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(hist3, '-', color='Black')
    ax.fill_between(bins, hist3-sd3,hist3+sd3, facecolor = 'Black', alpha = 0.3)
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    x_max = (np.nanmax(hist3) + np.nanmax(sd3))
    plt.locator_params(axis = 'y', nbins  = 4)
    Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_p.png', dpi=200)
    plt.close()

    """
    avg_spikes_on_track = plt.figure(figsize=(4,3.5))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(hist1, '-', color='Black')
    ax.fill_between(bins, hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.3)
    ax.plot(hist2, '-', color='red')
    ax.fill_between(bins, hist2-sd2,hist2+sd2, facecolor = 'red', alpha = 0.3)


    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    x_max = (np.nanmax(hist1) + np.nanmax(sd1))
    plt.locator_params(axis = 'y', nbins  = 4)
    Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    #plt.ylim(0,40)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_all.png', dpi=200)
    plt.close()






def plot_convolved_rate_with_fits(recording_folder, spike_data, fits, cluster, session_id, hist1, hist2, hist3, sd1, sd2, sd3):
    #print('I am plotting smoothed firing rate maps...')
    bins=np.arange(0,200,1)
    save_path = recording_folder + '/Figures/spike_rate_convolved'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    cluster_index = spike_data.cluster_id.values[cluster] - 1
    neuron_number = spike_data.neuron_number.values[cluster]

    # get fit for same session
    session_fits = fits['session_id'] == session_id
    session_fits = fits[session_fits]

    # get only beaconed trials
    b_fits = session_fits['trial_type'] == 'beaconed'
    b_fits = session_fits[b_fits]

    # find that neuron
    neuron_fits = b_fits['neuron'] == neuron_number
    neuron_fits = b_fits[neuron_fits]
    fit_curve = neuron_fits['meanCurves'].values[0] # extract fit


    # plot graph
    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(bins, hist1, '-', color='Black')
    ax.fill_between(bins, hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.2)
    ax.plot(np.arange(0,200,2), fit_curve, '-', color='Red', linewidth=2)
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    try:
        x_max = (np.nanmax(hist1) + np.nanmax(sd1))
        plt.locator_params(axis = 'y', nbins  = 4)
        Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
        Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    except ValueError:
        print('Axis limits cannot be NaN or Inf')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_b_fits.png', dpi=200)
    plt.close()



def plot_convolved_rate_per_segment(recording_folder, spike_data, cluster, hist1, hist2, hist3, sd1, sd2, sd3):
    #print('I am plotting smoothed firing rate maps...')
    bins=np.arange(0,200,1)
    save_path = recording_folder + '/Figures/spike_rate_convolved'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_index = spike_data.cluster_id.values[cluster] - 1

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(hist1, '-', color='Black', markersize=2)
    ax.fill_between(bins, hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.3)
    ax.plot(hist2, '-', color='red', markersize=2)
    ax.fill_between(bins, hist2-sd2,hist2+sd2, facecolor = 'red', alpha = 0.3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(30,90)
    #x_max = np.nanmax(hist1)
    plt.locator_params(axis = 'y', nbins  = 4)
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
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_outbound.png', dpi=200)
    plt.close()

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(hist1, '-', color='Black', markersize=2)
    ax.fill_between(bins, hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.3)
    ax.plot(hist2, '-', color='red', markersize=2)
    ax.fill_between(bins, hist2-sd2,hist2+sd2, facecolor = 'red', alpha = 0.3)
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(110,170)
    #x_max = np.nanmax(hist1)
    plt.locator_params(axis = 'y', nbins  = 4)
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
    Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_homebound.png', dpi=200)
    plt.close()




## for longer tracks


def make_convolved_firing_field_maps_for_tracks(server_path, spike_data):
    print('I am calculating the average firing rate ...')
    spike_data["convolved_firing_maps"] = ""

    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        cluster_firings = extract_firing_num_data(spike_data, cluster)
        beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings, spike_data, cluster, server_path)
        convolved_rate_b = gaussian_convolve_firing(beaconed_cluster_firings)
        convolved_rate_nb = gaussian_convolve_firing(nbeaconed_cluster_firings)
        convolved_rate_p = gaussian_convolve_firing(probe_cluster_firings)

        average_convolved_rate_b, sd_b = average_convolved_rate_over_trials(convolved_rate_b)
        average_convolved_rate_nb, sd_nb = average_convolved_rate_over_trials(convolved_rate_nb)
        average_convolved_rate_p, sd_p = average_convolved_rate_over_trials(convolved_rate_p)

        #spike_data = add_data_to_dataframe(spike_data, cluster, cluster_firings, convolved_rate)
        plot_convolved_rate_for_tracks(server_path, spike_data, cluster, average_convolved_rate_b, average_convolved_rate_nb, average_convolved_rate_p, sd_b, sd_nb, sd_p)
        #plot_convolved_rate_per_segment(server_path, spike_data, cluster, average_convolved_rate_b, average_convolved_rate_nb, average_convolved_rate_p, sd_b, sd_nb, sd_p)
    print('-------------------------------------------------------------')
    print('firing field maps processed')
    print('-------------------------------------------------------------')
    return spike_data


def plot_convolved_rate_for_tracks(recording_folder, spike_data, cluster, hist1, hist2, hist3, sd1, sd2, sd3):
    #print('I am plotting smoothed firing rate maps...')
    bins=np.arange(0,200,1)
    save_path = recording_folder + '/Figures/spike_rate_convolved_tracks'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_index = spike_data.cluster_id.values[cluster] - 1
    avg_spikes_on_track = plt.figure(figsize=(6,3.5))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(np.arange(0,100,0.5), hist1, '-', color='Black')
    ax.fill_between(np.arange(0,100,0.5), hist1-sd1,hist1+sd1, facecolor = 'Black', alpha = 0.2)
    hist4=(hist2+hist1)/2
    ax.plot(np.arange(0,100,0.5),hist4, '-', color='Red')
    ax.fill_between(np.arange(0,100,0.5), hist4-sd2,hist4+sd2, facecolor = 'Red', alpha = 0.2)
    ax.plot(hist3, '-', color='Blue')
    ax.fill_between(bins, hist3-sd3,hist3+sd3, facecolor = 'Blue', alpha = 0.3)

    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    plt.locator_params(axis = 'y', nbins  = 4)
    x_max = (np.nanmax(hist1) + (np.nanmax(sd1))/2)

    Python_PostSorting.plot_utility.style_vr_plot(ax, x_max,0)
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    ax.axvspan(88/2, (88+22)/2, facecolor='DarkGreen', alpha=.25, linewidth =0)
    ax.axvspan(0, 30/2, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan((200-30)/2, 200/2, facecolor='k', linewidth =0, alpha=.25)# black box

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_b.png', dpi=200)
    plt.close()



