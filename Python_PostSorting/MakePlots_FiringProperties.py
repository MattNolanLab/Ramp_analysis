import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
import numpy as np
import matplotlib.gridspec as gridspec
import Python_PostSorting.ExtractFiringData
import math


"""

## cluster firing properties
The following functions make plots of cluster firing properties:

-> spike histogram
-> spike autocorrelograms
-> waveforms

"""

def plot_spike_histogram(spatial_firing, prm):
    sampling_rate = prm.get_sampling_rate()
    print('I will plot spikes vs time for the whole session excluding opto tagging.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/spike_histograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster_index = spatial_firing.cluster_id.values[cluster] - 1
        firings_cluster = spatial_firing.firing_times[cluster]
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        spike_hist, ax = Python_PostSorting.plot_utility.style_plot(ax)
        number_of_bins = int((firings_cluster[-1] - firings_cluster[0]) / (5*sampling_rate))
        if number_of_bins > 0:
            hist, bins = np.histogram(firings_cluster, bins=number_of_bins)
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width, color='black')
        plt.title('total spikes = ' + str(spatial_firing.number_of_spikes[cluster]) + ', mean fr = ' + str(round(spatial_firing.mean_firing_rate[cluster], 0)) + ' Hz', y=1.08)
        plt.xlabel('time (sampling points)')
        plt.ylabel('number of spikes')
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster_index) + '_spike_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def calculate_autocorrelogram_hist(spikes, bin_size, window):

    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time



def plot_autocorrelograms(spike_data, prm):
    print('I will plot autocorrelograms for each cluster.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        #theta_index = spike_data.ThetaIndex.values[cluster]
        firing_times_cluster = spike_data.firing_times[cluster]
        '''
        #lags = plt.acorr(firing_times_cluster, maxlags=firing_times_cluster.size-1)
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 20)
        plt.xlim(-10, 10)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_10ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        '''
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        #ax.plot(snippets, color='lightslategray', linewidth=1, alpha=0.5)
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 500)
        ax.plot(time, corr, linewidth=1, color='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            width=2,
            length=4,
            labelsize=16)  # labels along the bottom edge are off
        ax.locator_params(axis='y', nbins=2)
        ax.locator_params(axis='x', nbins=2)
        plt.xlim(-250, 250)
        #ax.bar(time, corr, align='center', width=1, color='black')
        #plt.text(0.8,0.8, str(np.float(theta_index)), fontsize=10)
        plt.subplots_adjust(hspace = .35, wspace = .35, bottom = 0.16, left = 0.18, right = 0.92, top = 0.9)
        #x=np.max(corr)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_250ms.png', dpi=300)
        plt.close()


def plot_spikes_for_channel(grid, spike_data, cluster, channel):
    snippet_plot = plt.subplot(grid[int(channel/2), channel % 2])
    #plt.ylim(lowest_value - 10, highest_value + 30)
    Python_PostSorting.plot_utility.style_plot(snippet_plot)
    snippet_plot.plot(spike_data.random_snippets[cluster][channel, :, :] * -1, color='lightslategray', alpha=0.5)
    snippet_plot.plot(np.mean(spike_data.random_snippets[cluster][channel, :, :], 1) * -1, color='red')
    plt.xticks([0, 10, 30], [-10, 0, 20])


def plot_waveforms(spike_data, prm):
    print('I will plot the waveform shapes for each cluster.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/waveforms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        #max_channel = spike_data.primary_channel[cluster]
        #highest_value = np.max(spike_data.random_snippets[cluster][max_channel-1, :, :] * -1)
        #lowest_value = np.min(spike_data.random_snippets[cluster][max_channel-1, :, :] * -1)
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
        for channel in range(4):
            plot_spikes_for_channel(grid, spike_data, cluster, channel)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '_waveforms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def extract_mean_snippet_for_channel(spike_data, cluster, channel):
    mean_snippet = np.mean(spike_data.random_snippets[cluster][channel, :, :], 1) * -1
    return mean_snippet


def extract_all_snippet_for_channel(spike_data, cluster, channel):
    snippets = spike_data.random_snippets[cluster][channel, :, :] * -1
    return snippets


def plot_clean_waveforms(spike_data, prm):
    print('I will plot the waveform shapes for each cluster in clean format.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/waveforms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        fig = plt.figure(figsize=(3, 3))

        for channel in range(4):
            ax = fig.add_subplot(2, 2, (channel+1))  # specify (nrows, ncols, axnum)
            mean_snippet= extract_mean_snippet_for_channel(spike_data, cluster, channel)
            snippets= extract_all_snippet_for_channel(spike_data, cluster, channel)
            ax.plot(snippets, color='lightslategray', linewidth=1, alpha=0.5)
            ax.plot(mean_snippet, color='k', linewidth=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if channel==0:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=True,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    right=False,
                    left=True,
                    labelleft=True,
                    labelbottom=True,
                    width=2,
                    length=4)  # labels along the bottom edge are off
                ax.locator_params(axis='y', nbins=2)
                ax.locator_params(axis='x', nbins=2)
            else:
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    right=False,
                    left=False,
                    labelleft=False,
                    labelbottom=False)  # labels along the bottom edge are off

        #plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
        plt.subplots_adjust(hspace = .35, wspace = .35, bottom = 0.1, left = 0.08, right = 0.92, top = 0.9)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_waveforms_clean.png', dpi=300)
        plt.close()
