import numpy as np
import Python_PostSorting.ExtractFiringData
from scipy import signal
import os
from matplotlib.pylab import plt


"""

### Cross correlates firing rate across location for trial types (beaconed,nonbeaconed,probe)

1. extracts firing rate from frame for each trial type
2. performs cross correlation on pairs of trial types
3. puts cross correlation results into dataframe

"""


def cross_correlate_signal(signal1, signal2):
    #corr = signal.correlate(signal1, signal2, mode='same')/2000
    bins=np.arange(1,200,10)
    correlation_over_position = np.zeros((bins.shape[0]))
    for bcount, bin in enumerate(bins):
        signal1_bin = signal1[bin:bin+5]
        signal2_bin = signal2[bin:bin+5]
        corr = np.corrcoef(signal1_bin, signal2_bin)
        cc=corr[0,1]
        correlation_over_position[bcount] = cc
    return correlation_over_position


def correlation_coefficient_of_signal(signal1, signal2):
    corr = np.corrcoef(signal1, signal2)
    cc=corr[0,1]
    return cc


def correlation_coefficient_of_rewardzone_signal(signal1, signal2):
    signal1 = signal1[80:120]
    signal2 = signal2[80:120]
    corr = np.corrcoef(signal1, signal2)
    cc=corr[0,1]
    return cc


def correlation_coefficient_of_nonrewardzone_signal(signal1, signal2):
    nonrz_signal1 = np.hstack((signal1[30:80], signal1[120:170]))
    nonrz_signal2 = np.hstack((signal2[30:80], signal2[120:170]))
    corr = np.corrcoef(nonrz_signal1, nonrz_signal2)
    cc=corr[0,1]
    return cc


def correlation_coefficient_of_outbound_signal(signal1, signal2):
    outb_signal1 = signal1[30:80]
    outb_signal2 = signal2[30:80]
    corr = np.corrcoef(outb_signal1, outb_signal2)
    cc=corr[0,1]
    return cc


def correlation_coefficient_of_homebound_signal(signal1, signal2):
    outb_signal1 = signal1[110:170]
    outb_signal2 = signal2[110:170]
    corr = np.corrcoef(outb_signal1, outb_signal2)
    cc=corr[0,1]
    return cc


def add_columns_to_frame(spike_data):
    spike_data["cc_trials1"] = ""
    spike_data["cc_trials2"] = ""
    spike_data["cc_trials3"] = ""
    spike_data["cccoef_trials1"] = ""
    spike_data["cccoef_trials2"] = ""
    spike_data["cccoef_trials3"] = ""
    spike_data["cccoef_trials1_rz"] = ""
    spike_data["cccoef_trials2_rz"] = ""
    spike_data["cccoef_trials3_rz"] = ""
    spike_data["cccoef_trials1_nonrz"] = ""
    spike_data["cccoef_trials2_nonrz"] = ""
    spike_data["cccoef_trials3_nonrz"] = ""
    spike_data["cccoef_trials1_outbound"] = ""
    spike_data["cccoef_trials2_outbound"] = ""
    spike_data["cccoef_trials3_outbound"] = ""
    spike_data["cccoef_trials1_homebound"] = ""
    spike_data["cccoef_trials2_homebound"] = ""
    spike_data["cccoef_trials3_homebound"] = ""
    return spike_data


def generate_crosscorr_of_trialtypes(spike_data, recording_folder):
    spike_data = add_columns_to_frame(spike_data)
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        beaconed_spike_rate, nonbeaconed_spike_rate, probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)

        # beaconed trials versus non beaconed
        signal1 = cross_correlate_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        correlation_coefficient = correlation_coefficient_of_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        rz_correlation_coefficient = correlation_coefficient_of_rewardzone_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        #nonrz_correlation_coefficient = correlation_coefficient_of_nonrewardzone_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        outbound_correlation_coefficient = correlation_coefficient_of_outbound_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        homebound_correlation_coefficient = correlation_coefficient_of_homebound_signal(beaconed_spike_rate, nonbeaconed_spike_rate)
        spike_data.at[cluster, "cc_trials1"] = signal1
        spike_data.at[cluster, "cccoef_trials1"] = correlation_coefficient
        spike_data.at[cluster, "cccoef_trials1_rz"] = rz_correlation_coefficient
        #spike_data.at[cluster, "cccoef_trials1_nonrz"] = nonrz_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials1_outbound"] = outbound_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials1_homebound"] = homebound_correlation_coefficient
        plot_crosscorrelogram(recording_folder, spike_data, cluster, beaconed_spike_rate, nonbeaconed_spike_rate, signal1, prefix="type1")

        # beaconed trials versus probe trials
        signal2 = cross_correlate_signal(beaconed_spike_rate, probe_spike_rate)
        correlation_coefficient = correlation_coefficient_of_signal(beaconed_spike_rate, probe_spike_rate)
        outbound_correlation_coefficient = correlation_coefficient_of_outbound_signal(beaconed_spike_rate, probe_spike_rate)
        homebound_correlation_coefficient = correlation_coefficient_of_homebound_signal(beaconed_spike_rate, probe_spike_rate)
        spike_data.at[cluster, "cc_trials2"] = signal2
        spike_data.at[cluster, "cccoef_trials2"] = correlation_coefficient
        spike_data.at[cluster, "cccoef_trials2_rz"] = rz_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials2_outbound"] = outbound_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials2_homebound"] = homebound_correlation_coefficient
        #spike_data.at[cluster, "cccoef_trials2_nonrz"] = nonrz_correlation_coefficient
        plot_crosscorrelogram(recording_folder, spike_data, cluster, beaconed_spike_rate, probe_spike_rate, signal2, prefix="type2")

        # non beaconed trials versus probe trials
        signal3 = cross_correlate_signal(nonbeaconed_spike_rate, probe_spike_rate)
        correlation_coefficient = correlation_coefficient_of_signal(nonbeaconed_spike_rate, probe_spike_rate)
        outbound_correlation_coefficient = correlation_coefficient_of_outbound_signal(nonbeaconed_spike_rate, probe_spike_rate)
        homebound_correlation_coefficient = correlation_coefficient_of_homebound_signal(nonbeaconed_spike_rate, probe_spike_rate)
        spike_data.at[cluster, "cc_trials3"] = signal3
        spike_data.at[cluster, "cccoef_trials3"] = correlation_coefficient
        spike_data.at[cluster, "cccoef_trials3_rz"] = rz_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials3_outbound"] = outbound_correlation_coefficient
        spike_data.at[cluster, "cccoef_trials3_homebound"] = homebound_correlation_coefficient
        #spike_data.at[cluster, "cccoef_trials3_nonrz"] = nonrz_correlation_coefficient
        plot_crosscorrelogram(recording_folder, spike_data, cluster, nonbeaconed_spike_rate, probe_spike_rate, signal3, prefix="type3")
    return spike_data




def plot_crosscorrelogram(recording_folder, spike_data, cluster, signal1, signal2, corr, prefix):
    save_path = recording_folder + '/Figures/CrossCorrelation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    cluster_index = spike_data.cluster_id.values[cluster] - 1
    avg_spikes_on_track = plt.figure(figsize=(5,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(signal1, '-', color='Blue', linewidth=2)
    ax.plot(signal2, '-', color='green', linewidth=2)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'black'
    ax2.set_ylabel('Correlation', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(5,205,10), corr, '-', color='Black', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    #Python_PostSorting.plot_utility.style_vr_twin_plot(ax2, np.max(location_binned_isi), 0)

    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    ax.set_ylabel('Firing Rate (Hz)', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    #ax.set_ylim(0,1)
    plt.locator_params(axis = 'y', nbins  = 4)
    #Python_PostSorting.plot_utility.style_vr_plot(ax, np.max(location_binned_cv), 0)
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.4, left = 0.2, right = 0.8, top = 0.92)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_crosscorr_trial1_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
    plt.close()

