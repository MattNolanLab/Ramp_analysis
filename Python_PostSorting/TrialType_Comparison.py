import numpy as np
import Python_PostSorting.ExtractFiringData
from scipy import signal
import os
from matplotlib.pylab import plt



def normalise_signal(x):
    normalized = (x-min(x))/(max(x)-min(x))
    return normalized


def normalise_signals(x, y):
    if max(x) > max(y):
        max_of_signals = max(x)
    else:
        max_of_signals = max(y)

    if min(x) < min(y):
        min_of_signals = min(x)
    else:
        min_of_signals = min(y)

    normalized_x = (x-min_of_signals)/(max_of_signals-min_of_signals)
    normalized_y = (y-min_of_signals)/(max_of_signals-min_of_signals)
    return normalized_x, normalized_y


def find_signal_diff(signal1, signal2):
    diff = (signal1 - signal2)
    return diff


def mean_rz_diff(array):
    mean = np.nanmean(array[80:100])
    return mean


def mean_nonrz_diff(array):
    outbound_signal1 = np.nanmean(array[40:60])
    homebound_signal1 = np.nanmean(array[100:120])
    return outbound_signal1, homebound_signal1


def add_diff_columns_to_frame(spike_data):
    spike_data["FRdiff_RZ_trials1"] = ""
    spike_data["FRdiff_outbound_trials1"] = ""
    spike_data["FRdiff_outbound_trials2"] = ""
    spike_data["FRdiff_outbound_trials3"] = ""
    spike_data["FRdiff_homebound_trials1"] = ""
    spike_data["FRdiff_homebound_trials2"] = ""
    spike_data["FRdiff_homebound_trials3"] = ""
    return spike_data


def generate_firingrate_diff_between_trialtypes(spike_data, recording_folder):
    spike_data = add_diff_columns_to_frame(spike_data)
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        beaconed_spike_rate, nonbeaconed_spike_rate, probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)

        # beaconed trials versus non beaconed
        normalised_signal1, normalised_signal2 = normalise_signals(beaconed_spike_rate[10:190], nonbeaconed_spike_rate[10:190])
        #normalised_signal2 = normalise_signal(nonbeaconed_spike_rate[10:190])
        signal1 = find_signal_diff(normalised_signal1, normalised_signal2)
        signal1_rz = mean_rz_diff(signal1)
        signal1_outbound, signal1_homebound = mean_nonrz_diff(signal1)
        spike_data.at[cluster, "FRdiff_RZ_trials1"] = signal1_rz
        spike_data.at[cluster, "FRdiff_outbound_trials1"] = signal1_outbound
        spike_data.at[cluster, "FRdiff_homebound_trials1"] = signal1_homebound
        #plot_firing_rate_diff(recording_folder, spike_data, cluster, normalised_signal1, normalised_signal2, signal1, prefix="type1")

        normalised_signal1, normalised_signal2 = normalise_signals(beaconed_spike_rate[10:190], probe_spike_rate[10:190])
        #normalised_signal2 = normalise_signal(probe_spike_rate[10:190])
        signal1 = find_signal_diff(normalised_signal1, normalised_signal2)
        signal1_rz = mean_rz_diff(signal1)
        signal1_outbound, signal1_homebound = mean_nonrz_diff(signal1)
        spike_data.at[cluster, "FRdiff_RZ_trials2"] = signal1_rz
        spike_data.at[cluster, "FRdiff_outbound_trials2"] = signal1_outbound
        spike_data.at[cluster, "FRdiff_homebound_trials2"] = signal1_homebound
        #plot_firing_rate_diff(recording_folder, spike_data, cluster, normalised_signal1, normalised_signal2, signal1, prefix="type2")

        normalised_signal1, normalised_signal2 = normalise_signals(nonbeaconed_spike_rate[10:190], probe_spike_rate[10:190])
        #normalised_signal2 = normalise_signal(probe_spike_rate[10:190])
        signal1 = find_signal_diff(normalised_signal1, normalised_signal2)
        signal1_rz = mean_rz_diff(signal1)
        signal1_outbound, signal1_homebound = mean_nonrz_diff(signal1)
        spike_data.at[cluster, "FRdiff_RZ_trials3"] = signal1_rz
        spike_data.at[cluster, "FRdiff_outbound_trials3"] = signal1_outbound
        spike_data.at[cluster, "FRdiff_homebound_trials3"] = signal1_homebound
        #plot_firing_rate_diff(recording_folder, spike_data, cluster, normalised_signal1, normalised_signal2, signal1, prefix="type3")

    return spike_data


def plot_firing_rate_diff(recording_folder, spike_data, cluster, signal1, signal2, diff, prefix):
    save_path = recording_folder + '/Figures/TrialTypeComparison'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    cluster_index = spike_data.cluster_id.values[cluster] - 1
    avg_spikes_on_track = plt.figure(figsize=(5,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(signal1, '-', color='Blue', linewidth=2)
    ax.plot(signal2, '-', color='green', linewidth=2)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'black'
    ax2.set_ylabel('Difference (firing rate)', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(1,201,1), diff, '-', color='Black', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-1,1)
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
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_FiringRate_diff_trial1_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
    plt.close()



def add_columns_to_frame(spike_data):
    spike_data["FR_RZ_trials1"] = ""
    spike_data["FR_RZ_trials2"] = ""
    spike_data["FR_RZ_trials3"] = ""
    spike_data["FR_outbound_trials1"] = ""
    spike_data["FR_outbound_trials2"] = ""
    spike_data["FR_outbound_trials3"] = ""
    spike_data["FR_homebound_trials1"] = ""
    spike_data["FR_homebound_trials2"] = ""
    spike_data["FR_homebound_trials3"] = ""
    return spike_data



def generate_firingrate_comparison_between_trialtypes(spike_data):
    spike_data = add_columns_to_frame(spike_data)
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        beaconed_spike_rate, nonbeaconed_spike_rate, probe_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)

        # beaconed trials versus non beaconed
        normalised_signal1,normalised_signal2 = normalise_signals(beaconed_spike_rate[10:190], nonbeaconed_spike_rate[10:190])
        normalised_signal1_rz = mean_rz_diff(normalised_signal1)
        normalised_signal2_rz = mean_rz_diff(normalised_signal2)
        signal1_outbound, signal1_homebound = mean_nonrz_diff(normalised_signal1)
        signal2_outbound, signal2_homebound = mean_nonrz_diff(normalised_signal2)

        spike_data.at[cluster, "FR_RZ_trials1"] = normalised_signal1_rz
        spike_data.at[cluster, "FR_RZ_trials2"] = normalised_signal2_rz
        spike_data.at[cluster, "FR_RZ_trials3"] = 0
        spike_data.at[cluster, "FR_outbound_trials1"] = signal1_outbound
        spike_data.at[cluster, "FR_homebound_trials1"] = signal1_homebound
        spike_data.at[cluster, "FR_outbound_trials2"] = signal2_outbound
        spike_data.at[cluster, "FR_homebound_trials2"] = signal2_homebound
        spike_data.at[cluster, "FR_outbound_trials3"] = 0
        spike_data.at[cluster, "FR_homebound_trials3"] = 0
    return spike_data
