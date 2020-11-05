import numpy as np
import matplotlib.pyplot as plt
import Python_PostSorting.ExtractFiringData
import os




"""

## the following code calculates the mean and max firing rate along the track (30-170 cm).

1. extract mean firing rate across location
2. extract max firing rate across location
3. insert result into dataframe

"""


def calculate_max_firing_rate(spike_data):
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = Python_PostSorting.ExtractFiringData.extract_average_firing_rate_data(spike_data, cluster)

        max_fr = np.nanmax(avg_beaconed_spike_rate)
        mean_fr = np.nanmean(avg_beaconed_spike_rate)

        spike_data.at[cluster, "mean_fr"] = max_fr
        spike_data.at[cluster, "max_fr"] = mean_fr
    return spike_data




"""

## the following code calculates the difference between firing rate along the track (30-170 cm) and in the black boxes (30 cm flanking the start and end)

1. extract mean firing rate across location
2. average firing rate along track (30 - 170 cm)
3. average firing rate in the black boxes (30 cm flanking the start and end) 
4. difference in firing rate between black box and track
5. insert result into dataframe

"""


def calculate_track_and_bb_firing_rate_diff(recording_folder, spike_data):
    spike_data["bb_firing_diff"] = ""
    spike_data["bb1_rate"] = ""
    spike_data["bb2_rate"] = ""
    spike_data["reward_rate"] = ""

    for cluster in range(len(spike_data)):
        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = Python_PostSorting.ExtractFiringData.extract_average_firing_rate_data(spike_data, cluster)
        mean_track_firing_rate = np.nanmean(avg_beaconed_spike_rate[30:170])
        mean_bb1_firing_rate = np.nanmean(avg_beaconed_spike_rate[5:30])
        mean_bb2_firing_rate = np.nanmean(avg_beaconed_spike_rate[170:195])
        mean_RZ_firing_rate = np.nanmean(avg_beaconed_spike_rate[90:110])
        mean_bb_firing_rate = np.nanmean(np.hstack((mean_bb1_firing_rate, mean_bb2_firing_rate)))
        track_diff_firing_rate = mean_track_firing_rate-mean_bb_firing_rate

        spike_data.at[cluster, "bb_firing_diff"] = track_diff_firing_rate
        spike_data.at[cluster, "bb1_rate"] = mean_bb1_firing_rate
        spike_data.at[cluster, "bb2_rate"] = mean_bb2_firing_rate
        spike_data.at[cluster, "reward_rate"] = mean_RZ_firing_rate

    #plot_isi_verusus_bb_firing(recording_folder, spike_data)
    return spike_data



def plot_isi_verusus_bb_firing(recording_folder, spike_data):
    print('I am plotting isi and CV...')
    save_path = recording_folder + '/Figures/Overall'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    mean_isi = np.array(spike_data["mean_interspike_interval"])
    mean_bb_firing_diff = np.array(spike_data["bb_firing_diff"])
    avg_spikes_on_track = plt.figure(figsize=(6,3.5))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(mean_bb_firing_diff, mean_isi)
    plt.xlabel('Mean ISI (cm)', fontsize=14, labelpad = 10)
    plt.ylabel('FR[track]-FR[BB]', fontsize=14, labelpad = 10)
    plt.savefig(save_path + '/' + 'MeanISI_BBFR_' + '.png', dpi=200)
    plt.close()










###--------------------------------------------------------###


def calculate_track_and_bb_firing_rate(spike_data):
    print("calculating firing rate in the black box and track separately...")
    spike_data["bb_rate"] = ""
    spike_data["track_rate"] = ""

    for cluster in range(len(spike_data)):
        avg_beaconed_spike_rate, avg_beaconed_spike_rate, avg_beaconed_spike_rate, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)
        mean_track_firing_rate = np.nanmean(avg_beaconed_spike_rate[30:170])
        mean_bb1_firing_rate = np.nanmean(avg_beaconed_spike_rate[5:30])
        mean_bb2_firing_rate = np.nanmean(avg_beaconed_spike_rate[170:195])
        #mean_RZ_firing_rate = np.nanmean(avg_beaconed_spike_rate[90:110])
        mean_bb_firing_rate = np.nanmean(np.hstack((mean_bb1_firing_rate, mean_bb2_firing_rate)))
        #track_diff_firing_rate = mean_track_firing_rate-mean_bb_firing_rate

        #spike_data.at[cluster, "bb_firing_diff"] = track_diff_firing_rate
        spike_data.at[cluster, "bb_rate"] = mean_bb_firing_rate
        spike_data.at[cluster, "track_rate"] = mean_track_firing_rate
        #spike_data.at[cluster, "reward_rate"] = mean_RZ_firing_rate
    #plot_isi_verusus_bb_firing(recording_folder, spike_data)
    return spike_data
