import elephant as elephant
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms
import numpy as np
import matplotlib.pylab as plt
from elephant.statistics import isi, cv
import os
import Python_PostSorting.plot_utility
import scipy.stats

"""

## here im testing some functions in the elephant python package using dummy data

1. generate spikes along a poisson distribution
2. plot raster
3. generate interspike interval
4. plot CV of ISI

"""


def test_spikes_from_poisson():
    spiketrain_list = [homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s, t_stop=100.0*s)for i in range(100)]

    for i, spiketrain in enumerate(spiketrain_list):
            t = spiketrain.rescale(ms)
            plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)
    plt.axis('tight')
    plt.xlim(0, 1000)
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Spike Train Index', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.show()

    cv_list = [cv(isi(spiketrain)) for spiketrain in spiketrain_list]

    plt.hist(cv_list)
    plt.xlabel('CV', fontsize=16)
    plt.ylabel('count', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    return


"""
### This code calculates two main variables for each cluster : 
1. interspike interval
2. coefficient of variation of the interspike interval

# Interspike interval:
1. load data:
    x_position_cm : data is all locations the cluster fires along the track (0-200 cm)
    trial_numbers : data is trial numbers corresponding to x_position_cm
2. loop trials and calculate interspike interval for each trial using np.diff (code ripped from elephant ISI calcualtion)
3. averages ISI over trials 

# Coefficient of variation of the interspike interval:
1. loop trials and bins (5 cm wide) calculate CV of interspike interval for each trial and bin (CV2 and CV)
2. average ISI CV2 over trials


"""



def add_column_to_dataframe(spike_data):
    spike_data["location_binned_interspike_interval"] = ""
    spike_data["mean_interspike_interval_b"] = ""
    spike_data["mean_interspike_interval_nb"] = ""
    spike_data["mean_interspike_interval_p"] = ""
    spike_data["location_binned_cv2"] = ""
    spike_data["location_binned_cv1"] = ""
    spike_data["mean_cv2_b"] = ""
    spike_data["mean_cv2_nb"] = ""
    spike_data["mean_cv2_p"] = ""
    spike_data["isi_diff_b"] = ""
    spike_data["cv_diff_b"] = ""
    spike_data["isi_diff_nb"] = ""
    spike_data["cv_diff_nb"] = ""
    spike_data["isi_diff_p"] = ""
    spike_data["cv_diff_p"] = ""
    return spike_data


def calculate_cv2(spiketrain, spike_cv):
    if spiketrain.shape[0] >= 2:
        cv = elephant.statistics.cv2(spiketrain, with_nan=True) # calculate cv2
        spike_cv= np.append(spike_cv, cv)
        return spike_cv
    else:
        return spike_cv


def isi_by_trial(spike_trials, spike_locations):
    trials = np.unique(spike_trials)
    spike_isi = []
    spike_cv = []
    new_spike_locations = []
    new_spike_trials = []
    for tcount, trial in enumerate(trials):
        trial_spikes = np.take(spike_locations, np.where(spike_trials == trial)[0])
        trial_trials = np.take(spike_trials, np.where(spike_trials == trial)[0])
        spiketrain = np.absolute(np.diff(trial_spikes, axis=0)) #spiketrain = elephant.statistics.isi(trial_spikes, axis=-0) # not working, ripped from source code
        spike_isi = np.append(spike_isi, spiketrain)
        new_spike_locations = np.append(new_spike_locations, trial_spikes[1:])
        new_spike_trials = np.append(new_spike_trials, trial_trials[1:])
        spike_cv = calculate_cv2(trial_spikes, spike_cv)
    return spike_isi, spike_cv, new_spike_locations, new_spike_trials


def bin_isi_over_location(spike_isi, new_spike_locations):
    bins = np.arange(1,201,5)
    location_binned_isi = []
    if len(new_spike_locations) >0:
        for bcount, bin in enumerate(bins):
            isi_locations = spike_isi[np.where(np.logical_and(new_spike_locations > bin, new_spike_locations <= (bin+5)))]
            location_binned_isi = np.append(location_binned_isi, (np.nanmean(isi_locations)))
    else:
        return np.full(bins.shape[0], np.nan)
    return location_binned_isi


def bin_cv_over_location(spike_isi, spike_locations,spike_trials):
    trials = np.unique(spike_trials)
    bins = np.arange(1,201,5)
    location_binned_cv2 = np.zeros((bins.shape[0], trials.shape[0])); location_binned_cv2[:,:] = np.nan
    for tcount, trial in enumerate(trials):
        trial_isi = np.take(spike_isi, np.where(spike_trials == trial)[0])
        trial_spikes = np.take(spike_locations, np.where(spike_trials == trial)[0])
        for bcount, bin in enumerate(bins):
            trial_spikes_in_bin = np.take(trial_isi, np.where(np.logical_and(trial_spikes > bin, trial_spikes <= (bin+5)))[0])
            if trial_spikes_in_bin.shape[0] >= 2:
                cv2 = elephant.statistics.cv2(trial_spikes_in_bin, with_nan=True)# calculate cv2
            else:
                cv2 = np.nan
            location_binned_cv2[bcount, tcount] = cv2
    mean_binned_cv2 = np.nanmean(location_binned_cv2, axis=1)
    return mean_binned_cv2


def round_down(num, divisor):
	return num - (num%divisor)


def calculate_isi_for_max_location(spike_data, cluster, location_binned_isi, location_binned_cv):
    unscaled_max_location=spike_data.at[cluster, "max_firing_location"]
    max_location = int(unscaled_max_location/5)
    isi_at_max = np.nanmean(location_binned_isi[max_location-1:max_location+1])
    cv_at_max = np.nanmean(location_binned_cv[max_location-1:max_location+1])
    unscaled_min_location=spike_data.at[cluster, "min_firing_location"]
    min_location = int(unscaled_min_location/5)
    isi_at_min = np.nanmean(location_binned_isi[min_location-1:min_location+1])
    cv_at_min = np.nanmean(location_binned_cv[min_location-1:min_location+1])
    isi_diff = isi_at_max - isi_at_min
    cv_diff = cv_at_max - cv_at_min
    return isi_diff, cv_diff


def split_firing_on_reward(spike_locations, spike_trials, spike_data, cluster_index):
    rewarded_trials = np.array(spike_data.at[cluster_index, 'rewarded_trials'], dtype=np.int16)
    rewarded_spike_locations = spike_locations[np.isin(spike_trials,rewarded_trials)]
    rewarded_spike_trials = spike_trials[np.isin(spike_trials,rewarded_trials)]
    return rewarded_spike_locations, rewarded_spike_trials


def split_firing_by_trialtype(spike_locations, spike_trials, spike_trialtype):
    beaconed_locations = np.take(spike_locations, np.where(spike_trialtype == 0)[0]) #split location and trial number
    nonbeaconed_locations = np.take(spike_locations,np.where(spike_trialtype == 1)[0])
    probe_locations = np.take(spike_locations, np.where(spike_trialtype == 2)[0])
    beaconed_trials = np.take(spike_trials, np.where(spike_trialtype == 0)[0]) #split location and trial number
    nonbeaconed_trials = np.take(spike_trials,np.where(spike_trialtype == 1)[0])
    probe_trials = np.take(spike_trials, np.where(spike_trialtype == 2)[0])
    return beaconed_locations, beaconed_trials, nonbeaconed_locations, nonbeaconed_trials, probe_locations, probe_trials


def generate_spike_isi(server_path, spike_data):
    print('I am calculating the interspike interval ...')
    spike_data = add_column_to_dataframe(spike_data)
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        spike_locations = np.array(spike_data.at[cluster, "x_position_cm"])
        spike_trials = np.array(spike_data.at[cluster, "trial_number"], dtype=np.int16)
        spike_trialtype = np.array(spike_data.at[cluster, "trial_type"], dtype=np.int16)
        #spike_locations, spike_trials = split_firing_on_reward(spike_locations, spike_trials,spike_data, cluster)
        beaconed_locations, beaconed_trials, nonbeaconed_locations, nonbeaconed_trials, probe_locations, probe_trials = split_firing_by_trialtype(spike_locations, spike_trials, spike_trialtype)

        ## beaconed trial analysis
        spike_isi, cv, new_spike_locations, new_spike_trials = isi_by_trial(beaconed_trials, beaconed_locations)
        location_binned_isi = bin_isi_over_location(spike_isi, new_spike_locations)
        location_binned_cv2 = bin_cv_over_location(spike_isi, new_spike_locations, new_spike_trials)
        mean_isi = np.nanmean(spike_isi)
        isi_diff, cv_diff = calculate_isi_for_max_location(spike_data, cluster, location_binned_isi, location_binned_cv2)
        spike_data = add_beaconed_data_to_frame(spike_data, cluster, location_binned_isi, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv)

        # uncomment if want to plot ISI and CV over location
        plot_isi_data(server_path, spike_data, cluster, location_binned_isi, location_binned_cv2, prefix="beaconed")

        ## nonbeaconed trial analysis
        spike_isi, cv, new_spike_locations, new_spike_trials = isi_by_trial(nonbeaconed_trials, nonbeaconed_locations)
        location_binned_isi = bin_isi_over_location(spike_isi, new_spike_locations)
        location_binned_cv2 = bin_cv_over_location(spike_isi, new_spike_locations, new_spike_trials)
        mean_isi = np.nanmean(spike_isi)
        isi_diff, cv_diff = calculate_isi_for_max_location(spike_data, cluster, location_binned_isi, location_binned_cv2)
        spike_data = add_nonbeaconed_data_to_frame(spike_data, cluster, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv)

        plot_isi_data(server_path, spike_data, cluster, location_binned_isi, location_binned_cv2, prefix="nonbeaconed")

        ## probe trial analysis
        spike_isi, cv, new_spike_locations, new_spike_trials = isi_by_trial(probe_trials, probe_locations)
        location_binned_isi = bin_isi_over_location(spike_isi, new_spike_locations)
        location_binned_cv2 = bin_cv_over_location(spike_isi, new_spike_locations, new_spike_trials)
        mean_isi = np.nanmean(spike_isi)
        isi_diff, cv_diff = calculate_isi_for_max_location(spike_data, cluster, location_binned_isi, location_binned_cv2)
        spike_data = add_probe_data_to_frame(spike_data, cluster, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv)

        plot_isi_data(server_path, spike_data, cluster, location_binned_isi, location_binned_cv2, prefix="probe")

    return spike_data



def add_beaconed_data_to_frame(spike_data, cluster, location_binned_interspike_interval, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv):
    spike_data.at[cluster, "location_binned_interspike_interval"] = location_binned_interspike_interval
    spike_data.at[cluster, "location_binned_cv2"] = location_binned_cv2
    spike_data.at[cluster, "mean_interspike_interval_b"] = np.float(mean_isi)
    spike_data.at[cluster, "mean_cv2_b"] = np.nanmean(cv)
    spike_data.at[cluster, "isi_diff_b"] = isi_diff
    spike_data.at[cluster, "cv_diff_b"] = cv_diff
    return spike_data


def add_nonbeaconed_data_to_frame(spike_data, cluster, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv):
    spike_data.at[cluster, "mean_interspike_interval_nb"] = np.float(mean_isi)
    spike_data.at[cluster, "mean_cv2_nb"] = np.nanmean(cv)
    spike_data.at[cluster, "isi_diff_nb"] = isi_diff
    spike_data.at[cluster, "cv_diff_nb"] = cv_diff
    return spike_data


def add_probe_data_to_frame(spike_data, cluster, location_binned_cv2, mean_isi, isi_diff, cv_diff, cv):
    spike_data.at[cluster, "mean_interspike_interval_p"] = np.float(mean_isi)
    spike_data.at[cluster, "mean_cv2_p"] = np.nanmean(cv)
    spike_data.at[cluster, "isi_diff_p"] = isi_diff
    spike_data.at[cluster, "cv_diff_p"] = cv_diff
    return spike_data


def plot_isi_data(recording_folder, spike_data, cluster, location_binned_isi, location_binned_cv2, prefix):
    save_path = recording_folder + '/Figures/interspike_interval_analysis'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    cluster_index = spike_data.cluster_id.values[cluster] - 1
    avg_spikes_on_track = plt.figure(figsize=(5,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(np.arange(1,201,5),location_binned_cv2, '-', color='Blue', linewidth=2)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'black'
    ax2.set_ylabel('Mean ISI (cm)', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(1,201,5), location_binned_isi, '-', color='Black', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    #Python_PostSorting.plot_utility.style_vr_twin_plot(ax2, np.max(location_binned_isi), 0)

    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    #Python_PostSorting.plot_utility.makelegend(avg_spikes_on_track,ax, 0.3)
    ax.set_ylabel('CV2', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    #ax.set_ylim(0,1)
    plt.locator_params(axis = 'y', nbins  = 4)
    #Python_PostSorting.plot_utility.style_vr_plot(ax, np.max(location_binned_cv), 0)
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.4, left = 0.2, right = 0.8, top = 0.92)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_location_binned_cv_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
    plt.close()

