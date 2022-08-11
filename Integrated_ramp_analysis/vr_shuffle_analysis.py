import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.vr_sync_spatial_data
import traceback
import PostSorting.vr_spatial_firing
import control_sorting_analysis
import warnings
from scipy import stats
import plot_utility
import os
import sys
import settings
from astropy.convolution import convolve, Gaussian1DKernel


def add_speed(shuffle_firing, position_data):
    raw_speed_per200ms = np.array(position_data["speed_per200ms"])

    speed_per200ms = []
    for shuffle_index, shuffle_id in enumerate(shuffle_firing.shuffle_id):
        cluster_firing_indices = np.asarray(shuffle_firing[shuffle_firing.shuffle_id == shuffle_id].firing_times)[0]
        speed_per200ms.append(raw_speed_per200ms[cluster_firing_indices].tolist())

    shuffle_firing["speed_per200ms"] = speed_per200ms
    return shuffle_firing


def add_position_x(shuffle_firing, position_data):
    raw_x_position_cm = np.array(position_data["x_position_cm"])

    x_position_cm = []
    for shuffle_index, shuffle_id in enumerate(shuffle_firing.shuffle_id):
        cluster_firing_indices = np.asarray(shuffle_firing[shuffle_firing.shuffle_id == shuffle_id].firing_times)[0]
        x_position_cm.append(raw_x_position_cm[cluster_firing_indices].tolist())

    shuffle_firing["x_position_cm"] = x_position_cm
    return shuffle_firing


def add_trial_number(shuffle_firing, position_data):
    raw_trial_number = np.array(position_data["trial_number"])

    trial_number = []
    for shuffle_index, shuffle_id in enumerate(shuffle_firing.shuffle_id):
        cluster_firing_indices = np.asarray(shuffle_firing[shuffle_firing.shuffle_id == shuffle_id].firing_times)[0]
        trial_number.append(raw_trial_number[cluster_firing_indices].tolist())

    shuffle_firing["trial_number"] = trial_number
    return shuffle_firing


def add_trial_type(shuffle_firing, position_data):
    raw_trial_type = np.array(position_data["trial_type"])

    trial_type = []

    for shuffle_index, shuffle_id in enumerate(shuffle_firing.shuffle_id):
        cluster_firing_indices = np.asarray(shuffle_firing[shuffle_firing.shuffle_id == shuffle_id].firing_times)[0]
        trial_type.append(raw_trial_type[cluster_firing_indices].tolist())

    shuffle_firing["trial_type"] = trial_type
    return shuffle_firing


def get_stop_threshold_and_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, _ = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return stop_threshold, track_length

def generate_shuffled_times(cluster_firing, n_shuffles, sample_rate, downsample_rate):
    session_id = cluster_firing["session_id"].iloc[0]
    cluster_firing = cluster_firing[["cluster_id", "firing_times", "mean_firing_rate", "recording_length_sampling_points", "trial_number"]]

    shuffle_firing = pd.DataFrame()
    for i in range(n_shuffles):
        shuffle = cluster_firing.copy()
        firing_times = shuffle["firing_times"].to_numpy()[0]
        trial_numbers = shuffle["trial_number"].to_numpy()[0]
        recording_length = int(cluster_firing["recording_length_sampling_points"].iloc[0])

        # generate random index firing time addition independently for each trial spike
        random_firing_additions_by_trial = np.array([])
        for tn in np.unique(trial_numbers):
            random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate), high=int(580*settings.sampling_rate), size=len(firing_times[trial_numbers==tn]))
            random_firing_additions_by_trial = np.append(random_firing_additions_by_trial, random_firing_additions)
        firing_times = firing_times[:len(random_firing_additions_by_trial)] # in cases of any nans
        shuffled_firing_times = firing_times + random_firing_additions_by_trial
        shuffled_firing_times[shuffled_firing_times >= recording_length] = shuffled_firing_times[shuffled_firing_times >= recording_length] - recording_length # wrap around the firing times that exceed the length of the recording
        shuffled_firing_times = (shuffled_firing_times/(sample_rate/downsample_rate)).astype(np.int64) # downsample firing times so we can use the position data instead of the raw position data
        shuffle["firing_times"] = [shuffled_firing_times]

        shuffle_firing = pd.concat([shuffle_firing, shuffle], ignore_index=True)

    shuffle_firing["shuffle_id"] = np.arange(0, n_shuffles)
    shuffle_firing["session_id"] = session_id
    return shuffle_firing

def add_firing_rate_maps_by_trial_type(shuffle_firing, b_trial_numbers, nb_trial_numbers, p_trial_numbers):

    beaconed_maps = []; non_beaconed_maps = []; probe_maps = []
    for shuffle_index, shuffle_id in enumerate(shuffle_firing.shuffle_id):
        shuffle_df = shuffle_firing[shuffle_firing["shuffle_id"]==shuffle_id]
        fr_binned_in_space = np.array(shuffle_df["fr_binned_in_space"].iloc[0])

        b_fr_binned_in_space = fr_binned_in_space[b_trial_numbers-1]
        nb_fr_binned_in_space = fr_binned_in_space[nb_trial_numbers-1]
        p_fr_binned_in_space = fr_binned_in_space[p_trial_numbers-1]

        beaconed_maps.append(np.nanmean(b_fr_binned_in_space, axis=0).tolist())
        non_beaconed_maps.append(np.nanmean(nb_fr_binned_in_space, axis=0).tolist())
        probe_maps.append(np.nanmean(p_fr_binned_in_space, axis=0).tolist())

    shuffle_firing["beaconed_map"] = beaconed_maps
    shuffle_firing["non_beaconed_map"] = non_beaconed_maps
    shuffle_firing["probe_map"] = probe_maps
    return shuffle_firing


def run_shuffle_analysis(spike_data, processed_position_data, position_data, track_length, n_shuffles, by_rewarded):
    position_data = position_data.reset_index(drop=True)

    if by_rewarded:
        processed_position_data = processed_position_data[processed_position_data["rewarded"] == True]

    b_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 0]["trial_number"])
    nb_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 1]["trial_number"])
    p_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 2]["trial_number"])

    shuffled_data = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]

        # shuffle the firing times based on a cyclic shuffling procedure, we add any time between 20 seconds to 10 minutes for all spikes
        shuffle_firing = generate_shuffled_times(cluster_df, n_shuffles=n_shuffles, sample_rate=settings.sampling_rate, downsample_rate=settings.down_sampled_rate)
        shuffle_firing = add_position_x(shuffle_firing, position_data)
        shuffle_firing = add_speed(shuffle_firing, position_data)
        shuffle_firing = add_trial_number(shuffle_firing, position_data)
        shuffle_firing = add_trial_type(shuffle_firing, position_data)
        position_data['dwell_time_ms'] = 1/settings.down_sampled_rate

        shuffle_firing = PostSorting.vr_spatial_firing.bin_fr_in_space(shuffle_firing, position_data, track_length, smoothen=False)
        shuffle_firing = add_firing_rate_maps_by_trial_type(shuffle_firing, b_trial_numbers, nb_trial_numbers, p_trial_numbers)
        shuffle_firing = shuffle_firing[['session_id', 'cluster_id', 'shuffle_id', 'beaconed_map', 'non_beaconed_map', 'probe_map']]

        shuffled_data = pd.concat([shuffled_data, shuffle_firing], ignore_index=True)
        print("finished shuffle for cluster id: ", str(cluster_id))

    return shuffled_data


def process_recordings(vr_recording_path_list, n_shuffles=1000, by_rewarded=True):
    suffix = ""
    if by_rewarded:
        suffix = "_rewarded"

    vr_recording_path_list.sort()
    vr_recording_path_list = vr_recording_path_list[::-1]
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            stop_threshold, track_length = get_stop_threshold_and_track_length(recording)

            # check if shuffle has already ran
            run_shuffle = True
            if os.path.exists(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_by_trial_unsmoothened.pkl"):
                shuffled_data = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_by_trial_unsmoothened.pkl")
                print("there are ", (len(shuffled_data)/len(np.unique(shuffled_data["cluster_id"]))), " shuffles per cell")
                if int(len(shuffled_data)/len(np.unique(shuffled_data["cluster_id"]))) >= n_shuffles:
                    run_shuffle = False

            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            if ((len(spike_data)>0) and (run_shuffle)):
                print("running shuffle analysis on ", recording)
                shuffled_data = run_shuffle_analysis(spike_data, processed_position_data, position_data, track_length, n_shuffles, by_rewarded)
                shuffled_data.to_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_by_trial_unsmoothened.pkl")
            print("successfully processed on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


#  for testing
def main():
    print('-------------------------------------------------------------')
    by_rewarded = True
    n_shuffles = 1000

    # ramp project
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffles, by_rewarded)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffles, by_rewarded)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffles, by_rewarded)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffles, by_rewarded)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffles, by_rewarded)
    print("shuffle_data dataframes have been remade")


if __name__ == '__main__':
    main()
