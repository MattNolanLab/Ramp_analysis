import os
import glob
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
This file creates a concatenated dataframe of all the recording directories passed to it and saves it where specified,
for collection of the processed position data for the vr side, there will be a link pointing to the original processed position
"""
def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags

def process_running_parameter_tag(running_parameter_tags):
    stop_threshold = 4.7  # defaults
    track_length = 200 # default assumptions
    cue_conditioned_goal = False

    if not running_parameter_tags:
        return stop_threshold, track_length, cue_conditioned_goal

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('stop_threshold'):
            stop_threshold = float(tag.split("=")[1])
        elif tag.startswith('track_length'):
            track_length = int(tag.split("=")[1])
        elif tag.startswith('cue_conditioned_goal'):
            cue_conditioned_goal = bool(tag.split('=')[1])
        else:
            #print('Unexpected / incorrect tag in the third line of parameters file')
            unexpected_tag = True
    return stop_threshold, track_length, cue_conditioned_goal

def load_processed_position_data_collumns(spike_data, processed_position_data):
    for collumn in list(processed_position_data):
        collumn_data = processed_position_data[collumn].tolist()
        spike_data[collumn] = [collumn_data for x in range(len(spike_data))]
    return spike_data

def add_nested_time_binned_data(spike_data, processed_position_data):

    nested_lists = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        rates_ = []
        speeds_ = []
        positions_ = []
        trial_numbers_ = []
        trial_types_ = []
        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]
            trial_type = trial_proccessed_position_data["trial_type"].iloc[0]

            speed_o = pd.Series(trial_proccessed_position_data['speeds_binned_in_time'].iloc[0])
            position_o = pd.Series(trial_proccessed_position_data['pos_binned_in_time'].iloc[0])
            #acceleration_o = pd.Series(trial_proccessed_position_data['acc_binned_in_time'].iloc[0])
            rates_o = pd.Series(cluster_spike_data['fr_time_binned'].iloc[0][trial_number-1])

            if len(speed_o)>0: # add a catch for nans?

                # remove outliers
                rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                #acceleration = acceleration_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers

                # make trial type, trial number and whether it was a rewarded trial into longform
                trial_numbers = np.repeat(trial_number, len(rates))
                trial_types = np.repeat(trial_type, len(rates))

                rates_.extend(rates.tolist())
                speeds_.extend(speed.tolist())
                positions_.extend(position.tolist())
                trial_numbers_.extend(trial_numbers.tolist())
                trial_types_.extend(trial_types.tolist())

        spikes_in_time = [np.array(rates_), np.array(speeds_),
                          np.array(positions_), np.array(trial_numbers_), np.array(trial_types_)]

        nested_lists.append(spikes_in_time)

    spike_data["spike_rate_in_time"] = nested_lists

    return spike_data


def add_nested_space_binned_data(spike_data, processed_position_data):

    nested_lists = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        rates_ = []
        trial_numbers_ = []
        trial_types_ = []

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]

            rates = np.array(cluster_spike_data['fr_binned_in_space'].iloc[0][trial_number-1])
            trial_numbers = np.repeat(trial_number, len(rates))
            trial_types = np.repeat(trial_proccessed_position_data["trial_type"].iloc[0], len(rates))

            rates_.extend(rates.tolist())
            trial_numbers_.extend(trial_numbers.tolist())
            trial_types_.extend(trial_types.tolist())

        spikes_in_space = [np.array(rates_), np.array(trial_numbers_), np.array(trial_types_)]

        nested_lists.append(spikes_in_space)

    spike_data["spike_rate_on_trials_smoothed"] = nested_lists
    return spike_data

def add_stop_variables(spike_data, processed_position_data):
    rewarded_locations_clusters = []
    rewarded_trials_clusters = []
    rewarded_trial_types_clusters = []

    stop_locations_clusters = []
    stop_trials_clusters = []
    stop_trial_types_clusters = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        rewarded_locations = []
        rewarded_trials = []
        rewarded_trial_types = []

        stop_locations = []
        stop_trials = []
        stop_trial_type = []

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]
            trial_type = trial_proccessed_position_data["trial_type"].iloc[0]
            rewarded = trial_proccessed_position_data["rewarded"].iloc[0]

            # get rewarded stops and all stops in a given trial
            trial_rewarded_locations = trial_proccessed_position_data["reward_stop_location_cm"].iloc[0]
            trial_stop_locations = trial_proccessed_position_data["stop_location_cm"].iloc[0]

            # append stops, trial number, types and the same for rewarded stops
            stop_trials.extend(np.repeat(trial_number, len(trial_stop_locations)).tolist())
            stop_trial_type.extend(np.repeat(trial_type, len(trial_stop_locations)).tolist())
            stop_locations.extend(trial_stop_locations.tolist())
            if rewarded:
                rewarded_trials.append(trial_number)
                rewarded_trial_types.append(trial_type)
                rewarded_locations.append(trial_rewarded_locations[0])

        # append to cluster lists
        stop_trials_clusters.append(stop_trials)
        stop_trial_types_clusters.append(stop_trial_type)
        stop_locations_clusters.append(stop_locations)
        rewarded_trials_clusters.append(rewarded_trials)
        rewarded_trial_types_clusters.append(rewarded_trial_types)
        rewarded_locations_clusters.append(rewarded_locations)

    spike_data["stop_trials"] = stop_trials_clusters
    spike_data["stop_trial_types"] = stop_trial_types_clusters
    spike_data["stop_locations"] = stop_locations_clusters
    spike_data["rewarded_trials"] = rewarded_trials_clusters
    spike_data["rewarded_trial_types"] = rewarded_trials_clusters
    spike_data["rewarded_locations"] = rewarded_locations_clusters

    return spike_data

def remove_cluster_without_firing_events(spike_data):
    '''
    Removes rows where no firing times are found, this occurs when spikes are found in one session type and not the
    other when multiple sessions are spike sorted together
    '''

    spike_data_filtered = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]
        firing_times = cluster_spike_data["firing_times"].iloc[0]
        if len(firing_times)>0:
            spike_data_filtered = pd.concat([spike_data_filtered, cluster_spike_data])
        else:
            print("I am removing cluster ", cluster_id, " from this recording")
            print("because it has no firing events in this spatial firing dataframe")

    return spike_data_filtered


def process_dir(recordings_path, concatenated_spike_data=None, save_path=None, track_length=200):

    """
    Creates a dataset with spike data for ramp analysis and modelling

    :param recordings_path: path for a folder with all the recordings you want to process
    :param concatenated_spike_data: a pandas dataframe to append all processsed spike data to
    :param save_path: where we save the new processed spike data
    :return: processed spike data
    """

    # make an empty dataframe if concatenated frame given as none
    if concatenated_spike_data is None:
        concatenated_spike_data = pd.DataFrame()

    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recordings_path) if f.is_dir()]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        print("processing ", recording.split("/")[-1])

        spatial_dataframe_path = recording + '/MountainSort/DataFrames/processed_position_data.pkl'
        spike_dataframe_path = recording + '/MountainSort/DataFrames/spatial_firing.pkl'
        recording_stop_threshold, recording_track_length, _ = process_running_parameter_tag(get_tags_parameter_file(recording))

        mouse_id = str(recording.split("/")[-1].split("_")[0])

        # only take recordings with the given track length
        if int(track_length) == int(recording_track_length):

            if os.path.exists(spike_dataframe_path):
                spike_data = pd.read_pickle(spike_dataframe_path)

                if (len(spike_data)==0):
                    print("this recording has no units, ", recording.split("/")[-1])
                else:
                    spike_data = remove_cluster_without_firing_events(spike_data)

                    if os.path.exists(spatial_dataframe_path):
                        processed_position_data = pd.read_pickle(spatial_dataframe_path)

                        # look for key columns needed for ramp analysis
                        if ("fr_time_binned" in list(spike_data)) or ("fr_binned_in_space" in list(spike_data)):

                            spike_data = add_nested_time_binned_data(spike_data, processed_position_data)
                            spike_data = add_nested_space_binned_data(spike_data, processed_position_data)
                            spike_data = add_stop_variables(spike_data, processed_position_data)
                            spike_data["mouse_id"] = np.repeat(mouse_id, len(spike_data)).tolist()

                            columns_to_drop = ['all_snippets', 'beaconed_position_cm', 'beaconed_trial_number',
                                               'nonbeaconed_position_cm', 'nonbeaconed_trial_number', 'probe_position_cm',
                                               'probe_trial_number', 'beaconed_firing_rate_map', 'non_beaconed_firing_rate_map',
                                               'probe_firing_rate_map', 'beaconed_firing_rate_map_sem', 'non_beaconed_firing_rate_map_sem',
                                               'probe_firing_rate_map_sem']
                            for column in columns_to_drop:
                                if column in list(spike_data):
                                    del spike_data[column]

                            concatenated_spike_data = pd.concat([concatenated_spike_data, spike_data], ignore_index=True)

                        else:
                            print("could not find correct binned column in recording ", recording.split("/")[-1])
                    else:
                        print("couldn't find processed_position for ", recording.split("/")[-1])

    if save_path is not None:
        concatenated_spike_data.to_pickle(save_path+"concatenated_spike_data.pkl")

    return concatenated_spike_data

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # process individual recordings into a concatenated dataframe by cohort
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality", concatenated_spike_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort5_", track_length=200)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality", concatenated_spike_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort4_", track_length=200)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality", concatenated_spike_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort3_", track_length=200)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality", concatenated_spike_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort2_", track_length=200)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr",                              concatenated_spike_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort7_", track_length=200)
    spike_data = process_dir(recordings_path= "", concatenated_spike_data=None, save_path=None, track_length=200)
    print("were done for now ")

if __name__ == '__main__':
    main()
