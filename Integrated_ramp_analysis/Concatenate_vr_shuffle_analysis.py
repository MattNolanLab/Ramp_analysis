import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
import pyarrow.feather as feather
import pickle5 as pickle
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


def calculation_slopes(shuffled_firing, track_region):
    if track_region == "ob":
        start = 30; end=90;
    elif track_region == "hb":
        start = 110; end=170;

    beaconed_r_squared_values = []; beaconed_slope_values = []; beaconed_p_values = []; beaconed_intercept_values = []
    non_beaconed_r_squared_values = []; non_beaconed_slope_values = []; non_beaconed_p_values = []; non_beaconed_intercept_values = []
    probe_r_squared_values = []; probe_slope_values = []; probe_p_values = []; probe_intercept_values = []

    for index, row in shuffled_firing.iterrows():
        data = row.to_frame().T.reset_index(drop=True)

        # beaconed outbound
        y = np.array(data["beaconed_map"].iloc[0])
        if not np.isnan(y).any():
            x = np.arange(1, len(y)+1) # Position
            slope, intercept, r, p, se = stats.linregress(x[start:end], y[start:end])
            beaconed_r_sq = np.square(r)
            beaconed_slope = slope
            beaconed_pval = p
            beaconed_intercept = intercept
        else:
            beaconed_r_sq = np.nan
            beaconed_slope = np.nan
            beaconed_pval = np.nan
            beaconed_intercept = np.nan

            # non beaconed outbound
        y = np.array(data["non_beaconed_map"].iloc[0])
        if not np.isnan(y).any():
            x = np.arange(1, len(y)+1) # Position
            slope, intercept, r, p, se = stats.linregress(x[start:end], y[start:end])
            non_beaconed_r_sq = np.square(r)
            non_beaconed_slope = slope
            non_beaconed_pval = p
            non_beaconed_intercept = intercept
        else:
            non_beaconed_r_sq = np.nan
            non_beaconed_slope = np.nan
            non_beaconed_pval = np.nan
            non_beaconed_intercept = np.nan

        # probe outbound
        y = np.array(data["probe_map"].iloc[0])
        if not np.isnan(y).any():
            x = np.arange(1, len(y)+1) # Position
            slope, intercept, r, p, se = stats.linregress(x[start:end], y[start:end])
            probe_r_sq = np.square(r)
            probe_slope = slope
            probe_pval = p
            probe_intercept = intercept
        else:
            probe_r_sq = np.nan
            probe_slope = np.nan
            probe_pval = np.nan
            probe_intercept = np.nan

        beaconed_r_squared_values.append(beaconed_r_sq)
        beaconed_slope_values.append(beaconed_slope)
        beaconed_p_values.append(beaconed_pval)
        beaconed_intercept_values.append(beaconed_intercept)
        non_beaconed_r_squared_values.append(non_beaconed_r_sq)
        non_beaconed_slope_values.append(non_beaconed_slope)
        non_beaconed_p_values.append(non_beaconed_pval)
        non_beaconed_intercept_values.append(non_beaconed_intercept)
        probe_r_squared_values.append(probe_r_sq)
        probe_slope_values.append(probe_slope)
        probe_p_values.append(probe_pval)
        probe_intercept_values.append(probe_intercept)

    shuffled_firing["beaconed_r2_"+track_region] = beaconed_r_squared_values
    shuffled_firing["beaconed_slope_"+track_region] = beaconed_slope_values
    shuffled_firing["beaconed_p_val_"+track_region] = beaconed_p_values
    shuffled_firing["beaconed_intercept_"+track_region] = beaconed_intercept_values
    shuffled_firing["non_beaconed_r2_"+track_region] = non_beaconed_r_squared_values
    shuffled_firing["non_beaconed_slope_"+track_region] = non_beaconed_slope_values
    shuffled_firing["non_beaconed_p_val_"+track_region] = non_beaconed_p_values
    shuffled_firing["non_beaconed_intercept_"+track_region] = non_beaconed_intercept_values
    shuffled_firing["probe_r2_"+track_region] = probe_r_squared_values
    shuffled_firing["probe_slope_"+track_region] = probe_slope_values
    shuffled_firing["probe_p_val_"+track_region] = probe_p_values
    shuffled_firing["probe_intercept_"+track_region] = probe_intercept_values
    return shuffled_firing

def process_dir(recordings_path, concatenated_shuffle_data=None, save_path=None, by_rewarded=True):
    suffix = ""
    if by_rewarded:
        suffix = "_rewarded"

    """
    Creates a dataset with spike data for ramp analysis and modelling

    :param recordings_path: path for a folder with all the recordings you want to process
    :param concatenated_spike_data: a pandas dataframe to append all processsed spike data to
    :param save_path: where we save the new processed spike data
    :return: processed spike data
    """

    # make an empty dataframe if concatenated frame given as none
    if concatenated_shuffle_data is None:
        concatenated_shuffle_data = pd.DataFrame()

    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recordings_path) if f.is_dir()]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        print("processing ", recording.split("/")[-1])
        shuffled_firing_rate_maps_path = recording + '/MountainSort/DataFrames/shuffled_firing_rate_maps'+suffix+'_by_trial_unsmoothened.pkl'

        # report if its expected for a recording to have a shuffled dataset (only if there are cells in spatial firing to shuffle)
        spike_data = pd.read_pickle(recording + '/MountainSort/DataFrames/spatial_firing.pkl')
        if len(spike_data)>0:
            has_cells = True
            #print(recording, " should have a shuffled dataset")
        else:
            has_cells = False
            #print(recording, " won't have a shuffled dataset")

        # calculate slope, r2 and p values if shuffled rate maps exist
        if os.path.exists(shuffled_firing_rate_maps_path):
            has_shuffle=True
            shuffled_firing = pd.read_pickle(shuffled_firing_rate_maps_path)
            #print("there are ", (len(shuffled_firing)/len(np.unique(shuffled_firing["cluster_id"]))), " shuffles per cell")

            shuffled_firing = calculation_slopes(shuffled_firing, track_region="ob")
            shuffled_firing = calculation_slopes(shuffled_firing, track_region="hb")
            #shuffled_firing.drop(columns=['beaconed_map', 'non_beaconed_map', 'probe_map'])
            shuffled_firing.drop(columns=['non_beaconed_map', 'probe_map'])
            concatenated_shuffle_data = pd.concat([concatenated_shuffle_data, shuffled_firing], ignore_index=True)
        else:
            has_shuffle=False

        if has_cells and not has_shuffle:
            print(recording, " still needs to be run")

    if save_path is not None:
        concatenated_shuffle_data = concatenated_shuffle_data[['session_id', 'cluster_id', 'shuffle_id',
                                                               'beaconed_r2_ob', 'beaconed_slope_ob', 'beaconed_p_val_ob', 'beaconed_r2_hb', 'beaconed_slope_hb', 'beaconed_p_val_hb',
                                                               'non_beaconed_r2_ob', 'non_beaconed_slope_ob', 'non_beaconed_p_val_ob', 'non_beaconed_r2_hb', 'non_beaconed_slope_hb', 'non_beaconed_p_val_hb',
                                                               'probe_r2_ob', 'probe_slope_ob', 'probe_p_val_ob', 'probe_r2_hb', 'probe_slope_hb', 'probe_p_val_hb', 'beaconed_map']]
        concatenated_shuffle_data.to_pickle(save_path+"concatenated_shuffle_data"+suffix+"_unsmoothened.pkl")
    print("finished processing, ", recordings_path)
    return concatenated_shuffle_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    by_rewarded=True

    # ramp cell project
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality", concatenated_shuffle_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c5_", by_rewarded=by_rewarded)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality", concatenated_shuffle_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c4_", by_rewarded=by_rewarded)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality", concatenated_shuffle_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c3_", by_rewarded=by_rewarded)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality", concatenated_shuffle_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c2_", by_rewarded=by_rewarded)
    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr", concatenated_shuffle_data=None, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c7_", by_rewarded=by_rewarded)
    
    # save all to a single dataframe
    all_mice = pd.DataFrame()
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c2_concatenated_shuffle_data_rewarded_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c3_concatenated_shuffle_data_rewarded_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c4_concatenated_shuffle_data_rewarded_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c5_concatenated_shuffle_data_rewarded_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/c7_concatenated_shuffle_data_rewarded_unsmoothened.pkl")])
    all_mice.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/all_mice_concatenated_shuffle_data_rewarded_unsmoothened.pkl")

    # also save as feather file for import to R
    with open("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/all_mice_concatenated_shuffle_data_rewarded_unsmoothened.pkl", "rb") as fh:
        data = pickle.load(fh)
        feather.write_feather(data, "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/all_mice_concatenated_shuffle_data_rewarded_unsmoothened.feather")
    print("were done for now ")


if __name__ == '__main__':
    main()
