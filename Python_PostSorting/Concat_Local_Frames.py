import os
import glob
import pandas as pd
import numpy as np



"""

## This script concatinates whole mouse frames (each frame with all days from one mouse) into an overall dataframe for all mice and all days. 

note: assumßes Concat_Server_Frames.py or equivelant has been ran to obtain whole mouse frames

local folder with individual mouse frames: /Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames/
output folder for all mice, all days frame: /Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame


"""


def concat_all_mice_dir():
    local_output_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Alldays_cohort1_dataset.pkl'
    frames_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/IndividualFrames/c3/'

    allmice_data = pd.DataFrame()
    if os.path.exists(frames_path):
        print('I found the data frames.')

    for frame in glob.glob(frames_path + '*'):
        os.path.isdir(frame)
        if os.path.exists(frame):
            spatial_firing = pd.read_pickle(frame)
            '''
            
            'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate'
            'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
            'trial_number' 'trial_type' 'random_snippets' 'binned_speed_ms_per_trial'
            'spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
            'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
            'rewarded_stop_locations'
            
            '''
            print('I found a firing data frame with ', spatial_firing.shape[0], ' cell(s)')
            spatial_firing = spatial_firing[['session_id', 'cluster_id', 'mean_firing_rate', 'firing_times', 'random_snippets', 'x_position_cm', 'trial_number', 'trial_type', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'spike_rate_in_time', 'position_rate_in_time', 'speed_rate_in_time', 'rewarded_trials', 'rewarded_locations', 'stop_location_cm', 'stop_trial_number']].copy()
            allmice_data = allmice_data.append(spatial_firing)

    allmice_data.to_pickle(local_output_path)
    return allmice_data


def remove_false_positives(df):
    df["max_trial_number"] = ""
    for cluster in range(len(df)):
        df.at[cluster,"max_trial_number"] = max(df.loc[cluster].spike_rate_on_trials_smoothed[1])
    df = df.drop(df[df.max_trial_number < 15].index)
    df = df.dropna(axis=0)
    return df


def concat_all_mice_spatial_dir():
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/Alldays_cohort4_2.pkl'
    frames_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames/Spatial_frames/cohort4/'

    allmice_data = pd.DataFrame()

    if os.path.exists(frames_path):
        print('I found the data frames.')

    for frame in glob.glob(frames_path + '*'):
        os.path.isdir(frame)
        if os.path.exists(frame):
            spatial_firing = pd.read_pickle(frame)
            '''
            
            'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate'
            'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
            'trial_number' 'trial_type' 'random_snippets' 'binned_speed_ms_per_trial'
            'spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
            'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
            'rewarded_stop_locations'
            
            '''
            print('I found a spatial data frame with ', spatial_firing.shape[0], ' day(s)')
            spatial_firing = spatial_firing[['rewarded_trials', 'rewarded_locations', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type']].copy()
            allmice_data = allmice_data.append(spatial_firing)

    allmice_data.to_pickle(local_output_path)
    return allmice_data



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    ### For clusters
    spike_data = concat_all_mice_dir()

    spike_data.reset_index(drop=True, inplace=True)
    spike_data=remove_false_positives(spike_data)
    spike_data.reset_index(drop=True, inplace=True)

    ### For spatial data
    #concat_all_mice_spatial_dir()

    #spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/Allmice_alldays_finaldf.pkl')


if __name__ == '__main__':
    main()

