import os
import glob
import pandas as pd
import numpy as np
import csv

"""
Required columns to concatinate from server: 

        '''
        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate'
        'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
        'trial_number' 'trial_type' 'random_snippets'
        'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
        'rewarded_trials' 'rewarded_stop_locations' 'spike_rate_in_time'

        

"""


def process_allmice_dir(recording_folder, prm):
    #spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/concatenated_spike_data_cohort2.pkl'
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Concat_cohort4_sarah.pkl'

    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.'),
        spike_data = pd.read_pickle(spike_data_frame_path)

        '''
        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate'
        'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
        'trial_number' 'trial_type' 'random_snippets' 'binned_speed_ms_per_trial'
        'spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
        'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
        'rewarded_stop_locations' 'position_rate_in_time' 'spike_rate_in_time'
        'speed_rate_in_time' 'binned_apsolute_elapsed_time'
                
        '''
    return spike_data



def process_allmice_dir_of(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/combined_Cohort3.pkl'


    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.'),
        spike_data = pd.read_pickle(spike_data_frame_path)

    return spike_data

