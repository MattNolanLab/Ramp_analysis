import os
import glob
import pandas as pd
import numpy as np
import csv

"""
Required columns to concatinate from server: 

        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
         'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
         'trial_numbers' 'trial_types' 'rewarded_trials' 'rewarded_stop_locations'
         'spike_num_on_trials' 'spike_rate_on_trials' 'binned_time_ms_per_trial' 
         'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 
        

"""


def process_allmice_dir(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/c4_m2_position.pkl'
    #spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Alldays_cohort4_dataset.pkl'

    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.')
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


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] +  '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice

