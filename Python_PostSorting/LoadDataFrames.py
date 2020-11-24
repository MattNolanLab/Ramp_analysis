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

def process_a_dir(recording_folder, prm):
    #spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing-ramp.pkl'
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/spatial_firing_ramp.pkl'

    if os.path.exists(recording_folder):
        print('I found the test file.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.')
        spike_data = pd.read_pickle(spike_data_frame_path)

        '''
        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
         'isolation' 'noise_overlap' 'peak_snr'  'firing_times' 'avg_b_spike_rate' 
         'avg_nb_spike_rate' 'avg_p_spike_rate' 'x_position_cm' 'beaconed_trial_number'
         'spike_num_on_trials' 'b_spike_num_on_trials' 'nb_spike_num_on_trials'
         'p_spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed' 
         'b_spike_rate_on_trials' 'nb_spike_rate_on_trials' 'p_spike_rate_on_trials'
    
        '''

    return spike_data



def process_a_local_dir(recording_folder, prm):
    #spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing-ramp.pkl'
    spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'

    if os.path.exists(recording_folder):
        print('I found the test file.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.')
        spike_data = pd.read_pickle(spike_data_frame_path)

        '''
        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
         'isolation' 'noise_overlap' 'peak_snr'  'firing_times' 'avg_b_spike_rate' 
         'avg_nb_spike_rate' 'avg_p_spike_rate' 'x_position_cm' 'beaconed_trial_number'
         'spike_num_on_trials' 'b_spike_num_on_trials' 'nb_spike_num_on_trials'
         'p_spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed' 
         'b_spike_rate_on_trials' 'nb_spike_rate_on_trials' 'p_spike_rate_on_trials'
    
        '''

    return spike_data



def process_allmice_dir(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Alldays_cohort1_dataset.pkl'
    #spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/tests/noisy_clusters.pkl'

    #load individual frames for mice - first stop learning curves
    #spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames/cohort1/M1_position_data.pkl'

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


def process_fit_data(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/Teris/selected_beaconed_trials.pkl'

    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a data frame with fits.')
        spike_data = pd.read_pickle(spike_data_frame_path)

    return spike_data


def process_a_position_dir(recording_folder, prm):
    position_data_frame_path = recording_folder + '/DataFrames/processed_position_data.pkl'

    if os.path.exists(recording_folder):
        print('I found the test file.')

    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(position_data_frame_path):
        print('I found a processed spatial data frame.')
        position_data = pd.read_pickle(position_data_frame_path)

        '''
        'binned_time_ms' 'binned_time_moving_ms' 'binned_time_stationary_ms' 'binned_speed_ms' 'beaconed_total_trial_number'
         'nonbeaconed_total_trial_number' 'probe_total_trial_number' 'stop_location_cm' 'stop_trial_number'
         'stop_trial_type' 'rewarded_stop_locations' 'rewarded_trials' 'average_stops' 'position_bins'
    
        '''
    return position_data


def process_raw_position_dir(recording_folder):
    position_data_frame_path = recording_folder + '/Mountainsort/DataFrames/raw_position_data.pkl'

    if os.path.exists(recording_folder):
        print('I found the test file.')

    os.path.isdir(recording_folder)
    if os.path.exists(position_data_frame_path):
        print('I found a raw spatial data frame.')
        position_data = pd.read_pickle(position_data_frame_path)

    return position_data


def process_multi_dir(server_path, prm):
    #server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/M1_D31_2018-11-01_12-28-25/parameters.txt'
    local_output_path = prm.get_output_path() + '/all_mice_df.pkl'

    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            '''
            
            'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
             'number_of_spikes' 'mean_firing_rate' 'isolation' 'noise_overlap' 
             'peak_snr' 'peak_amp' 'random_snippets' 'x_position_cm'
             'trial_number' 'trial_type' 
            
            '''
            if ('spike_rate_on_trials_smoothed' in spatial_firing) and ('spike_rate_in_time' in spatial_firing):
                #spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'random_snippets', 'firing_times', 'avg_b_spike_rate', 'avg_nb_spike_rate', 'avg_p_spike_rate','x_position_cm', 'spike_num_on_trials', 'b_spike_num_on_trials', 'nb_spike_num_on_trials', 'p_spike_num_on_trials', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'b_spike_rate_on_trials', 'nb_spike_rate_on_trials', 'p_spike_rate_on_trials', 'beaconed_position_cm', 'nonbeaconed_position_cm', 'probe_position_cm', 'beaconed_trial_number', 'nonbeaconed_trial_number', 'probe_trial_number']].copy()
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'mean_firing_rate', 'x_position_cm', 'trial_number', 'trial_type', 'spike_rate_on_trials_smoothed', 'spike_rate_in_time', 'position_rate_in_time', 'speed_rate_in_time', 'rewarded_trials', 'rewarded_locations', ]].copy()

                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(local_output_path)
    return spatial_firing_data

'''
## process spatial firing data frame and position dataframe for multiple recordings

1. opens desired columns of position dataframe (processed)
2. opens spatial firing
3. appends columns of position dataframe to spatial firing : 
        - repeats column as a row for as many clusters (rows) are in spatial firing 
4. concatenates resulting spatial firing for multiple recordings

'''


def process_multi_spatial_dir(server_path, prm):
    local_output_path = prm.get_output_path() + '/all_mice_position_df.pkl'

    processed_position_data = pd.DataFrame()
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        spatial_data_frame_path = recording_folder + '/MountainSort/DataFrames/processed_position_data.pkl'
        spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'

        if os.path.exists(spatial_data_frame_path):
            print('I found a processed position data frame.')
            session_id = recording_folder.split('/')[-1]
            processed_position = pd.read_pickle(spatial_data_frame_path)
            '''
            
            'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 
            
            '''
            binned_time = np.array(processed_position['binned_time_ms_per_trial']) # select desired column to concatenate
            processed_position_data = processed_position_data.append({"session_id": session_id, "binned_time_ms_per_trial": binned_time}, ignore_index=True)

            # load spatial firing frame and desired columns
            spatial_firing = pd.read_pickle(spike_data_frame_path)
            '''
            
            'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
             'number_of_spikes' 'mean_firing_rate' 'isolation' 'noise_overlap' 
             'peak_snr' 'peak_amp' 'random_snippets' 'x_position_cm'
             'trial_number' 'trial_type' 
            
            '''
            if ('spike_rate_on_trials' in spatial_firing) and ('beaconed_trial_number' in spatial_firing):
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'random_snippets', 'firing_times', 'avg_b_spike_rate', 'avg_nb_spike_rate', 'avg_p_spike_rate','x_position_cm', 'spike_num_on_trials', 'b_spike_num_on_trials', 'nb_spike_num_on_trials', 'p_spike_num_on_trials', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'b_spike_rate_on_trials', 'nb_spike_rate_on_trials', 'p_spike_rate_on_trials', 'beaconed_position_cm', 'nonbeaconed_position_cm', 'probe_position_cm', 'beaconed_trial_number', 'nonbeaconed_trial_number', 'probe_trial_number']].copy()
                spatial_firing.reset_index(drop=True, inplace=True)

                frames= pd.concat([processed_position_data]*(spatial_firing.shape[0]), ignore_index= True)
                spatial_firing["binned_time_ms_per_trial"]=frames["binned_time_ms_per_trial"]

                spatial_firing_data = spatial_firing_data.append(spatial_firing)
                print(spatial_firing_data['binned_time_ms_per_trial'].head())

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(local_output_path)
    return spatial_firing_data


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


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25' # test recording
    local_output_path = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25/'
    print('Processing ' + str(recording_folder))

    spike_data = process_a_dir(recording_folder, local_output_path)



if __name__ == '__main__':
    main()
