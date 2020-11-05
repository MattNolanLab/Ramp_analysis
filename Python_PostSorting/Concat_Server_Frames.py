import os
import glob
import pandas as pd
import numpy as np
import pickle

"""

## This script connects to the server (assumes server is mounted locally) and loops through recording folders and concatinates dataframes 

#concat_multi_dir :

Concatinates only the spatial_firing.pkl frames

#concat_multi_spatial_dir:

Appends selected columns from processed_position_data.pkl to spatial_firing.pkl and then appends for each recording folder.
This is so we also have processed position data in the multi frame


## ----------------------------------------------------------- ## 


Required columns to concatinate from server: 

        'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
         'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
         'trial_numbers' 'trial_types' 'rewarded_trials' 'rewarded_stop_locations'
         'spike_num_on_trials' 'spike_rate_on_trials' 'binned_time_ms_per_trial' 
         'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'random_snippets'
        

"""
## ----------------------------------------------------------- ##


def concat_multi_dir(server_path, prm):
    #server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/M1_D31_2018-11-01_12-28-25/parameters.txt'
    server_test_file = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/test/M3_D2_2019-03-05_13-55-31/MountainSort/DataFrames/spatial_firing.pkl'
    server_path = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/M6_sorted/'

    local_output_path = prm.get_output_path() + '/M6_alldays_df.pkl'

    if os.path.exists(server_path):
        print('I found the server.')

    if os.path.exists(server_test_file):
        print('I found the test file.')

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
            if ('spike_rate_on_trials' in spatial_firing) and ('beaconed_trial_number' in spatial_firing):
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'random_snippets', 'firing_times', 'avg_b_spike_rate', 'avg_nb_spike_rate', 'avg_p_spike_rate','x_position_cm', 'spike_num_on_trials', 'b_spike_num_on_trials', 'nb_spike_num_on_trials', 'p_spike_num_on_trials', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'b_spike_rate_on_trials', 'nb_spike_rate_on_trials', 'p_spike_rate_on_trials', 'beaconed_position_cm', 'nonbeaconed_position_cm', 'probe_position_cm', 'beaconed_trial_number', 'nonbeaconed_trial_number', 'probe_trial_number']].copy()

                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(local_output_path)
    return spatial_firing_data



def concat_multi_spatial_dir():
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames'
    server_path = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted/'
    #server_path = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted/'

    if os.path.exists(server_path):
        print('I found the server.')

    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        processed_position_data = pd.DataFrame()
        os.path.isdir(recording_folder)
        spatial_data_frame_path = recording_folder + '/MountainSort/DataFrames/processed_position_data.pkl'
        spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing_all.pkl'

        if os.path.exists(spatial_data_frame_path):

            if os.path.exists(spike_data_frame_path):
                session_id = recording_folder.split('/')[-1]
                print('I found a processed position data frame at ', session_id)
                processed_position = pd.read_pickle(spatial_data_frame_path)
                '''
                
                'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
                'rewarded_stop_locations' 'binned_apsolute_elapsed_time'
                'stop_location_cm' 'stop_trial_number' 'stop_trial_type'
                
                '''
                processed_position = processed_position[['binned_speed_ms', 'rewarded_trials', 'rewarded_stop_locations', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type']].copy()

                processed_position_data = processed_position_data.append({"session_id": session_id,
                                        'binned_speed_ms': np.array(processed_position['binned_speed_ms']),
                                        'rewarded_trials': np.array(processed_position['rewarded_trials']),
                                        'rewarded_locations': np.array(processed_position['rewarded_stop_locations']),
                                        'stop_location_cm': np.array(processed_position['stop_location_cm']),
                                        'stop_trial_number': np.array(processed_position['stop_trial_number']),
                                        'stop_trial_type': np.array(processed_position['stop_trial_type'])}, ignore_index=True)
                print('Position data extracted from frame, loading spatial data...')

                # load spatial firing frame and desired columns
                spatial_firing = pd.read_pickle(spike_data_frame_path)
                '''
                
                'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
                 'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
                 'trial_number' 'trial_type' 'random_snippets' 
                 'spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
                 'position_rate_in_time' 'spike_rate_in_time' 'speed_rate_in_time'
                '''

                print('I found a firing data frame with ', spatial_firing.shape[0], ' cell(s)')
                if (spatial_firing.shape[0] >0 and 'position_rate_in_time' in spatial_firing):
                    #spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'firing_times', 'x_position_cm', 'trial_number', 'trial_type', 'spike_num_on_trials', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'random_snippets', 'spike_rate_in_time', 'position_rate_in_time', 'position_rate_in_time']].copy()
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'mean_firing_rate','firing_times', 'random_snippets','x_position_cm', 'trial_number', 'trial_type','spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'spike_rate_in_time', 'position_rate_in_time', 'speed_rate_in_time']].copy()
                    spatial_firing.reset_index(drop=True, inplace=True)

                    frames= pd.concat([processed_position_data]*(spatial_firing.shape[0]), ignore_index= True)
                    #spatial_firing["binned_time_ms_per_trial"]=frames["binned_time_ms_per_trial"]
                    spatial_firing["binned_speed_ms"]=frames["binned_speed_ms"]
                    spatial_firing["stop_trial_number"]=frames["stop_trial_number"]
                    spatial_firing["stop_location_cm"]=frames["stop_location_cm"]
                    spatial_firing["rewarded_trials"]=frames["rewarded_trials"]
                    spatial_firing["rewarded_locations"]=frames["rewarded_locations"]
                    print('appending data frames ...')

                    spatial_firing_data = spatial_firing_data.append(spatial_firing)
                print(spatial_firing_data.shape[0], ' cells found so far in this mouse')
            else:
                print("Dataframe not found...")

    #spatial_firing.to_pickle(get_output_path() + 'Allmice_alldays_position2_df.pkl')

    #pickle.dump(spatial_firing_data,protocol=pickle.HIGHEST_PROTOCOL)
    spatial_firing_data.to_pickle(local_output_path + '/cohort1/M1_dataset.pkl')
    #pickle.dump(x, fp, protocol = 4)
    #file = local_output_path + '/cohort1/245_new.pkl'
    #pickle.dump(spatial_firing_data,open("file", 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    return spatial_firing_data



## ----------------------------------------------------------- ##

###

## ----------------------------------------------------------- ##


def concat_behaviour_dir():
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames'
    server_path = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted/'

    if os.path.exists(server_path):
        print('I found the server.')

    position_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        processed_position_data = pd.DataFrame()
        os.path.isdir(recording_folder)
        spatial_data_frame_path = recording_folder + '/MountainSort/DataFrames/processed_position_data.pkl'
        #spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing_all.pkl'

        if os.path.exists(spatial_data_frame_path):

        #if os.path.exists(spike_data_frame_path):
            session_id = recording_folder.split('/')[-1]
            print('I found a processed position data frame at ', session_id)
            processed_position = pd.read_pickle(spatial_data_frame_path)
            '''
            
            'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
            'rewarded_stop_locations' 'binned_apsolute_elapsed_time'
            'stop_location_cm' 'stop_trial_number' 'stop_trial_type'
            
            '''
            processed_position = processed_position[['rewarded_trials', 'rewarded_stop_locations', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type']].copy()

            processed_position_data = processed_position_data.append({'session_id': session_id,
                                    'rewarded_trials': np.array(processed_position['rewarded_trials']),
                                    'rewarded_locations': np.array(processed_position['rewarded_stop_locations']),
                                    'stop_location_cm': np.array(processed_position['stop_location_cm']),
                                    'stop_trial_number': np.array(processed_position['stop_trial_number']),
                                    'stop_trial_type': np.array(processed_position['stop_trial_type'])}, ignore_index=True)
                                    #'binned_apsolute_elapsed_time': np.array(processed_position['binned_apsolute_elapsed_time'])}, ignore_index=True)
            print('Position data extracted from frame')
            print('appending data frames ...')
            position_data = position_data.append(processed_position_data)
            print(position_data.shape[0], ' days found so far in this mouse')
        else:
            print("Dataframe not found...")

    #pickle.dump(spatial_firing_data,protocol=pickle.HIGHEST_PROTOCOL)
    position_data.to_pickle(local_output_path + '/cohort4/M2_position_data.pkl')
    #pickle.dump(x, fp, protocol = 4)
    #file = local_output_path + '/cohort1/245_new.pkl'
    #pickle.dump(spatial_firing_data,open("file", 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    return position_data



## ----------------------------------------------------------- ##

###

## ----------------------------------------------------------- ##



def concat_waveform_dir():
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames'
    server_path = '/Volumes/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/VirtualReality/245_sorted/'

    if os.path.exists(server_path):
        print('I found the server.')

    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing_all.pkl'

        if os.path.exists(spike_data_frame_path):
            # load spatial firing frame and desired columns
            spatial_firing = pd.read_pickle(spike_data_frame_path)
            '''
            
            'session_id' 'cluster_id' 'tetrode' 'number_of_spikes' 'mean_firing_rate' 
             'isolation' 'noise_overlap' 'firing_times' 'x_position_cm' 'speed_per200ms'
             'trial_number' 'trial_type' 'random_snippets' 
             'spike_num_on_trials' 'spike_rate_on_trials' 'spike_rate_on_trials_smoothed'
             'position_rate_in_time' 'spike_rate_in_time' 'speed_rate_in_time'
            '''

            print('I found a firing data frame with ', spatial_firing.shape[0], ' cell(s)')
            if (spatial_firing.shape[0] >0 and 'position_rate_in_time' in spatial_firing):
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'mean_firing_rate', 'random_snippets']].copy()
                spatial_firing.reset_index(drop=True, inplace=True)

                print('appending data frames ...')
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
            print(spatial_firing_data.shape[0], ' cells found so far in this mouse')
        else:
            print("Dataframe not found...")


    spatial_firing_data.to_pickle(local_output_path + '/cohort2/245_new.pkl')
    return spatial_firing_data



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    spike_data = concat_multi_spatial_dir()
    #spike_data = concat_behaviour_dir()

    return spike_data

if __name__ == '__main__':
    main()






