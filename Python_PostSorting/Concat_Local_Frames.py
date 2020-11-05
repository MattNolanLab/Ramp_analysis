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
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/Alldays_cohort4_dataset.pkl'
    frames_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames/cohort4/'
    #frames_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/WholeFrame/Merged_frame/all/'
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
            #spatial_firing = spatial_firing[['session_id', 'cluster_id', 'mean_firing_rate', 'isolation', 'noise_overlap', 'firing_times', 'x_position_cm', 'speed_per200ms', 'trial_number', 'trial_type', 'spike_num_on_trials', 'spike_rate_on_trials', 'spike_rate_on_trials_smoothed', 'random_snippets', 'binned_time_ms_per_trial', 'binned_speed_ms_per_trial', 'rewarded_trials', 'rewarded_locations', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type']].copy()
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


def drop_columns_from_frame(spike_data):
    #raw_position_data.drop(['time_seconds'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['avg_b_spike_rate'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['avg_nb_spike_rate'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['avg_p_spike_rate'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['beaconed_trial_number'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['x_position_cm'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['firing_times'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['spike_rate_on_trials_smoothed'], axis='columns', inplace=True, errors='ignore')
    return spike_data






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

    #spike_data = drop_columns_from_frame(spike_data)
    #spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/Allmice_alldays_finaldf.pkl')

    #return spatial_data

if __name__ == '__main__':
    main()





def concat_spike_and_spatial_dir(server_path):
    local_output_path = '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/Individual_Mouse_Frames'


    if os.path.exists(server_path):
        print('I found the server.')

    spatial_firing_data = pd.DataFrame()
    processed_position_data = pd.DataFrame()

    spatial_data_frame_path = server_path + '/Mountainsort/DataFrames/processed_position_data.pkl'
    spike_data_frame_path = server_path + '/Mountainsort/DataFrames/spatial_firing_ramp.pkl'

    if os.path.exists(spatial_data_frame_path):
        session_id = server_path.split('/')[-1]
        print('I found a processed position data frame at ', session_id)
        processed_position = pd.read_pickle(spatial_data_frame_path)
        '''
        
        'binned_time_ms_per_trial' 'binned_speed_ms_per_trial' 'rewarded_trials'
        'rewarded_stop_locations' 'binned_apsolute_elapsed_time'
        'stop_location_cm' 'stop_trial_number' 'stop_trial_type'
        
        '''
        processed_position = processed_position[['binned_time_ms_per_trial', 'binned_speed_ms_per_trial', 'rewarded_trials', 'rewarded_stop_locations', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type']].copy()

        processed_position_data = processed_position_data.append({"session_id": session_id,
                                'binned_time_ms_per_trial': np.array(processed_position['binned_time_ms_per_trial']),
                                "binned_speed_ms_per_trial": np.array(processed_position['binned_speed_ms_per_trial']),
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
        if spatial_firing.shape[0] >0:
            spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'firing_times', 'x_position_cm', 'trial_number', 'trial_type', 'random_snippets', 'spike_rate_on_trials_smoothed']].copy()
            spatial_firing.reset_index(drop=True, inplace=True)

            frames= pd.concat([processed_position_data]*(spatial_firing.shape[0]), ignore_index= True)
            spatial_firing["binned_time_ms_per_trial"]=frames["binned_time_ms_per_trial"]
            spatial_firing["rewarded_trials"]=frames["rewarded_trials"]
            spatial_firing["rewarded_locations"]=frames["rewarded_locations"]
            spatial_firing["stop_location_cm"]=frames["stop_location_cm"]
            spatial_firing["stop_trial_number"]=frames["stop_trial_number"]
            spatial_firing["stop_trial_type"]=frames["stop_trial_type"]
            spatial_firing["binned_speed_ms_per_trial"]=frames["binned_speed_ms_per_trial"]
            print('appending data frames ...')

            spatial_firing_data = spatial_firing_data.append(spatial_firing)
            print(spatial_firing_data.shape[0], ' cells found so far in this mouse')

    #spatial_firing.to_pickle(get_output_path() + 'Allmice_alldays_position2_df.pkl')

    #spatial_firing_data.to_pickle(local_output_path + '/cohort1/test.pkl')
    #pickle.dump(x, fp, protocol = 4)
    return spatial_firing_data
    

