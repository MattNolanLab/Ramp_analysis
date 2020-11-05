import os
import pickle

def save_data_frames(prm, spatial_firing, processed_position_data):
    if os.path.exists(prm.get_local_recording_folder_path() + '/DataFrames') is False:
        os.makedirs(prm.get_local_recording_folder_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/spatial_firing_ramp.pkl')
    processed_position_data.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/processed_position_data.pkl')



def save_multi_dataframe(prm, spatial_firing):
    #spatial_firing.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/spatial_firing.pkl')
    spatial_firing.to_pickle(prm.get_output_path() + '/all_mice_df.pkl')


def save_multi_position_dataframe(prm, spatial_firing):
    #spatial_firing.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/spatial_firing.pkl')
    spatial_firing.to_pickle(prm.get_output_path() + 'Allmice_alldays_position2_df.pkl')
    #pickle.dump(spatial_firing, f, protocol=pickle.HIGHEST_PROTOCOL)
