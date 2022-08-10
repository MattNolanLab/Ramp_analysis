import pandas as pd
import numpy as np
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters
import os

prm = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters.Parameters()


def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)


def process_allmice_of(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/combined_Cohort7.pkl'
    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.'),
        spike_data = pd.read_pickle(spike_data_frame_path)
    return spike_data


def process_allmice_vr(recording_folder, prm):
    spike_data_frame_path = '/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Processed_cohort7_sarah.pkl'
    if os.path.exists(prm.get_output_path()):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.'),
        spike_data = pd.read_pickle(spike_data_frame_path)
    return spike_data


def add_date_to_frame(df):
    df["Date"] = ""
    for cluster in range(len(df)):
        session_id = df.session_id.values[cluster]
        date = session_id.rsplit("_", 6)[2]
        df.at[cluster,"Date"] = date
    return df


def add_mouse_to_frame(df):
    print("Adding ID for Mouse and Day to dataframe...")
    df["Mouse"] = ""
    #df["Cohort"] = ""
    df["Day"] = ""
    df["Day_numeric"] = ""
    for cluster in range(len(df)):
        session_id = df.session_id.values[cluster]
        numericday, day, mouse = extract_mouse_and_day(session_id)
        df.at[cluster,"Mouse"] = mouse
        df.at[cluster,"Day"] = day
        df.at[cluster,"Day_numeric"] = numericday
        #df.at[cluster,"cohort"] = 5
    return df

# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse


def load_openfield_data_into_frame(of_data, vr_data):
    print('I am loading open field data into frame ...')
    vr_data["speed_score"] = ""
    vr_data["speed_score_p_values"] = ""
    vr_data["hd_score"] = ""
    vr_data["rayleigh_score"] = ""
    vr_data["spatial_information_score"] = ""
    vr_data["grid_score"] = ""
    vr_data["border_score"] = ""
    vr_data["speed_threshold_pos"] = ""
    vr_data["speed_threshold_neg"] = ""
    vr_data["hd_threshold"] = ""
    vr_data["border_threshold"] = ""
    vr_data["rayleigh_threshold"] = ""
    vr_data["spatial_threshold"] = ""
    vr_data["grid_threshold"] = ""
    vr_data["mean_firing_rate_of"] = ""
    vr_data["spike_width"] = ""

    for cluster in range(len(vr_data)):
        session_id=vr_data.at[cluster, "Date"]
        cluster_id=vr_data.at[cluster, "cluster_id"]
        mouse=vr_data.at[cluster, "Mouse"]
        day=vr_data.at[cluster, "Day"]

        mouse_data = of_data['Mouse'] == mouse
        mouse_data = of_data[mouse_data]

        day_data = mouse_data['Day'] == day
        day_data = mouse_data[day_data]

        #find data for that neuron
        session_fits = day_data['Date'] == session_id
        session_fits = day_data[session_fits]
        # find that neuron
        neuron_fits = session_fits['cluster_id'] == cluster_id
        neuron_fits = session_fits[neuron_fits]

        neurons = neuron_fits.shape[0]
        if neurons > 0:
            speed_score = neuron_fits['speed_score'].values # extract fit
            speed_score_p_values = neuron_fits['speed_score_p_values'].values # extract fit
            rayleigh_score = neuron_fits['rayleigh_score'].values # extract fit
            hd_score = neuron_fits['hd_score'].values # extract fit
            spatial_information_score = neuron_fits['spatial_information_score'].values # extract fit
            grid_score = neuron_fits['grid_score'].values # extract fit
            border_score = neuron_fits['border_score'].values # extract fit
            speed_threshold_pos = neuron_fits['speed_threshold_pos'].values # extract fit
            speed_threshold_neg = neuron_fits['speed_threshold_neg'].values # extract fit
            hd_threshold = neuron_fits['hd_threshold'].values # extract fit
            rayleigh_threshold = neuron_fits['rayleigh_threshold'].values # extract fit
            spatial_threshold = neuron_fits['spatial_threshold'].values # extract fit
            grid_threshold = neuron_fits['grid_threshold'].values # extract fit
            border_threshold = neuron_fits['border_threshold'].values # extract fit
            mean_firing_rate_of = neuron_fits['mean_firing_rate_of'].values # extract fit
            spike_width = neuron_fits['spike_width'].values # extract fit

            vr_data.at[cluster,"speed_score"] = speed_score[0]
            vr_data.at[cluster,"speed_score_p_values"] = speed_score_p_values[0]
            vr_data.at[cluster,"hd_score"] = hd_score[0]
            vr_data.at[cluster,"rayleigh_score"] = rayleigh_score[0]
            vr_data.at[cluster,"spatial_information_score"] = spatial_information_score[0]
            vr_data.at[cluster,"grid_score"] = grid_score[0]
            vr_data.at[cluster,"border_score"] = border_score[0]
            vr_data.at[cluster,"speed_threshold_pos"] = speed_threshold_pos[0]
            vr_data.at[cluster,"speed_threshold_neg"] = speed_threshold_neg[0]
            vr_data.at[cluster,"hd_threshold"] = hd_threshold[0]
            vr_data.at[cluster,"rayleigh_threshold"] = rayleigh_threshold[0]
            vr_data.at[cluster,"spatial_threshold"] = spatial_threshold[0]
            vr_data.at[cluster,"grid_threshold"] = grid_threshold[0]
            vr_data.at[cluster,"border_threshold"] = border_threshold[0]
            vr_data.at[cluster,"mean_firing_rate_of"] = mean_firing_rate_of[0]
            vr_data.at[cluster,"spike_width"] = spike_width[0]

        else:
            vr_data.at[cluster,"speed_score"] = np.nan
            vr_data.at[cluster,"speed_score_p_values"] = np.nan
            vr_data.at[cluster,"hd_score"] = np.nan
            vr_data.at[cluster,"rayleigh_score"] = np.nan
            vr_data.at[cluster,"spatial_information_score"] = np.nan
            vr_data.at[cluster,"grid_score"] = np.nan
            vr_data.at[cluster,"border_score"] = np.nan
            vr_data.at[cluster,"speed_threshold_pos"] = np.nan
            vr_data.at[cluster,"speed_threshold_neg"] = np.nan
            vr_data.at[cluster,"hd_threshold"] = np.nan
            vr_data.at[cluster,"rayleigh_threshold"] = np.nan
            vr_data.at[cluster,"spatial_threshold"] = np.nan
            vr_data.at[cluster,"grid_threshold"] = np.nan
            vr_data.at[cluster,"border_threshold"] = np.nan
            vr_data.at[cluster,"mean_firing_rate_of"] = np.nan
            vr_data.at[cluster,"spike_width"] = np.nan

    print('finished loading open field data into frame ...')
    return vr_data



def add_OF_stats_as_nans(vr_data):
    vr_data["speed_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["speed_score_p_values"] = np.repeat(np.nan, len(vr_data))
    vr_data["hd_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["rayleigh_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["spatial_information_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["grid_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["border_score"] = np.repeat(np.nan, len(vr_data))
    vr_data["speed_threshold_pos"] = np.repeat(np.nan, len(vr_data))
    vr_data["speed_threshold_neg"] = np.repeat(np.nan, len(vr_data))
    vr_data["hd_threshold"] = np.repeat(np.nan, len(vr_data))
    vr_data["border_threshold"] = np.repeat(np.nan, len(vr_data))
    vr_data["rayleigh_threshold"] = np.repeat(np.nan, len(vr_data))
    vr_data["spatial_threshold"] = np.repeat(np.nan, len(vr_data))
    vr_data["grid_threshold"] = np.repeat(np.nan, len(vr_data))
    vr_data["mean_firing_rate_of"] = np.repeat(np.nan, len(vr_data))
    vr_data["spike_width"] = np.repeat(np.nan, len(vr_data))
    return vr_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #LOAD DATA
    of_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/of_vr_combined_dataframes/combined_Cohort7_for_R.pkl")
    vr_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort7_unsmoothened.pkl")
    vr_data = add_date_to_frame(vr_data)
    of_data = add_date_to_frame(of_data)
    vr_data = add_mouse_to_frame(vr_data)
    of_data = add_mouse_to_frame(of_data)
    data = load_openfield_data_into_frame(of_data, vr_data)
    # SAVE DATAFRAMES for R
    data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort7_with_OF_unsmoothened.pkl")


    # Some dataframes don't have corresponding OF statistics so we need to pass the OF statistics as nans
    vr_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort2_unsmoothened.pkl")
    data = add_OF_stats_as_nans(vr_data)
    data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort2_with_OF_unsmoothened.pkl")

    vr_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort3_unsmoothened.pkl")
    data = add_OF_stats_as_nans(vr_data)
    data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort3_with_OF_unsmoothened.pkl")

    vr_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort4_unsmoothened.pkl")
    data = add_OF_stats_as_nans(vr_data)
    data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort4_with_OF_unsmoothened.pkl")

    vr_data = pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort5_unsmoothened.pkl")
    data = add_OF_stats_as_nans(vr_data)
    data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/Processed_cohort5_with_OF_unsmoothened.pkl")

if __name__ == '__main__':
    main()

