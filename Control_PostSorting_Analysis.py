import Python_PostSorting.MakePlots
import Python_PostSorting.MakePlots_Behaviour
import Python_PostSorting.MakePlots_FiringProperties
import Python_PostSorting.LoadDataFrames
import Python_PostSorting.parameters
import Python_PostSorting.RewardFiring
import Python_PostSorting.Speed_Analysis
import Python_PostSorting.Curation
import Python_PostSorting.StopAnalysis
import Python_PostSorting.FitAnalysis
import Python_PostSorting.CalculateAcceleration
import Python_PostSorting.AnalyseRewardedSpikes
import Python_PostSorting.Add_BrainRegion_Classifier
import Python_PostSorting.SplitDataBySpeed
import Python_PostSorting.BehaviourAnalysis
import Python_PostSorting.RewardAnalysis_behaviour
import Python_PostSorting.FirstStopAnalysis_behaviour
import Python_PostSorting.Split_By_Trial_Outcome
import Python_PostSorting.PlotFiringRate_Update
import Python_PostSorting.Split_SpeedBy_Trial_Outcome
import Python_PostSorting.FR_relative_to_Behaviour
import Python_PostSorting.FR_relative_to_FS
import Python_PostSorting.Calculate_Stops_from_Speed
import Python_PostSorting.Detect_Peaks_Sarah
import Python_PostSorting.ISI_Analysis

import numpy as np
import pandas as pd

prm = Python_PostSorting.parameters.Parameters()


def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)


def add_mouse_to_frame(df):
    print("Adding ID for Mouse and Day to dataframe...")
    df["Mouse"] = ""
    df["Cohort"] = ""
    df["Day"] = ""
    df["Day_numeric"] = ""
    for cluster in range(len(df)):
        session_id = df.session_id.values[cluster]
        numericday, day, mouse = extract_mouse_and_day(session_id)
        df.at[cluster,"Mouse"] = mouse
        df.at[cluster,"Day"] = day
        df.at[cluster,"Day_numeric"] = numericday
        df.at[cluster,"cohort"] = 5
    return df


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse


def run_example_plots_for_paper(spike_data, server_path):
    #Python_PostSorting.MakePlots_FiringProperties.plot_autocorrelograms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_waveforms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_clean_waveforms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_spike_histogram(spike_data, prm)

    #Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm) # from postprocessing spatial data

    Python_PostSorting.MakePlots.plot_beaconed_spikes_on_track(server_path,spike_data)  ## for example cells in Figure1
    Python_PostSorting.MakePlots.plot_beaconed_and_probe_spikes_on_track(server_path,spike_data)  ## for example cells in Figure3
    #spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_per_trialtype(server_path, spike_data) ## for all example cells
    #spike_data = Python_PostSorting.MakePlots.plot_tiny_raw(server_path, spike_data) ## for Figure3A

    Python_PostSorting.MakePlots.plot_rewarded_spikes_on_track(server_path,spike_data)  ## for all example cells
    #spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials(server_path, spike_data) ## for all example cells
    #spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials_outbound(server_path, spike_data) ## for all example cells
    return spike_data


def test_stop_data(spike_data,server_path ):
    # TEST stop data
    spike_data = Python_PostSorting.Calculate_Stops_from_Speed.calculate_stops_from_200ms_speed(spike_data)
    #spike_data = Python_PostSorting.Calculate_Stops_from_Speed.calculate_rewards_from_stops(spike_data)
    Python_PostSorting.Calculate_Stops_from_Speed.plot_stops_on_track_per_cluster(spike_data, prm)
    Python_PostSorting.Calculate_Stops_from_Speed.calculate_average_stops(spike_data)
    Python_PostSorting.Calculate_Stops_from_Speed.plot_stop_histogram(server_path, spike_data, prm)
    return spike_data


def test_speed_data(spike_data,server_path ):
    spike_data = Python_PostSorting.Speed_Analysis.calculate_speed_from_position(spike_data, server_path)
    return spike_data


def test_FR_data(spike_data):
    spike_data = Python_PostSorting.Calculate_Stops_from_Speed.calculate_average_spikes(spike_data)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.extract_time_binned_firing_rate(spike_data, prm)
    spike_data = Python_PostSorting.Calculate_Stops_from_Speed.plot_firing_rate_probe(spike_data, prm)
    return spike_data


def run_figure_analysis(spike_data,server_path):
    spike_data = Python_PostSorting.ISI_Analysis.generate_spike_isi(server_path, spike_data)
    spike_data = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data_for_rewarded_trials2(spike_data)
    spike_data = Python_PostSorting.RewardFiring.split_time_data_by_reward(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate_rewarded_original(spike_data,prm)
    spike_data = Python_PostSorting.CalculateAcceleration.generate_acceleration_rewarded_trials(spike_data, server_path)
    return spike_data



def drop_columns_from_frame(spike_data):
    spike_data.drop(['random_snippets'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['x_position_cm'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['trial_numbers'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['trial_types'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['firing_times'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['rewarded_trials'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['stop_location_cm'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['stop_trial_number'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_num_on_trials'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_rate_on_trials'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['binned_apsolute_elapsed_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['binned_speed_ms_per_trial'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['isolation'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['noise_overlap'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['peak_snr'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['position_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['speed_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    return spike_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    server_path= '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis'
    initialize_parameters(server_path)
    print('Processing ' + str(server_path))

    #LOAD DATA
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir(server_path, prm) # overall data
    #spike_data = spike_data.head(n=5)
    spike_data.reset_index(drop=True, inplace=True)

    # CURATION (for spike data frame only)
    spike_data = Python_PostSorting.Curation.remove_false_positives(spike_data)
    spike_data = Python_PostSorting.Curation.curate_data(spike_data)
    spike_data = Python_PostSorting.Curation.make_neuron_number(spike_data)
    spike_data = add_mouse_to_frame(spike_data)


    # Add brain region and ramp score data for each neuron to dataframe
    #spike_data = Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data)
    #spike_data = Python_PostSorting.FitAnalysis.load_Teris_ramp_score_data_into_frame(spike_data)

    ## use if wanting to test on specific mouse/day - otherwise COMMENT OUT
    #spike_data = spike_data[spike_data['Day'] == "D9"]
    #spike_data = spike_data.head(n=20)
    #spike_data.reset_index(drop=True, inplace=True)

    #spike_data = Python_PostSorting.Calculate_Stops_from_Speed.calculate_average_spikes(spike_data)
    #spike_data = Python_PostSorting.Calculate_Stops_from_Speed.plot_firing_rate_probe(spike_data, prm)
    # RUN EXAMPLE PLOTS
    #spike_data = run_example_plots_for_paper(spike_data,server_path)

    # TEST stop data
    #spike_data = test_stop_data(spike_data,server_path)

    # TEST speed data
    #spike_data = test_speed_data(spike_data,server_path)

    # RUN FIGURE ANALYSIS
    spike_data = run_figure_analysis(spike_data, server_path)

    # time binned firing rate, all trials (not based on rewarded trials)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.extract_time_binned_firing_rate(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_firing_rate(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_firing_rate_probe(spike_data, prm)

    #spike_data = Python_PostSorting.PlotFiringRate_Update.extract_time_binned_firing_rate_rewarded(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_rewarded_firing_rate_probe_outbound(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_rewarded_firing_rate_nonbeaconed_outbound(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.calculate_trial_by_trial_peaks(spike_data)

    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_heatmap_by_trial(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_heatmap_by_trial_uncued(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_rewarded_firing_rate(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_rewarded_firing_rate_probe(spike_data, prm)

    # Align firing rates by reward or first stop
    #spike_data = Python_PostSorting.FR_relative_to_Behaviour.run_reward_aligned_analysis(server_path,spike_data, prm)
    #spike_data = Python_PostSorting.FR_relative_to_FS.run_firststop_aligned_analysis(server_path,spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.calculate_trial_by_trial_peaks(spike_data)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.calculate_trial_by_trial_minpeaks(spike_data)

    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_data(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_data_with_min(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_norm_data_alone_with_peak(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_reward_data_alone_with_peak(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_fs_data(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_fs_data_with_min(spike_data, prm)
    #spike_data = Python_PostSorting.PlotFiringRate_Update.plot_aligned_fs_data_alone_with_peak(spike_data, prm)
    # Run peak analysis
    #spike_data = Python_PostSorting.Detect_Peaks_Sarah.run_peak_analysis(spike_data)


    # Split data by TRIAL OUTCOME (HIT/TRY/RUN) : Analysis for Figure 7
    #spike_data = Python_PostSorting.Split_By_Trial_Outcome.split_time_data_by_trial_outcome(spike_data, prm)
    #spike_data = Python_PostSorting.Split_By_Trial_Outcome.extract_time_binned_firing_rate_runthru_allspeeds(spike_data)
    #spike_data = Python_PostSorting.Split_By_Trial_Outcome.extract_time_binned_firing_rate_try_allspeeds(spike_data)
    #spike_data = Python_PostSorting.Split_By_Trial_Outcome.extract_time_binned_firing_rate_rewarded_allspeeds(spike_data)
    #spike_data = Python_PostSorting.FR_relative_to_FS.run_firststop_aligned_analysis_for_trytrials(server_path,spike_data, prm)

    #spike_data = Python_PostSorting.Split_SpeedBy_Trial_Outcome.split_and_save_speed_data(spike_data)
    #spike_data = Python_PostSorting.Split_SpeedBy_Trial_Outcome.extract_time_binned_speed_by_outcome(spike_data)

    #spike_data = Python_PostSorting.SplitDataBySpeed.calc_histo_speed(spike_data)

    # calculate acceleration and plot instant rates : ## Analysis for Figure 3
    #Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment(server_path, spike_data)

    # SAVE DATAFRAMES for R
    spike_data = drop_columns_from_frame(spike_data)
    spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/All_cohort_harry.pkl')



if __name__ == '__main__':
    main()

