import Python_PostSorting.MakePlots
import Python_PostSorting.MakePlots_FiringProperties
import Python_PostSorting.LoadDataFrames
import Python_PostSorting.parameters
import Python_PostSorting.Split_DataByReward
import Python_PostSorting.Test_SpeedData
import Python_PostSorting.Curation
import Python_PostSorting.Add_Teris_RampScore
import Python_PostSorting.Calculate_Acceleration
import Python_PostSorting.AnalyseRewardedSpikes
import Python_PostSorting.Add_BrainRegion_Classifier
import Python_PostSorting.Calculate_RewardSpeed_ByOutcome
import Python_PostSorting.BehaviourAnalysis
import Python_PostSorting.RewardAnalysis_behaviour
import Python_PostSorting.FirstStopAnalysis_behaviour
import Python_PostSorting.Split_DataByTrialOutcome_slowandfast
import Python_PostSorting.Split_SpeedByTrialOutcome
import Python_PostSorting.Calculate_FRAlignedToReward
import Python_PostSorting.Calculate_Stops_from_Speed
import Python_PostSorting.Calculate_SpikeHalfWidth
import Python_PostSorting.Plot_TrialHeatmaps_SuppleFig6
import Python_PostSorting.Shuffle_Analysis
import Python_PostSorting.AvgRewardedSpikes
import Python_PostSorting.Plot_RatesinTime_Fig2D
import Python_PostSorting.Plot_RawData_Fig2A
import numpy as np
import pandas as pd
import os

prm = Python_PostSorting.parameters.Parameters()


"""

## Control script to run python post sorting 



"""



def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(4.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)


# this function makes analysis based on mice/cohort/days easier later down the line in R
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
        df.at[cluster,"cohort"] = 2 # Change this to current cohort analysed!!
    return df


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse


def run_example_plots(spike_data, server_path):
    #Python_PostSorting.MakePlots_FiringProperties.plot_autocorrelograms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_waveforms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_spike_histogram(spike_data, prm)

    #Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm) # from postprocessing spatial data

    Python_PostSorting.MakePlots.plot_beaconed_spikes_on_track(server_path,spike_data)  ## for example cells in Figure1
    Python_PostSorting.MakePlots.plot_beaconed_and_probe_spikes_on_track(server_path,spike_data)  ## for example cells in Figure3
    #spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_per_trialtype(server_path, spike_data) ## for all example cells
    #spike_data = Python_PostSorting.MakePlots.plot_tiny_raw(server_path, spike_data) ## for Figure3A

    Python_PostSorting.MakePlots.plot_rewarded_spikes_on_track(server_path,spike_data)  ## for all example cells
    spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials(server_path, spike_data) ## for all example cells
    #spike_data = Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials_outbound(server_path, spike_data) ## for all example cells

    #Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment(server_path, spike_data)
    return spike_data


def plot_behaviour(spike_data):
    Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm) # from postprocessing spatial data
    #spike_data = Python_PostSorting.MakePlots_Behaviour.calculate_average_nonbeaconed_stops(spike_data) # from postprocessing spatial data
    #spike_data = Python_PostSorting.MakePlots_Behaviour.calculate_average_stops(spike_data) # from postprocessing spatial data
    #Python_PostSorting.MakePlots_Behaviour.plot_stop_histogram_per_cluster(spike_data, prm) # from postprocessing spatial data
    #spike_data = Python_PostSorting.RewardFiring.split_data_by_reward(spike_data, prm)
    #spike_data = Python_PostSorting.MakePlots_Behaviour.calculate_average_nonbeaconed_speed(spike_data) # from postprocessing spatial data
    #spike_data = Python_PostSorting.MakePlots_Behaviour.calculate_average_speed(spike_data) # from postprocessing spatial data
    #Python_PostSorting.MakePlots_Behaviour.plot_speed_histogram(spike_data, prm) # from postprocessing spatial data
    return spike_data



def test_speed_data(spike_data,server_path ):
    spike_data = Python_PostSorting.Test_SpeedData.calculate_speed_from_position(spike_data, server_path)
    return spike_data


def run_main_figure_analysis(spike_data,server_path):
    spike_data = Python_PostSorting.Split_DataByReward.split_data_by_reward(spike_data)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate_rewarded(spike_data)
    #spike_data = Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data)
    #spike_data = Python_PostSorting.AnalyseRewardedSpikes.plot_rewarded_rates(spike_data, prm)
    spike_data = Python_PostSorting.Calculate_Acceleration.generate_acceleration_rewarded_trials(spike_data, server_path)
    return spike_data


def run_stuff(spike_data):
    spike_data = Python_PostSorting.Split_DataByReward.convert_spikes_in_time_to_ms(spike_data)
    spike_data = Python_PostSorting.Split_DataByReward.rename_columns(spike_data)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate(spike_data)
    spike_data = Python_PostSorting.AvgRewardedSpikes.rewrite_smoothed_average_firing_rate_data(spike_data)
    #spike_data = Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data)
    return spike_data


def run_supple_figure_analysis(spike_data):
    # Split data by TRIAL OUTCOME (HIT/TRY/RUN) : Analysis for Figure 7
    spike_data = Python_PostSorting.Split_DataByTrialOutcome_slowandfast.split_time_data_by_trial_outcome(spike_data, prm)
    spike_data = Python_PostSorting.Split_DataByTrialOutcome_slowandfast.extract_time_binned_firing_rate_runthru_allspeeds(spike_data)
    spike_data = Python_PostSorting.Split_DataByTrialOutcome_slowandfast.extract_time_binned_firing_rate_try_allspeeds(spike_data)
    spike_data = Python_PostSorting.Split_DataByTrialOutcome_slowandfast.extract_time_binned_firing_rate_try_slow_allspeeds(spike_data)
    spike_data = Python_PostSorting.Split_DataByTrialOutcome_slowandfast.extract_time_binned_firing_rate_rewarded_allspeeds(spike_data)

    #spike_data = Python_PostSorting.Plot_TrialHeatmaps_SuppleFig6.plot_heatmap_by_trial(spike_data, prm)
    #spike_data = Python_PostSorting.Plot_TrialHeatmaps_SuppleFig6.plot_average(spike_data, prm)

    #spike_data = Python_PostSorting.Split_SpeedByTrialOutcome.split_and_save_speed_data(spike_data)
    #spike_data = Python_PostSorting.Split_SpeedByTrialOutcome.extract_time_binned_speed_by_outcome(spike_data)

    #spike_data = Python_PostSorting.Calculate_RewardSpeed_ByOutcome.calc_histo_speed(spike_data)

    return spike_data


def run_reward_aligned_analysis(server_path,spike_data):
    spike_data = Python_PostSorting.Calculate_FRAlignedToReward.run_reward_aligned_analysis(server_path,spike_data, prm)
    return spike_data


def plot_raw_spike_data(spike_data):
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

    save_path= '/Users/sarahtennant/Work/Analysis/Ramp_Plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    initialize_parameters(save_path)
    print('Processing ' + str(save_path))

    #LOAD DATA
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir(save_path, prm) # overall data
    spike_data.reset_index(drop=True, inplace=True)

    # CURATION (for spike data frame only)
    spike_data = add_mouse_to_frame(spike_data)
    #spike_data = Python_PostSorting.Curation.remove_lick_artefact(spike_data)
    spike_data = Python_PostSorting.Curation.remove_false_positives(spike_data) # removes cells with low trial numbers
    spike_data = Python_PostSorting.Curation.curate_data(spike_data) # removes cells with low numbers of rewards
    spike_data = Python_PostSorting.Curation.make_neuron_number(spike_data) # this is for matching with Teris's ramp score dataframe
    #spike_data = Python_PostSorting.Curation.load_crtieria_data_into_frame(spike_data) # this is for curating data based on graduation day

    # ADD brain region and ramp score for each neuron to dataframe - COMMENT OUT IF NOT NEEDED
    #spike_data = Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data)
    #spike_data = Python_PostSorting.Add_Teris_RampScore.load_Teris_ramp_score_data_into_frame(spike_data)

    ## use if wanting to test on specific mouse/day - otherwise COMMENT OUT
    #spike_data = spike_data[spike_data['Mouse'] == "M1"]
    #spike_data = spike_data[spike_data['Day_numeric'] == "5"]
    #spike_data = spike_data.head(n=2)
    #spike_data.reset_index(drop=True, inplace=True) # Make sure you reset the index if you subset the data because otherwise some analysis based on rowcount wont work!!


    # RUN EXAMPLE PLOTS - use if wanting to plot example data - otherwise COMMENT OUT
    #spike_data = run_example_plots(spike_data,server_path)
    #spike_data = Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track_with_tt(server_path,spike_data)

    #spike_data = plot_behaviour(spike_data) # plot stops, average stops etc
    spike_data = Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track_with_tt(save_path,spike_data)
    spike_data = Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data)
    spike_data = Python_PostSorting.MakePlots.plot_firing_rate_maps_for_rewarded_trials(save_path, spike_data) ## for all example cells
    Python_PostSorting.Plot_RatesinTime_Fig2D.plot_color_coded_instant_rates_according_to_segment(save_path, spike_data)
    spike_data = Python_PostSorting.Plot_RawData_Fig2A.plot_tiny_raw(save_path, spike_data) ## for Figure2A

    # TEST speed data - use if wanting to check speed is correct measurement - otherwise COMMENT OUT
    #spike_data = test_speed_data(spike_data,server_path)

    # RUN FIGURE ANALYSIS
    #spike_data = run_main_figure_analysis(spike_data, server_path)
    #spike_data = Python_PostSorting.Shuffle_Analysis.generate_shuffled_data(spike_data)
    #spike_data = run_supple_figure_analysis(spike_data)
    #spike_data = run_stuff(spike_data)
    # SAVE DATAFRAMES for R
    #spike_data = drop_columns_from_frame(spike_data) # UNCOMMENT if you want to drop unused columns from the dataframe so the saved file is smaller
    #spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/processed_spike_data_cohort2.pkl') # path to where you want the pkl to be saved



if __name__ == '__main__':
    main()

