import Python_PostSorting.MakePlots
import Python_PostSorting.MakePlots_Behaviour
import Python_PostSorting.MakePlots_Shuffled
import Python_PostSorting.MakePlots_FiringProperties
import Python_PostSorting.LoadDataFrames
import Python_PostSorting.parameters
import Python_PostSorting.RewardFiring
import Python_PostSorting.FiringProperties
import Python_PostSorting.Speed_Analysis
import Python_PostSorting.RampAnalysis
import Python_PostSorting.Curation
import Python_PostSorting.FirstStopAnalysis
import Python_PostSorting.StopAnalysis
import Python_PostSorting.RateAnalysis
import Python_PostSorting.SpikeWidth
import Python_PostSorting.RewardAnalysis
import Python_PostSorting.FitAnalysis
import Python_PostSorting.CalculateAcceleration
import Python_PostSorting.Spike_Time_Analysis
import Python_PostSorting.AnalyseSpikes
import Python_PostSorting.AnalyseRewardedSpikes
import Python_PostSorting.Add_BrainRegion_Classifier
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


def make_plots(recording_folder,spike_data):
    #Python_PostSorting.MakePlots.plot_spikes_on_track(recording_folder,spike_data, prm, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_spikes_on_track_per_trialtype(recording_folder,spike_data, prm, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_spikes_on_track_example(recording_folder,spike_data, prm, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_spikes_on_track_per_trialtype_example(recording_folder,spike_data, prm, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_firing_rate_maps(recording_folder, spike_data, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps(recording_folder, spike_data, prefix='_movement')
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_per_trialtype(recording_folder, spike_data, prefix='_movement')

    #Python_PostSorting.MakePlots.plot_rewarded_spikes_on_track(recording_folder,spike_data)
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials(recording_folder, spike_data)

    #Python_PostSorting.Spike_Analysis.extract_time_binned_firing_rate(spike_data, prm)
    #Python_PostSorting.Spike_Analysis.extract_time_binned_firing_rate_per_trialtype(spike_data, prm)
    #Python_PostSorting.Spike_Analysis.extract_time_binned_firing_rate_per_trialtype_shuffled(spike_data, prm)
    #Python_PostSorting.Spike_Analysis.extract_time_binned_firing_rate_per_trialtype_outbound(spike_data, prm)
    #Python_PostSorting.Spike_Analysis.extract_time_binned_firing_rate_per_trialtype_probe(spike_data, prm)

    Python_PostSorting.MakePlots.plot_color_coded_instant_rates(recording_folder, spike_data)
    Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment(recording_folder, spike_data)
    Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment_nonbeaconed(recording_folder, spike_data)
    Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment_probe(recording_folder, spike_data)
    #Python_PostSorting.MakePlots.plot_color_trial_coded_instant_rates_according_to_segment(recording_folder, spike_data)
    #Python_PostSorting.MakePlots.plot_color_coded_instant_rates_by_trial(recording_folder, spike_data)
    return spike_data


def make_behaviour_plots(recording_folder,spike_data, prm):
    #Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track(recording_folder, processed_position_data, prm)
    Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(recording_folder, spike_data, prm)
    Python_PostSorting.MakePlots_Behaviour.plot_speed(recording_folder,spike_data)
    return


def make_firing_plots(spike_data):
    #Python_PostSorting.MakePlots_FiringProperties.plot_waveforms(spike_data, prm)
    #Python_PostSorting.MakePlots_FiringProperties.plot_clean_waveforms(spike_data, prm)
    Python_PostSorting.MakePlots_FiringProperties.plot_autocorrelograms(spike_data, prm)
    Python_PostSorting.MakePlots_FiringProperties.plot_spike_histogram(spike_data, prm)
    return


def load_local_frames(server_path, prm):
    #spike_data = Python_PostSorting.LoadDataFrames.process_a_dir(server_path, prm) # test data
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir(server_path, prm) # overall data
    #spike_data = PostSorting.LoadDataFrames.process_multi_spatial_dir(server_path, prm) # loads spatial data into the spike dataframe
    return spike_data


def drop_columns_from_frame(spike_data):
    spike_data.drop(['random_snippets'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['x_position_cm'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['trial_numbers'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['trial_types'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['firing_times'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['rewarded_trials'], axis='columns', inplace=True, errors='ignore')
    #spike_data.drop(['rewarded_locations'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['stop_location_cm'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['stop_trial_number'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_num_on_trials'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_rate_on_trials'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_rate_on_trials_smoothed'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['binned_apsolute_elapsed_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['binned_speed_ms_per_trial'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['isolation'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['noise_overlap'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['peak_snr'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['spike_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['position_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    spike_data.drop(['speed_rate_in_time'], axis='columns', inplace=True, errors='ignore')
    return spike_data


def run_behavioural_analysis(spike_data, server_path):
    #speed amalysis
    Python_PostSorting.Speed_Analysis.extract_time_binned_speed(spike_data, prm) # from data binned in time
    #spike_data = Python_PostSorting.Speed_Analysis.calculate_average_speed(spike_data) # from data binned in space
    # First stop and reward rates (learning curves)
    #spike_data = Python_PostSorting.FirstStopAnalysis.calculate_first_stop(spike_data)
    #spike_data = Python_PostSorting.FirstStopAnalysis.calculate_firststop_learning_curve(spike_data)
    #spike_data = Python_PostSorting.FirstStopAnalysis.multimouse_firststop_plot()
    #spike_data = Python_PostSorting.RewardAnalysis.calculate_reward_rate(spike_data)
    #spike_data = Python_PostSorting.RewardAnalysis.calculate_rewardrate_learning_curve(spike_data)
    #spike_data = Python_PostSorting.RewardAnalysis.multimouse_rewardrate_plot()
    # Average stop and average shuffled stop (histograms)
    #spike_data = Python_PostSorting.ShuffleStops.generate_shuffled_data_for_stops(spike_data)
    #spike_data = Python_PostSorting.StopAnalysis.calculate_avg_stop(spike_data)
    #spike_data = Python_PostSorting.StopAnalysis.calculate_avg_shuff_stop(spike_data)
    return spike_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    server_path= '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/'
    initialize_parameters(server_path)
    print('Processing ' + str(server_path))

    #LOAD DATA
    spike_data = load_local_frames(server_path, prm)
    spike_data.reset_index(drop=True, inplace=True)

    #run_behavioural_analysis(spike_data, server_path)
    #make_firing_plots(spike_data)

    # CURATION (for spike frame only)
    spike_data = Python_PostSorting.Curation.remove_false_positives(spike_data)
    spike_data = Python_PostSorting.Curation.curate_data(spike_data)
    spike_data = Python_PostSorting.Curation.make_neuron_number(spike_data)
    spike_data = Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data)
    spike_data = Python_PostSorting.FitAnalysis.load_Teris_ramp_score_data_into_frame(spike_data)

    #Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm)
    #spike_data = Python_PostSorting.RewardFiring.generate_reward_indicator(spike_data) # for saving data into dataframe for R
    #Python_PostSorting.MakePlots.plot_rewarded_spikes_on_track2(server_path,spike_data)
    #Python_PostSorting.MakePlots.plot_failed_spikes_on_track2(server_path,spike_data)
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials(server_path, spike_data)
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_failed_trials(server_path, spike_data)

    #spike_data = Python_PostSorting.AnalyseSpikes.extract_time_binned_firing_rate_overtrial_per_trialtype(spike_data, prm)

    spike_data = Python_PostSorting.RewardFiring.split_time_data_by_reward(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate_rewarded(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.plot_rewarded_firing_rate(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.plot_rewarded_nb_firing_rate(spike_data, prm)

    spike_data = Python_PostSorting.CalculateAcceleration.generate_acceleration(spike_data, server_path)

    # MAKE PLOTS
    #make_behaviour_plots(server_path, spike_data, prm)
    #make_shuffled_plots(server_path,spike_data)
    #spike_data = plot_reward_based_analysis(server_path, spike_data) # split trials based on reward then plot spikes on trials and firing rate

    # ANALYSIS FOR R
    #spike_data = run_analysis_for_r(spike_data, server_path)
    #spike_data = make_plots(server_path,spike_data)

    # SAVE DATAFRAMES
    spike_data = drop_columns_from_frame(spike_data)
    spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Alldays_cohort_4.pkl')



if __name__ == '__main__':
    main()

