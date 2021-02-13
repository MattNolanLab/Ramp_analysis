import Python_PostSorting.MakePlots
import Python_PostSorting.MakePlots_Behaviour
import Python_PostSorting.MakePlots_Shuffled
import Python_PostSorting.MakePlots_FiringProperties
import Python_PostSorting.LoadDataFrames
import Python_PostSorting.parameters
import Python_PostSorting.RewardFiring
import Python_PostSorting.Speed_Analysis
import Python_PostSorting.Curation
import Python_PostSorting.StopAnalysis
import Python_PostSorting.SpikeWidth
import Python_PostSorting.FitAnalysis
import Python_PostSorting.CalculateAcceleration
import Python_PostSorting.Spike_Time_Analysis
import Python_PostSorting.AnalyseSpikes
import Python_PostSorting.AnalyseRewardedSpikes
import Python_PostSorting.Add_BrainRegion_Classifier
import Python_PostSorting.SplitDataBySpeed
import Python_PostSorting.BehaviourAnalysis
import Python_PostSorting.RewardAnalysis_behaviour
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


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    server_path= '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/'
    initialize_parameters(server_path)
    print('Processing ' + str(server_path))



    #LOAD DATA
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir(server_path, prm) # overall data
    spike_data.reset_index(drop=True, inplace=True)

    # CURATION (for spike data frame only)
    #spike_data = Python_PostSorting.Curation.remove_false_positives(spike_data)
    #spike_data = Python_PostSorting.Curation.curate_data(spike_data)

    # Add brain region and ramp score data for each neuron to dataframe
    #spike_data = Python_PostSorting.Curation.make_neuron_number(spike_data)

    #spike_data = Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data)
    #spike_data = Python_PostSorting.FitAnalysis.load_Teris_ramp_score_data_into_frame(spike_data)

    #spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_reward_rate(spike_data)
    #spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_rewardrate_learning_curve(spike_data)

    #spike_data = Python_PostSorting.BehaviourAnalysis.graduation_day2(spike_data)
    #spike_data = Python_PostSorting.BehaviourAnalysis.calculate_progression(spike_data)

    #Python_PostSorting.MakePlots_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm)
    #spike_data = Python_PostSorting.RewardFiring.generate_reward_indicator(spike_data) # for saving data into dataframe for R
    #Python_PostSorting.MakePlots.plot_rewarded_spikes_on_track2(server_path,spike_data)
    #Python_PostSorting.MakePlots.plot_failed_spikes_on_track2(server_path,spike_data)
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_rewarded_trials(server_path, spike_data)
    #Python_PostSorting.MakePlots.plot_smoothed_firing_rate_maps_for_failed_trials(server_path, spike_data)

    spike_data = Python_PostSorting.RewardFiring.split_time_data_by_reward(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate_rewarded(spike_data, prm)
    #spike_data = Python_PostSorting.AnalyseRewardedSpikes.extract_time_binned_firing_rate_failed(spike_data, prm)
    spike_data = Python_PostSorting.AnalyseRewardedSpikes.plot_rewarded_rates(spike_data, prm)

    spike_data = Python_PostSorting.CalculateAcceleration.generate_acceleration(spike_data, server_path)
    Python_PostSorting.MakePlots.plot_color_coded_instant_rates_according_to_segment(server_path, spike_data)

    # MAKE PLOTS
    #make_behaviour_plots(server_path, spike_data, prm)
    #spike_data = plot_reward_based_analysis(server_path, spike_data) # split trials based on reward then plot spikes on trials and firing rate
    #spike_data = make_plots(server_path,spike_data)



if __name__ == '__main__':
    main()

