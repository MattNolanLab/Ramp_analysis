import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Add_BrainRegion_Classifier
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Add_RampScore
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_Acceleration
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_RewardSpeed_ByOutcome
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_DataByTrialOutcome
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_SpeedByTrialOutcome
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_FiringRateMap
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_RatesinTime_Fig2D
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_FRAlignedToReward
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

prm = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters.Parameters()

def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(4.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)

# this function makes analysis based on mice/cohort/days easier later down the line in R
# I add cohort in manually which is annoying, but its for further down the line in R when I do stats based on animals. Problem is a lot of animals have a similar id i.e M2. So adding cohort distinguishes them
def add_mouse_to_frame(df, cohort):
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
        df.at[cluster,"cohort"] = cohort # Change this to current cohort analysed!! This should be on the name of the
    return df

# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse

def run_example_plots(spike_data, save_path):
    # extract firing rate maps
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=True, smoothen=True)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=False, smoothen=True)

    # plot spike rasters
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track_beaconed(save_path, spike_data, rewarded=True)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track(save_path, spike_data, rewarded=True)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track_beaconed(save_path, spike_data, rewarded=False)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_FRAlignedToReward.plot_rewarded_spikes_on_track(save_path, spike_data, rewarded=False)

    # plot average firing rate maps
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_FiringRateMap.plot_firing_rate_maps_for_trials(save_path, spike_data, rewarded=True, smoothen=True)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_FiringRateMap.plot_firing_rate_maps_for_trials(save_path, spike_data, rewarded=False, smoothen=True)

    # plot instantaneous rates as a function of kinematic variables
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_RatesinTime_Fig2D.plot_color_coded_instant_rates_according_to_segment(save_path, spike_data, smoothen=True)

def plot_behaviour(spike_data):
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.curate_stops(spike_data)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm, plot_p=False) # from postprocessing spatial data
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.plot_stops_on_track_per_cluster(spike_data, prm, plot_p=True) # from postprocessing spatial data
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_nonbeaconed_stops(spike_data) # from postprocessing spatial data
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_stops(spike_data) # from postprocessing spatial data
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.plot_stop_histogram_per_cluster(spike_data, prm, plot_p=False) # from postprocessing spatial data
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.plot_stop_histogram_per_cluster(spike_data, prm, plot_p=True) # from postprocessing spatial data
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.plot_speed_histogram(spike_data, prm) # from postprocessing spatial data


def run_main_figure_analysis(spike_data):
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=False, smoothen=False)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=False, smoothen=True)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=True, smoothen=False)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.AvgRewardedSpikes.extract_smoothed_average_firing_rate_data(spike_data, rewarded=True, smoothen=True)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_Acceleration.generate_acceleration(spike_data, rewarded=True)

    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.curate_stops(spike_data)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_nonbeaconed_stops(spike_data) # from postprocessing spatial data
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_stops(spike_data) # from postprocessing spatial data
    return spike_data


def run_supple_figure_analysis(spike_data):
    # Split data by TRIAL OUTCOME (HIT/TRY/RUN) : Analysis for Figure 3

    # split by hit, try, run columns
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_DataByTrialOutcome.split_time_data_by_trial_outcome(spike_data)

    # calculate firing rate maps for different trial types and hit run try and try slow/fast
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_DataByTrialOutcome.extract_firing_rate_map_by_hit_try_run(spike_data)

    # calculates speed maps and time binned statistics
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_SpeedByTrialOutcome.split_and_save_speed_data(spike_data)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_SpeedByTrialOutcome.extract_time_binned_speed_by_outcome(spike_data)
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_RewardSpeed_ByOutcome.calc_histo_speed(spike_data)

    return spike_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots"
    ramp_score_path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/all_rampscore.csv"
    brain_region_path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Analysis/data_in/tetrode_locations.csv"
    criteria_path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Analysis/data_in/Criteria_days.csv"

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    initialize_parameters(save_path)
    print('Processing ' + str(save_path))

    n_cells = 0
    for spike_data_path, cohort, output_spike_path in zip(["/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort2_concatenated_spike_data_unsmoothened.pkl",
                                                           "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort3_concatenated_spike_data_unsmoothened.pkl",
                                                           "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort4_concatenated_spike_data_unsmoothened.pkl",
                                                           "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort5_concatenated_spike_data_unsmoothened.pkl",
                                                           "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/vr_dataframes/cohort7_concatenated_spike_data_unsmoothened.pkl"],
                                                          [2, 3, 4, 5, 7],
                                                            ["/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_unsmoothened.pkl",
                                                             "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_unsmoothened.pkl",
                                                             "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_unsmoothened.pkl",
                                                             "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_unsmoothened.pkl",
                                                             "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_unsmoothened.pkl"]):

        #LOAD DATA
        spike_data = pd.read_pickle(spike_data_path)
        spike_data.reset_index(drop=True, inplace=True)

        # CURATION (for spike data frame only)
        spike_data = add_mouse_to_frame(spike_data, cohort=cohort)
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation.add_peaks_to_troughs(spike_data)
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation.remove_false_positives(spike_data) # removes cells with low trial numbers
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation.curate_data(spike_data) # removes cells with low numbers of rewards
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation.make_neuron_number(spike_data) # this is for matching with the ramp score dataframe
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Curation.load_crtieria_data_into_frame(spike_data, criteria_path) # this is for curating data based on graduation day

        # ADD brain region and ramp score for each neuron to dataframe
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data, brain_region_path)
        spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Add_RampScore.load_ramp_score_data_into_frame(spike_data, ramp_score_path)

        # remove cells in V1
        spike_data = spike_data[spike_data["brain_region"] != "V1"]

        # remove lick artefacts
        spike_data = spike_data[spike_data["snippet_peak_to_trough"] < 500] # uV

        # remove non graduated days
        spike_data = spike_data[spike_data["graduation"] == 1]
        spike_data.reset_index(drop=True, inplace=True)

        # RUN EXAMPLE PLOTS - use if wanting to plot example data - otherwise COMMENT OUT
        run_example_plots(spike_data, save_path)
        plot_behaviour(spike_data) # plot stops, average stops etc

        # RUN FIGURE ANALYSIS
        spike_data = run_main_figure_analysis(spike_data)
        spike_data = run_supple_figure_analysis(spike_data)

        # SAVE DATAFRAMES for R
        spike_data.to_pickle(output_spike_path) # path to where you want the pkl to be saved

        print("I have processed cohort ", cohort, ", it is saved here: ", output_spike_path, ". There were ", str(len(spike_data)), " cells")
        n_cells += len(spike_data)

    print("I have analysed everything you asked me to Sir")
    print("There is a total of ", n_cells, " in the datasets")

if __name__ == '__main__':
    main()

