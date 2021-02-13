import Python_PostSorting.MakePlots_Behaviour
import Python_PostSorting.LoadPositionDataFrame
import Python_PostSorting.parameters
import Python_PostSorting.RewardFiring
import Python_PostSorting.Speed_Analysis
import Python_PostSorting.Curation
import Python_PostSorting.StopAnalysis
import Python_PostSorting.FitAnalysis
import Python_PostSorting.AnalyseSpikes
import Python_PostSorting.AnalyseRewardedSpikes
import Python_PostSorting.Add_BrainRegion_Classifier
import Python_PostSorting.SplitDataBySpeed
import Python_PostSorting.BehaviourAnalysis
import Python_PostSorting.RewardAnalysis_behaviour
import Python_PostSorting.FirstStopAnalysis_behaviour
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
    df["Day"] = ""
    df["Day_numeric"] = ""
    for cluster in range(len(df)):
        session_id = df.session_id.values[cluster]
        numericday, day, mouse = extract_mouse_and_day(session_id)
        df.at[cluster,"Mouse"] = mouse
        df.at[cluster,"Day"] = day
        df.at[cluster,"Day_numeric"] = numericday
    return df


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    server_path= '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/'
    initialize_parameters(server_path)
    print('Processing ' + str(server_path))

    #LOAD DATA
    spike_data = Python_PostSorting.LoadPositionDataFrame.process_allmice_dir(server_path, prm) # overall data
    spike_data.reset_index(drop=True, inplace=True)

    spike_data = add_mouse_to_frame(spike_data)
    spike_data = Python_PostSorting.Add_BrainRegion_Classifier.load_brain_region_data_into_frame(spike_data)
    spike_data = Python_PostSorting.FitAnalysis.load_Teris_ramp_score_data_into_frame(spike_data)

    #spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_reward_rate(spike_data)
    #spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_rewardrate_learning_curve(spike_data)

    #spike_data = Python_PostSorting.BehaviourAnalysis.graduation_day2(spike_data)
    #spike_data = Python_PostSorting.BehaviourAnalysis.calculate_progression(spike_data)

    spike_data = Python_PostSorting.RewardFiring.split_time_data_by_reward(spike_data, prm)
    spike_data_firststop = Python_PostSorting.FirstStopAnalysis_behaviour.extract_first_stop_rewarded(spike_data, prm)

    # SAVE DATAFRAMES
    #spike_data = drop_columns_from_frame(spike_data)
    spike_data_firststop.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/FirstStop_c4.pkl')



if __name__ == '__main__':
    main()

