import Python_PostSorting.MakePlots_Behaviour
import Python_PostSorting.LoadDataFrames
import Python_PostSorting.parameters
import Python_PostSorting.Split_DataByReward
import Python_PostSorting.Test_SpeedData
import Python_PostSorting.Curation
import Python_PostSorting.Add_BrainRegion_Classifier
import Python_PostSorting.Split_SpeedByTrialOutcome
import Python_PostSorting.BehaviourAnalysis
import Python_PostSorting.RewardAnalysis_behaviour
import Python_PostSorting.FirstStopAnalysis_behaviour
import numpy as np
import pandas as pd
import csv


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
        df.at[cluster,"cohort"] = 7
    return df


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return day, day1, mouse


## Save first stop learning curves to .csv file
def write_to_csv(csvData):
    with open('/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/rewardrate-' + '7' + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    return



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    server_path= '/Users/sarahtennant/Work/Analysis/Ephys/OverallAnalysis/'
    initialize_parameters(server_path)
    print('Processing ' + str(server_path))

    #LOAD DATA
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir(server_path, prm) # overall data
    spike_data.reset_index(drop=True, inplace=True)

    spike_data = add_mouse_to_frame(spike_data)

    #spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_rewardrate_learning_curve(spike_data)
    spike_data = Python_PostSorting.Split_DataByReward.split_data_by_reward(spike_data, prm) # function to run if loading from spike dataframe
    spike_data = Python_PostSorting.FirstStopAnalysis_behaviour.extract_first_stop_rewarded(spike_data) # function to run if loading from spike dataframe

    spike_data = Python_PostSorting.RewardAnalysis_behaviour.calculate_reward_rate(spike_data)
    spike_data = Python_PostSorting.BehaviourAnalysis.calculate_graduation_day(spike_data)
    spike_data = Python_PostSorting.BehaviourAnalysis.calculate_progression(spike_data)

    #spike_data = Python_PostSorting.FirstStopAnalysis_behaviour.calculate_first_stop(spike_data) # function to run if loading from position dataframe


    #behaviour_data = spike_data[["session_id", "Mouse", "Day", "Day_numeric", "cohort", "FirstStop", "Reward_Rate"]]
    #behaviour_data.sort_values(by=['Day_numeric'])
    # SAVE DATAFRAMES
    #spike_data = drop_columns_from_frame(spike_data)
    #behaviour_data.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/Behaviour_cohort7.pkl')
    #write_to_csv(np.asarray(behaviour_data))

if __name__ == '__main__':
    main()

