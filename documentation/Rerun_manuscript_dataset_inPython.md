
# How to rerun data for the Tennant et al., 2022 Ramp cell manuscript (Python pipeline)

## What this analysis pipeline does:
1. Concatenates spatial firing data frames from each cohort directories, 
(Individual recordings from each mouse, session and cohort are spike sorted using the following pipeline:
'https://github.com/MattNolanLab/in_vivo_ephys_openephys')
2. Adds brain regions and ramp scores to concatenatedspatial firing dataframes
3. Curates the concatenated spatial firing dataframes by graduation day, lick artefacts, low trial number sessions (<30 trials) and brain region
4. Plots behaviour and firing rate plots for each cell by different trial types and trial outcomes. 
5. Saves dataframes for reading in R

Additionally
6. Runs shuffled analysis for individual spatial firing dataframes
7. Concatenates shuffled spatial firing dataframes and saves as a .feather file for reading in R

## Preconditions
1. You have the following datafiles that contain all cells for each cohort:
  - ‘cohortX_concatenated_spike_data.pkl’ x 5 cohorts
  - ‘ramp_scores.csv’: The ramp manuscript utilises a 'ramp score' as described in Methods
  - ‘tetrode_locations.csv’: Brain regions are labelled by recording session based on inspection of theta LFP and anatomical images collected via microCT imaging or cresyl violet staining
  - ‘criteria_days.csv’: Graduation days indicates the day each mouse graduates to probe trials as per Methods

 You are now ready to run the Python pipeline to analyse, plot and save data for the Tennant et al (2022) manuscript.


## Setting up
Note: Concatenated spatial firing and shuffle dataframes are already provided for users looking to reproduce the analysis, therefore Integrated_ramp_analysis/Concatenate_spatial_firing.py, Integrated_ramp_analysis/shuffle_analysis.py, Integrated_ramp_analysis/Concatenate_vr_shuffle_analysis.py python scipts are not possible without the large recording files (not provided).

1. Navigate to Integrated_ramp_analyis/Python_PostSorting/Control_PostSorting_Analysis, ensure load and save paths for the cohort spatial firing concatenated dataframes, ramp score dataframe, tetrode location dataframe, criteria day dataframe and the plots save path all correspond to where you want to load and save data from. 
2. Navigate to Integrated_ramp_analyis/Python_PostSorting/Control_PostSorting_Analysis_of, ensure load and save paths for the cohort open field spatial firing concatenated correspond to where you want to load and save data from
3. Navigate to Integrated_ramp_analyis/Python_PostSorting/Match_Session_and_Cluster, ensure load and save paths for the cohort open field spatial firing concatenated correspond to where you want to load and save data from (only the cohort 7 dataframe is appended with open field metrics)

## Steps to run RampAnalysis (Python pipeline)

There are three main stages to running the python post sorting pipeline. 

1. Post Sorting data from the virtual reality
2. Post Sorting data from the open field 
3. Concatenating the two together

The third part of this assumes for each mouse and day you have a recording from the virtual reality AND a recording from the open arena and requires that you have sorted your virtual reality and matching open field recordings together (i.e. from Harry's pipeline). This ensures clusters are matched from the virtual reality and open field and have the same cluster_id number across the two dataframes.


## Post Sorting data from the virtual reality

1. Copy the path to your data frame (the concatenated one for a whole cohort) in the process_allmice_dir function in ‘LoadDataFrames.py’
2. Go to the main function in ‘Control_PostSorting_Analysis.py’. The following functions should be uncommented : 
- remove_false_positives
- curate_data
- make_neuron_number
- add_mouse_to_frame
- load_crtieria_data_into_frame
- load_brain_region_data_into_frame
- load_Teris_ramp_score_data_into_frame
- Run_main_figure_analysis   
- run_supple_figure_analysis
- drop_columns_from_frame

3. In run_main_figure_analysis  the following functions should be uncommented: 
- split_data_by_reward
- extract_time_binned_firing_rate
- generate_acceleration_rewarded_trials
  
4. In run_supple_figure_analysis the following functions should be uncommented:
- split_time_data_by_trial_outcome
- extract_time_binned_firing_rate_runthru_allspeeds
- extract_time_binned_firing_rate_try_allspeeds
- extract_time_binned_firing_rate_rewarded_allspeeds
- split_and_save_speed_data
- extract_time_binned_speed_by_outcome
- calc_histo_speed

5. The last line of the main function contains a path for the dataframe to be saved too, change this to your desired output and dataframe name (lets call it dataframe1). 
 
## Post sorting open field recordings

6. Copy the path to your data frame in the process_allmice_dir_of function in ‘LoadDataFrames.py’. Note : this is the open field dataframe (i.e. of recordings from the open field), sorted WITH the VR
7. Go to the main function in ‘Control_PostSorting_Analysis_of.py’. The following functions should be uncommented : 
- calculate_spike_width
- generate_spike_isi

8. The last line of the main function contains a path for the dataframe to be saved too, change this to your desired output and dataframe name. Note : choose a DIFFERENT name to the output of the VR analysis, let's call it dataframe2.  

## Match open field and virtual reality recordings

9. Go to ‘Match_Session_and_Cluster.py’ and in process_allmice_of copy the path to your data frame output from the ‘Control_PostSorting_Analysis_of.py’ i.e. dataframe2. 
10. In process_allmice_VR copy the path to your data frame output from the ‘Control_PostSorting_Analysis.py’ i.e. dataframe1. 
11. Navigate to the main function in ‘Match_Session_and_Cluster.py’. Make sure the output path of the dataframe has a distinct name. 
12. Run the main function in ‘Match_Session_and_Cluster.py’.

Now you have ran the python postsorting pipeline, you are ready to run your datasets in R. Please switch to 

## How to contribute
Please submit an issue to discuss.
