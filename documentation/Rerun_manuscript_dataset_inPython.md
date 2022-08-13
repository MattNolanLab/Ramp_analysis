
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

There are three main stages to running the python post sorting pipeline:
1. Post Sorting data from the virtual reality
2. Post Sorting data from the open field 
3. Concatenating the two together

Run in this order:
1. Integrated_ramp_analyis/Python_PostSorting/Control_PostSorting_Analysis.py
2. Integrated_ramp_analyis/Python_PostSorting/Control_PostSorting_Analysis_of.py
3. 1.Integrated_ramp_analyis/Python_PostSorting/Match_Session_and_Cluster.py

Once all scripts are run, you can begin running the R scripts following:
https://github.com/MattNolanLab/Ramp_analysis/blob/master/documentation/Rerun_Manuscript_Dataset_inR.md


## How to contribute
Please submit an issue to discuss.
