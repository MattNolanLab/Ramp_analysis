
# How to rerun data for the Tennant et al., 2022 Ramp cell manuscript (R pipeline)

## Overview
This repository was designed to carry out analysis of in vivo electrophysiology recordings from the virtual reality for the Ramp manuscript (Tennant et al., 2022). The aim is to classify neurons based on their firing activity along the linear track of the virtual reality task.

Specifically, this analyses determines whether firing rates of individual neurons show 'ramping' activity (linear increases or decreases in activity) within specific regions of the track (0-60 cm and 80 - 140 cm), whether firing rate can be explained best by position, speed or acceleration (or a combination), if ramping activity is stable across trial types (beaconed, non-beaconed, probe) and outcomes (hit, run through, try). Subsequent analyses in generates the plots used in figures and performs statistical analysis on processed datasets. 


## What this analysis pipeline does:
1. Runs a linear model on average firing rates binned in space for each neuron to determine if there are linear increases or decreases in activity with position.
2. Compares the linear model results to 1000 shuffles from a trial-independent cyclic shuffling procedure at the level of spikes (see Methods), to assign a lassifier (+, -, unclassified) for track regions before the reward zone (0-60 cm) and after (80-140 cm)
3. Runs a general linear mixed effect model on firing rates binned in time (100 ms blocks) to assess the contribution of speed, position and acceleration on firing rates. 
4. Runs linear model for different trial outcomes (hit, try, run through)
5. Runs linear model on different trial types (beaconed, probe)
6. Compare open field statistics for ramping and non-ramping neurons

## Preconditions
1. You have the following datafiles that contain all cells for each cohort:
  - ‘PythonOutput_Concat_final.Rda’
  - ‘all_mice_concatenated_shuffle_data_rewarded_unsmoothened.feather’
Alternatively, you have run the python postsorting side of the pipeline, and have a .pkl dataframe containing all cells for each cohort (or a .pkl file for each cohort)
_if you have these .pkl dataframes go to loading data IF running code for the first time from Python output_

2. You have all the correct packages installed (for a list, see ‘Setup.Rmd’) via install.packages("X")
3. You have the correct folder structure
4. Your working directory in R is set to your local repository 

You are now ready to run the R pipeline to analyse, plot and save data for the Tennant et al (2022) manuscript.

## Setting up
1. Navigate to ‘Setup.Rmd’ and ensure save_figures = 1 if you want to save the output figures of the analysis, save_data = 1 if you want to save the output data of the analysis. Set these parameters to 0 if you do not want to save the output of the analysis pipeline. 
2. Ensure you have the correct folder structure. 
Ramp_Analysis as a parent folder, Data, plots and Data_out as child folders. This should already be the structure if you downloaded the Ramp_Analysis repo. 

3. Ensure the datafile/files are in the '/Data' folder.

4. Ensure your RProject has the correct working directory (i.e. to the current repo) for loading/saving data.


## Loading data: IF running code for the first time from Python output, 
first the .pkl dataframe needs to be loaded and converted into a .Rda file for R. to do this run the following : 

1.  Open the R Project (RampAnalysis.RProj) in RStudio and navigate to the ‘ConvertPickletoRda.Rmd’ file. Here change the ‘dataframe_to_load’ parameter to the path and name of your .pkl file to be analysed. 
Note : for the Tennant et al., 2022 manuscript, 5 cohorts of animals are loaded individually and then concatenated into one dataframe. _Change accordingly if just one dataframe is needed_
2. Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 
3. Ensure the output of this is saved as ‘PythonOutput_Concat_final.Rda’ - this is so you can reload it in future to avoid this step.
4. Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat_final.Rda’ is in the path to the dataframe to load in future. 
_this is so when rerunning analysis you don't have to convert .pkl dataframes again but just reload the concatenated frame._

## Loading data: IF rerunning code using already saved .Rda file 
1. Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat_final.Rda’ is in the path to the dataframe to load 
ALTERNATIVELY if you have intermediate version of the R dataframe load this instead. 
2. Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 


## Running analysis 
Once your dataframe is loaded and your R environment is set up, you are ready to rerun the analysis!

1. Run the entire markdown document to generate plots and statistical results for Figure 1. 
2. Navigate to Figure2_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 2. 
3. And so on for the rest of the figures and supplemental figures. 

Output plots should be saved to 'plots/...'
Output data should be saved to 'Data_out/...'


## How to contribute
Please submit an issue to discuss.
