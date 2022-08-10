
# How to rerun data for the Tennant et al., 2022 Ramp cell manuscript (R pipeline)

## Overview
This repository was designed to carry out analysis of in vivo electrophysiology recordings from the virtual reality for the Ramp manuscript (Tennant et al., 2022). The aim is to classify neurons based on their firing activity along the linear track of the virtual reality task. 

Specifically, this analyses determines whether firing rates of individual neurons show 'ramping' activity (linear increases or decreases in activity) within specific regions of the track (0-60 cm and 80 - 140 cm), whether firing rate can be explained best by position, speed or acceleration (or a combination), if ramping activity is stable across trial types (beaconed, non-beaconed, probe) and outcomes (hit, run through, try). Subsequent analyses in generates the plots used in figures and performs statistical analysis on processed datasets. 

## What this analysis pipeline does:
1. Runs a linear model on average firing rates binned in space for each neuron to determine if there are linear increases or decreases in activity with position.
2. Shuffles average firing rates binned in space 1000 times, runs the above linear model on each shuffled dataset then classify's the neurons based on the coefficients of the linear model in respect to the coefficients from the 1000 shuffles
3. Runs the above linear model on first the region of the track before the reward zone (0-60 cm) then after (80-140 cm) and assigns a classifyer to the cell based on their activity in both regions. 
4. Runs a general linear mixed effect model on firing rates binned in time (100 ms blocks) to assess the contribution of speed, position and acceleration on firing rates. 
5. Classify neurons based on what variable best explains their firing rate (P,S,A or combination)
6. Run above linear model for nonbeaconed and probe trials
7. Run the above linear model on different trial outcomes (hit, run through, try)

## Preconditions

1. You have the following datafiles that contain all cells for each cohort:
  - ‘PythonOutput_Concat.Rda’
  - OR 'spatial_firing_with1000shuffles.Rda'
Note : These are the same dataframes but the latter has the shuffled analysis already performed (from Figure1_Analysis.Rmd) which saves running time.
_if you have either of these dataframes go to loading data IF rerunning code using already saved .Rda file_

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

## Loading data 

### IF running code for the first time from Python output, 

first the .pkl dataframe needs to be loaded and converted into a .Rda file for R. to do this run the following : 

1.  Open the R Project (RampAnalysis.RProj) in RStudio and navigate to the ‘ConvertPickletoRda.Rmd’ file. Here change the ‘dataframe_to_load’ parameter to the path and name of your .pkl file to be analysed. 
Note : for the Tennant et al., 2022 manuscript, 5 cohorts of animals are loaded individually and then concatenated into one dataframe. _Change accordingly if just one dataframe is needed_
2. Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 
3. Ensure the output of this is saved as ‘PythonOutput_Concat.Rda’ - this is so you can reload it in future to avoid this step.
4. Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat.Rda’ is in the path to the dataframe to load in future. 
_this is so when rerunning analysis you don't have to convert .pkl dataframes again but just reload the concatenated frame._

### IF rerunning code using already saved .Rda file 

1. Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat.Rda’ is in the path to the dataframe to load 
ALTERNATIVELY if you have the spatial_firing_with1000shuffles.Rda datafile load this. This is the same dataframe as PythonOutput_Concat.Rda but with the shuffle analysis done on it already. This will just save time to run the shuffles.
2. Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 
3. Ensure you have curated the data by graduation day and removed artefacts (this should happen automatically if you run the whole markdown document). Spatial_firing should be 1261 rows long (i.e. 1261 cells in the dataset). 

## Running analysis 

Once your dataframe is loaded and your R environment is set up, you are ready to rerun the analysis!

1. Navigate to Figure1_Analysis.Rmd’ and ensure the ‘shuffles’ parameter is set to 1000.
Note : use 10 for testing! Running 1000 takes around 14 hours. 
Note : spatial_firing_with1000shuffles.Rda already has shuffled datasets so should skip this step automatically

2. Run the entire markdown document to generate plots and statistical results for Figure 1. 
3. Navigate to Figure2_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 2. 
4. Navigate to Figure3_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 3. 
4. Navigate to Figure4_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 4. 
5. And so on for the rest of the figures and supplemental figures. 

Output plots should be saved to 'plots/...'
Output data should be saved to 'Data_out/...'


## How to contribute
Please submit an issue to discuss.
