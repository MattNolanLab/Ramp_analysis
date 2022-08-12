# Extracellular electrophysiology analysis for tetrode data


## Overview
This repository was designed to carry out analysis of in vivo electrophysiology recordings from the virtual reality for the Ramp manuscript (Tennant et al., 2022). The aim is to classify neurons based on their firing activity along the linear track of the virtual reality task. 

For the analysis of electrophysiological and behavioural data from the virtual reality task we have developed a framework that uses data frames implemented in Python and R. Python is used to pre-process the experimental data. For example, splitting data based on trial outcome. The outputs of this pre-processing, saved in pickled (.pkl) format, are then used for subsequent analyses in R. Analysis in R generates most of the plots used in figures and performs statistical analysis on processed datasets. 

Analysis in R classifies whether firing rates of individual neurons show 'ramping' activity (continuous increases or decreases in activity) within specific regions of the track (0-60 cm and 80 - 140 cm). For this, a linear model is ran on average firing rates binned in space for each neuron. We are also interested to see whether firing rate can be explained best by position, speed or acceleration (or a combination). For this we run a GLMER on firing rates binned in time for each neuron. 


## Versions
The current pipeline runs on a mac computer (macOS Catalina v10.15.7) platform x86_64-apple-darwin19.6.0 (64-bit) and uses Python v3.5.1 in PyCharm v2020.2.3 (Edu) and R version 4.1.1 (2021-08-10) in R studio v1.3.1093 (2009-2020).

The following python packages are required for the pipeline to run : 
Pandas v1.3.2, numpy v1.19.2, matplotlib v3.3.2, scipy v1.5.4

The following R and RStudio packages are required for the pipeline to run : 
dplyr v1.0.7, purrr v0.3.4, Pheatmap v1.0.12, RColorBrewer v1.1-2, tidyr v1.1.4 , ggplot2 v3.3.5,  tidyverse v1.3.1, broom v0.7.11, lme4 v1.1-27.1, agricolae v1.3-5, plotrix v3.8-2, Metrics v0.1.4, Hmisc v4.6-0 , scales v1.1.1, networkD3 v0.4 


## Preconditions to running RampAnalysis
1. All Open Ephys recordings in virtual reality have been sorted as described : https://github.com/MattNolanLab/in_vivo_ephys_openephys/blob/master/documentation/user_guide_for_running_analysis.md
2. The postsorting analysis for processing spatial firing outputs the following columns : ‘spike_rates_on_trials_smoothed’, ‘spike_rates_in_time’ *note these are necessary for the RampAnalysis pipeline. 
_Note : Entire list of necessary columns for analysis is in ‘Data_Organisation.md’._ 
3. Dataframes have been concatenated across days/mice of interest using ‘concantenate_spatial_firing.py’ 
Note : Harry also has a version of this
4. If necessary (i.e. if the data is needed), paired Open field recordings are sorted together with the virtual reality recordings and dataframes from paired open field recordings
5. _You now have a single dataframe per cohort of mice/per mouse with all the recordings days in it and you want to analyse the data further._

## Steps to reproduce analysis
1. follow guide for pre-processing Open Ephys recordings in Python:
https://github.com/MattNolanLab/Ramp_analysis/blob/master/documentation/Rerun_manuscript_dataset_inPython.md
2. Follow guide for producing figures and stats in R:
https://github.com/MattNolanLab/Ramp_analysis/blob/master/documentation/Rerun_Manuscript_Dataset_inR.md


## How to contribute
Please submit an issue to discuss.
