# Extracellular electrophysiology analysis for tetrode data recorded in the virtual reality


## Overview
This repository was designed to carry out analysis of in vivo electrophysiology recordings from the virtual reality for the Ramp manuscript (Tennant et al., 2022). The aim is to classify neurons based on their firing activity along the linear track of the virtual reality task. 


For the analysis of electrophysiological and behavioural data from the virtual reality task we have developed a framework that uses data frames implemented in Python and R. Python is used to pre-process the experimental data. For example, splitting data on trials based on success or failure, calculating acceleration and removing stationary activity. The outputs of this pre-processing, saved in pickled (.pkl) format, are then used for subsequent analyses in R. Analysis in R generates the plots used in figures and performs statistical analysis on processed datasets. 

Analysis in R classifies whether firing rates of individual neurons show 'ramping' activity (continuous increases or decreases in activity) within specific regions of the track (0-60 cm and 80 - 140 cm). For this, a linear model is ran on average firing rates binned in space for each neuron. We are also interested to see whether firing rate can be explained best by position, speed or acceleration (or a combination). For this we run a GLMER on firing rates binned in time for each neuron. 


Please see more detailed documentation in the /documentation folder.


## How to contribute
Please submit an issue to discuss.
