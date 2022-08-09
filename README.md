 
# Overview
This repository contains code for analyses in the manuscript 'Spatial representation by ramping activity of neurons in the retrohippocampal cortex', Tennant et al., 2022 (https://www.biorxiv.org/content/10.1101/2021.03.15.435518v3). 

The repository is organised as follows. The top level folder contains code for analyses carried out in R. This code is used for Figures 1-5 of the manuscript and for the associated supplemental figures. The code uses as an input data that is output from our spike sorting pipeline and then pre-processed with Python scripts. The pre-processing scripts are in the 'Python_PostSorting' folder. The pipeline code is here: https://github.com/MattNolanLab/in_vivo_ephys_openephys

Data used for analyses in the study will be deposited here: https://datashare.ed.ac.uk/handle/10283/777



# Getting started

There are various ways to run analyses. 


## From the pre-processing output:
1. First make a folder in the top level directory called 'data_in'.  Download the file 'PythonOutput_Concat_final.Rda' from the Datashare repository into this folder. 

2. In the file 'Setup.Rmd' uncomment line 60 and comment line 63. Then execute the code in this file. This should load the data, initialise most of the required libraries and initialise functions used for analyses .

3. You should then be able to execute the analyses. Files are named according to the figure for which they generate analyses. Our standard workflow is to execute each analysis in order. There are some dependencies and things may not work if you execute out of order. E.g. Some analyses for Figures 3 and 4 depend on outputs from Figure 2.


## With intermediate R analyses already completed:
Some of the R analyses, for example shuffling and fitting generalised linear mixed effect models are time consuming. To avoid this you can do the following:

1. First make a folder in the top level directory called 'data_in'. Download the file 'SpatialFiring_with_Results.Rda' from the Datashare repository into this folder.

2. Execute the code in 'Setup.Rmd' as above.

3. Proceed as for step 3 above.


## Parameters for saving outputs
In the R code the variables save_figures and save_results are set to zero. This reduces the time to run the code by avoiding saving the outputs. If you wish to save outputs then make high level folders called ' 'data_out' and 'plots' and set these variables to 1.


# Additional documentation
Most analysis files for specific figures also contain documentation. We have tried to make these as complete as possible. Please be aware though that they are working notes and may not be perfect. Some additional documentation is in the /documentation folder (these refer in part to older versions of the code and may no longer be fully accurate).


# Disclaimer
The code has been written with the purpose of analysing data in the manuscript by people with varying levels of coding expertise. It will reproduce analyses in the manuscript. We have done our best given the time constraints available to make the code as readable as possible, but be aware that there are no doubt many areas where the code can be made more elegant.


# Contact
The code is tested using Python (version 3.5.1) and R versions: 3.3.1 to 4.2.0. If you have problems running the code please make sure you are using these versions and you have relevant libraries installed. If you still run into issues, or have suggestions to improve the code, then please submit an issue. When submitting an issue, if you'd like a response then please include a reproducible example of what you're trying to achieve and details of errors produced. If you've done this and don't receive a response within a couple of weeks (we will try to respond sooner, but may have other pressing commitments) then please contact the corresponding author.


