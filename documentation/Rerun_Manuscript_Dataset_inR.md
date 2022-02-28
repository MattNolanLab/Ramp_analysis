
# How to rerun data for the Tennant et al., 2022 Ramp cell manuscript (R pipeline)

Once you have run the python side of the pipeline, you are ready to run the R pipeline to analyse, plot and save data for the Tennant et al (2022) manuscript. 

##Loading data 

IF running code for the first time from Python output, first the .pkl dataframe needs to be loaded and converted into a .Rda file for R. to do this run the following : 

Open the R Project (RampAnalysis.RProj) in RStudio and navigate to the ‘ConvertPickletoRda.Rmd’ file. Here change the ‘dataframe_to_load’ parameter to the path and name of your .pkl file to be analysed. 
Note : for the Tennant et al., 2022 manuscript, 5 cohorts of animals are loaded individually and then concatenated into one dataframe. Change accordingly if just one dataframe is needed. 
Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 
Ensure the output of this is saved as ‘PythonOutput_Concat.Rda’ 
Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat.Rda’ is in the path to the dataframe to load in future. 
Note : this is so when rerunning analysis you don't have to convert .pkl dataframes again but just reload the concatenated frame. 

IF rerunning code using already saved .Rda file 

Navigate to ‘Setup.Rmd’ and under the readRDS function make sure ‘PythonOutput_Concat.Rda’ is in the path to the dataframe to load. 
Run the entire markdown document to give one dataframe for all cohorts (this is called spatial_firing in the code). 

##Running analysis 

Navigate to Figure1_Analysis.Rmd’ and ensure the ‘shuffles’ parameter is set to 1000.
Note : use 10 for testing! Running 1000 takes around 14 hours. 
Run the entire markdown document to generate plots and statistical results for Figure 1. 
Navigate to Figure2_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 2. 
Navigate to Figure3_Analysis.Rmd’ and run the entire markdown document to generate plots and statistical results for Figure 3. 
And so on for the rest of the figures and supplemental figures. 


## How to contribute
Please submit an issue to discuss.
