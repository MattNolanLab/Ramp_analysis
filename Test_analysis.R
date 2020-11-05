# Test analysis
# Start with simple code to load and look at data

library(reticulate) # package that allows R to call python code
dataframe_to_load <- "data/Allmice_alldays_cohort1_processed1.pkl" 

use_python("/Users/mattnolan/anaconda/bin/python")
py_config()
source_python("pickle_reader.py")

spatial_firing_test <- read_pickle_file(file.path(dataframe_to_load))

colnames(spatial_firing_test)

#Look at just the first cluster
spatial_firing_test[1,]$session_id
spatial_firing_test[1,]$rewarded_trials

str(spatial_firing_test[1,]$spike_rate_on_trials)

