# Data organisation


##Overview  
Dataframes containing in vivo electrophysiology recordings from the virtual reality task saved in pickled (.pkl) format.
The current pipeline runs on a mac computer (Catalina v10.15.7) and uses Python v3.5.1 in PyCharm v2020.2.3 (Edu). 

The framework uses one data frame for analyses at the level of spike clusters. Data from new behavioural sessions, or clusters, are added to the frames as new rows. Outputs of new analyses are added as new columns. Because the data frames store data and analyses, for multiple sessions and clusters respectively, it's straightforward to implement new analyses over many sessions or clusters without writing complicated looping code.
 
##Parameters of the task
The virtual track : the track is 200 cm from 0 - 200 cm. The black boxes are located from 0 - 30 cm and 170 - 200 cm. The reward zone is located from 90 - 110 cm. 
Trial types : 0 : beaconed trials (cued, rewarded)
		1 : non- beaconed trials (uncued, rewarded)
		2 : probe trials (uncued, unrewarded)
Note : not all sessions have probe trials
Rewarded trials : trials in which the animal has stopped successfully in the reward zone (90 - 110 cm) and receives a reward of soymilk through a feeding tube (beaconed and non-beaconed trials). No reward is received in probe trials but it is still considered ‘successful’ if the animal stops in the correct place. 
 
##Organisation of dataframes
The 'clusters' data frame contains data for each cluster and their spatial firing. Each row is a cluster. The columns are organised as follows:
spatial_firing (this is the name of the df in the main code)
session_id : name of main recording folder (example: M5_2018-03-06_15-34-44)
Mouse : id of the mouse (i.e. M3)
Day : Recording day (i.e. D4)
Day_numeric : Recording day (i.e. 4)
cluster_id : id of cluster within session (1 - number of clusters)
mean_firing_rate : total number of spikes / total time
cohort : cohort recorded from (1 - 5)
brain_region : region of the brain tetrodes were identified in (MEC : medial entorhinal cortex; PS : parasubiculum; VC : visual cortex; UC : unidentified)
max_trial_number : max number of trials within the session
firing_times : array of all firing event times that belong to cluster from the virtual reality session (in sampling points - 30kHz)
x_position_cm : array of x coordinates of position of animal in virtual track in cm for each firing event, synchronised with the ephys data 
trial_number : array of the current trial number for each firing event, synchronised with the ephys data
trial_type : arrays of the current trial type (beaconed, non beaconed, probe), for each firing event, synchronised with the ephys data
spike_rate_in_time : nested list of 5 lists, each list has shape of length of recording in ms / 100.  [ [spike rate] [speed] [position] [trial numbers] [trial type] ]
How to extract this data for each cluster :

 
spike_rate_in_time_rewarded : nested list of 5 lists, each list has shape of length of recording in ms / 100. Only includes data for rewarded trials. Speeds < 3 cm/s are removed. [ [spike rate] [speed] [position] [trial numbers] [trial type] ]
How to extract this data for each cluster : 

spikes_in_time : nested list of 6 lists, each list has shape of length of recording in ms / 100. Only includes data for rewarded trials. Speed outliers are removed (> 3 SD from mean). Rates and speed data is convolved (2cm, 2 SD). [ [spike rate] [speed] [position] [acceleration] [trial numbers] [trial type] ]
 
Rates_averaged_rewarded_b : binned data array for each cluster with firing rate maps averaged over beaconed trials 
Rates_averaged_rewarded_nb : binned data array for each cluster with firing rate maps averaged over non-beaconed trials 
Rates_averaged_rewarded_p : binned data array for each cluster with firing rate maps averaged over probe trials 
stop_location_cm : array of positions along the track in cm which the animal has stopped (speed < 4.7 cm/s* *depending on the stage of experiment). 
stop_trial_numbers : array of trial numbers for each stop along the track (1 - max trial number). 
rewarded_stop_locations : array of positions along the track in cm which the animal has stopped (speed < 4.7 cm/s* *depending on stage of experiment). Only includes rewarded trials.
rewarded_trial_numbers : array of trial numbers for each stop along the track. Only includes rewarded trials.

spatial_firing.tail(n=5)

