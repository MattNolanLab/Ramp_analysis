
#This file contains the parameters for analysis


##########
# Folder locations
mountainsort_tmp_folder ='/tmp/mountainlab/'
sorting_folder = '/home/ubuntu/to_sort/recordings/'
to_sort_folder = '/home/ubuntu/to_sort/'
server_path_first_half = '/mnt/datastore/'
downtime_lists_path = '/home/ubuntu/to_sort/sort_downtime/'
param_file = '/parameters.txt'
dead_channel_file_name = "/dead_channels.txt"

##########
# Recording setting
sampling_rate = 30000
down_sampled_rate = 1000
num_tetrodes = 4
movement_ch_suffix = f'ADC2' #channel that contains the movement data
opto_ch_suffix = f'ADC3'
data_file_prefix = f'_CH' #prefix of data files
data_file_suffix = ''
wave_form_size = 40
tetrodeNum = 4 #how many channel in one tetrode

#########
# sorter configuration
sorterName = 'MountainSort'
is_tetrode_by_tetrode = False #set to True if you want the spike sorting to be done tetrode by tetrode
all_tetrode_together = True #set to True if you want the spike sorting done on all tetrodes combined


############
# Analysis
location_bin_num = 200 #number of location bin
stop_threshold = 4.7 #threshold for detecting stop
location_ds_rate = 1000 #the sampleing frequency in Hz to downsample the location signal to

##########
# VR
track_length = 200 # cm
vr_bin_size_cm = 1
first_trial_channel_suffix = f'ADC4' #channel for the start of trial
second_trial_channel_suffix = f'ADC5' #channel for the stp of trial
reward_start = 90 #position for the reward
reward_end = 110 #position for the reward
movement_threshold = 3
time_bin_size = 0.1 #seconds
guassian_std_for_smoothing_in_time_seconds = 0.2
guassian_std_for_smoothing_in_space_cm = 2
movement_channel='100_ADC2.continuous'
first_trial_channel='100_ADC4.continuous'
second_trial_channel='100_ADC5.continuous'
goal_location_chennl='100_ADC7.continuous'

##########
# Experiment
session_type = 'vr'


##########
# open field
opto_tagging_start_index = None
pixel_ratio = 440
sync_channel_suffix = 'ADC1' #channel for the sync pulse
bonsai_sampling_rate = 30
gauss_sd_for_speed_score = 250
open_field_bin_size_cm = 2.5

# useful for a shuffled analysis run on eddie
use_vectorised_rate_map_function = True
impose_num_cores = False
fixed_num_cores = 1
