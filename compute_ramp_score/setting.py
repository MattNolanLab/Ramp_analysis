###################
# recording setting
numChan = 16 #total number of recording channel
numTetrode = 4 #number of tetrode
Fs = 30000 #sampling frequency
positionChannel = 'ADC2'

###################
# EEG setting
eeg_ds = 100 #downsmaple to 100Hz

###################
# Binning
theta_bin = 18
position_bin = 100
speed_bin = 20
accel_bin = 20 #old:20
binSize = 100 # in ms

speed_thres = 3 # in cm/s , threshold above which spike train will be considered
###################
# VR
trackLength = 200

base_folder = 'E:/in_vivo_vr/sarah_glm_202006/'

debug_folder = base_folder+'cohort_3/M1_D31_2018-11-01_12-28-25'


##################
# Modelling
valFolds = 10 #number of fold for cross-validation