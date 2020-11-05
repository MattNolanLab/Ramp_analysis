# Import packages
import numpy as np


'''
### Copied from my behaviour code


# SHUFFLE STOPS
def shuffle_stops2( stops,n ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n

    return shuffled_stops




# Input: array[:,4] (columns: location, time, trialno, reward, zeros), array[unique trialnumbers]
# Output: array[20], array[20], array[20], array[20]
# Function: creates shuffled stops datasets from real dataset
# CREATE SHUFFLED DATASETS
def shuffle_analysis(stopsdata, trialids):
    if stopsdata.size == 0:
        return np.zeros((BINNR, )), np.zeros((BINNR, )), np.zeros((BINNR, )), np.zeros((BINNR, ))
    SHUFFLE1 = 100
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = stats.sem(srbin, axis=0)                                 # Array(BINNR)
    # Shuffling random 100 trials 1000 times
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    for i in range(SHUFFLE_N): # for i in 1000
        shuffledtrials = np.zeros((SHUFFLE1, 5))
        shuffleddata =np.zeros((SHUFFLE1, BINNR))
        for n in range(SHUFFLE1): # Create sample data with 100 trials
            trial = random.choice(trialids) # select random trial from real dataset
            data = stopsdata[stopsdata[:,2] ==trial,:] # get data only for each tria
            shuffledtrial = shuffle_stops2(data,n) # shuffle the locations of stops in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stops
        trialids2 = np.unique(shuffledtrials[:, 2]) # find unique trials in the data
        shuffled_srbin = create_srdata( shuffledtrials, trialids2 ) #
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin
    shuffled_mean = np.mean(shuffled_srbin_mean, axis=0)                # Array(BINNR)
    shuffled_std = np.std(shuffled_srbin_mean, axis=0)                  # Array(BINNR)
    return srbin_mean, srbin_std, shuffled_mean, shuffled_std



# Input: array[:,4] (columns: location, time, trialno, reward, empty)
# Output: array[trialnumbers, locationbins]
# Function: Creates histogram of stops in bins
# BIN STOPS INTO 20, 10 CM LOCATION BINS
def create_srdata( stops, trialids ):
    if stops.size == 0:
        return np.zeros((BINNR,))

    # create histogram
    posrange = np.linspace(0, HDF_LENGTH, num=BINNR+1) # 0 VU to 20 VU split into 20
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1

    return H


'''



def add_columns_to_dataframe(spike_data):
    spike_data["shuffled_stops"] = ""
    return spike_data


def load_stop_data(spatial_data):
    locations = np.array(spatial_data.at[1,'stop_location_cm'])
    trials = np.array(spatial_data.at[1,'stop_trial_number'])
    trial_type = np.array(spatial_data.at[1,'stop_trial_type'])
    return np.transpose(np.vstack((locations,trials,trial_type)))


def shuffle_stops_in_time( spikes):
    shufflen=10
    shuffled_trials = np.zeros((spikes.shape[0]))
    for n in range(shufflen):
        shuffled_spikes = np.copy(spikes) # this is required as otherwise the original dataset would be altered
        shuffled_spikes[:,0] = np.random.shuffle(shuffled_spikes[:,0])
        shuffled_trials = np.vstack((shuffled_trials, shuffled_spikes[:,0]))
    avg_shuffled_trials=np.nanmean(shuffled_trials, axis=0)
    shuffled_spikes[:,0] = avg_shuffled_trials
    return shuffled_spikes


def shuffle_analysis(cluster_firings, trialids):
    shuffledtrials = np.zeros((0, 3))
    for trial in range(1,int(trialids)):
        trial_data = cluster_firings[cluster_firings[:,1] == trial,:]# get data only for each trial
        shuffledtrial = shuffle_stops_in_time(trial_data) # shuffle the locations of spikes in the trial
        shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop
    return shuffledtrials


def generate_shuffled_data_for_stops(spike_data):
    print('generating shuffled data')
    spike_data=add_columns_to_dataframe(spike_data)

    for cluster in range(len(spike_data)):
        cluster_firings = load_stop_data(spike_data)
        max_trial = np.nanmax(cluster_firings[:,1])+1
        shuffled_cluster_firings = shuffle_analysis(cluster_firings, max_trial)
        spike_data = add_data_to_dataframe(cluster, shuffled_cluster_firings, spike_data)
    return spike_data


def add_data_to_dataframe(cluster_index, shuffled_cluster_firings, spike_data):
    spike_data.at[cluster_index, 'shuffled_stops'] = np.array(shuffled_cluster_firings[:,0])
    return spike_data
