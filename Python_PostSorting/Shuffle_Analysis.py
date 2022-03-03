# Import packages
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

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





def load_data(spike_data, cluster):
    rates = np.array(spike_data.at[cluster,'Rates_bytrial_rewarded_b'])
    position = np.array(np.tile(np.arange(0,199,1), rates.shape[1]))
    trials = np.array(np.repeat(np.arange(1,rates.shape[1]+1,1), 199))

    rates = np.reshape(rates, (rates.shape[0]*rates.shape[1]))
    data = np.transpose(np.vstack((rates,position,trials )))
    return data


def run_linearmodel_on_shuffle(shuffledtrials):
    trials = int(shuffledtrials.shape[0]/199)
    data_b = np.reshape(np.array(shuffledtrials[:,0]), (199,trials))
    avg_shuffled_trials=np.nanmean(data_b, axis=1)
    position = np.arange(0,199,1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(avg_shuffled_trials,position)

    return(slope, r_value, p_value)



def shuffle_spikes_on_trial( spikes):
    shuffled_spikes = np.copy(spikes) # this is required as otherwise the original dataset would be altered
    np.random.shuffle(shuffled_spikes[:,0])
    return shuffled_spikes


def shuffle_analysis(cluster_firings, trialids):
    shufflen=10
    shuffled_values = np.zeros((0, 3))

    for n in range(shufflen):
        shuffledtrials = np.zeros((0, 3))
        for trial in range(1,int(trialids)):
            trial_data = cluster_firings[cluster_firings[:,2] == trial,:]# get data only for each trial
            shuffledtrial = shuffle_spikes_on_trial(trial_data) # shuffle the locations of spikes in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop

        slope, r_value, p_value = run_linearmodel_on_shuffle(shuffledtrials)
        values = np.array([slope, r_value, p_value])
        shuffled_values = np.vstack((shuffled_values,values)) # stack shuffled stop

    return shuffled_values


def generate_shuffled_data(spike_data):
    print('generating shuffled data')
    spike_data["shuffled_values"] = ""

    for cluster in range(len(spike_data)):
        cluster_firings = load_data(spike_data, cluster)
        max_trial = np.nanmax(cluster_firings[:,2])+1
        shuffled_values = shuffle_analysis(cluster_firings, max_trial)
        spike_data.at[cluster, 'shuffled_values'] = list(shuffled_values)
    return spike_data

