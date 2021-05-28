import numpy as np
import matplotlib.pylab as plt
import csv
import pandas as pd
import math

### -------------------------------------------------------------- ###

### Following calculates the average first stop for each cluster's session

### -------------------------------------------------------------- ###


## Save first stop learning curves to .csv file
def write_to_csv(array, deviation_array):
    csvData = np.vstack((array,deviation_array))
    with open('/Users/sarahtennant/Work/Analysis/RampAnalysis/data/FirstStop_allmice.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    return


def extract_first_stop_rewarded(spike_data):
    spike_data["FirstStop"] = ""
    spike_data["first_stop_locations"] = ""
    spike_data["first_stop_trials"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[4], dtype= np.int32)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        try:
            # bin data over position bins
            trial_numbers = np.arange(min(trials),max(trials), 1)
            binned_data = np.zeros((trial_numbers.shape[0])); binned_data[:] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                trial_data = trial_data[trial_data[:,2] > 30,:]
                trial_data = trial_data[trial_data[:,2] <= 110,:]

                speed = trial_data[trial_data[:,1] < 4.7, 2]
                try:
                    first_speed = speed[1]
                except IndexError:
                    first_speed = np.nan
                binned_data[tcount] = first_speed
                mean_first_stop=np.nanmean(binned_data)
            spike_data.at[cluster, 'FirstStop'] = mean_first_stop
        except ValueError:
            spike_data.at[cluster, 'FirstStop'] = np.nan

    return spike_data




def extract_first_stop_per_cluster(spike_data):
    print("calculating first stop per cluster...")
    spike_data["first_stop_locations"] = ""
    spike_data["first_stop_trials"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[4], dtype= np.int32)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        try:
            # bin data over position bins
            trial_numbers = np.unique(trials)
            binned_data = np.zeros((trial_numbers.shape[0], 2)); binned_data[:,:] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                trial_data = trial_data[trial_data[:,2] > 30,:]
                trial_data = trial_data[trial_data[:,2] <= 110,:]

                speed = trial_data[trial_data[:,1] < 4.7, 2]
                try:
                    first_speed = speed[1]
                except IndexError:
                    first_speed = np.nan
                binned_data[tcount,0] = first_speed
                binned_data[tcount,1] = trial
            spike_data.at[cluster, 'first_stop_locations'] = binned_data[:,0]
            spike_data.at[cluster, 'first_stop_trials'] = binned_data[:,1]
        except ValueError:
            spike_data.at[cluster, 'first_stop_locations'] = np.nan
            spike_data.at[cluster, 'first_stop_trials'] = np.nan

    return spike_data





def calculate_first_stop(spike_data):
    print('calculating first stop')
    spike_data["FirstStop"] = ""
    spike_data["SD_FirstStopcm"] = ""

    for cluster in range(len(spike_data)):
        stop_location_cm=np.array(spike_data.loc[cluster].stop_location_cm)
        stop_trial_number=np.array(spike_data.loc[cluster].stop_trial_number)
        rewarded_trials=np.array(spike_data.loc[cluster].rewarded_trials)

        rewarded_stop_location_cm = stop_location_cm[np.isin(stop_trial_number,rewarded_trials)]
        rewarded_stop_trial_number = stop_trial_number[np.isin(stop_trial_number,rewarded_trials)]

        data=np.vstack((rewarded_stop_location_cm,rewarded_stop_trial_number))
        data = np.transpose(data)

        data = data[data[:,0] > 32,:] # filter data for beaconed trials
        data = data[data[:,0] <= 110,:] # filter data for beaconed trials

        trials = data[:,1]
        position = data[:,0]

        firststop_over_trials = []
        for trialcount, trial in enumerate(trials):
            locations = np.take(position, np.where(trials == trial)[0])
            try:
                first_location = locations[0]
                firststop_over_trials = np.append(firststop_over_trials, first_location)
            except IndexError:
                print("")

        avg_firststop = np.nanmean(firststop_over_trials)
        sd_firststop = np.nanstd(firststop_over_trials)
        spike_data.at[cluster, 'FirstStop'] = avg_firststop
        spike_data.at[cluster, 'SD_FirstStopcm'] = sd_firststop

    return spike_data





def calculate_first_stop_per_cell(spike_data):
    print('calculating first stop')
    #spike_data["FirstStop"] = ""
    #spike_data["SD_FirstStopcm"] = ""

    spike_data["first_stop_locations"] = ""
    spike_data["first_stop_trials"] = ""

    for cluster in range(len(spike_data)):
        stop_location_cm=np.array(spike_data.loc[cluster].stop_location_cm)
        stop_trial_number=np.array(spike_data.loc[cluster].stop_trial_number)
        rewarded_trials=np.array(spike_data.loc[cluster].rewarded_trials)

        rewarded_stop_location_cm = stop_location_cm[np.isin(stop_trial_number,rewarded_trials)]
        rewarded_stop_trial_number = stop_trial_number[np.isin(stop_trial_number,rewarded_trials)]

        data=np.vstack((rewarded_stop_location_cm,rewarded_stop_trial_number))
        data = np.transpose(data)

        data = data[data[:,0] > 32,:] # filter data for beaconed trials
        data = data[data[:,0] <= 150,:] # filter data for beaconed trials

        trials = data[:,1]
        position = data[:,0]

        firststop_over_trials = []
        trials_over_trials = []

        for trialcount, trial in enumerate(trials):
            locations = np.take(position, np.where(trials == trial)[0])
            try:
                first_location = locations[0]
                firststop_over_trials = np.append(firststop_over_trials, first_location)
                trials_over_trials = np.append(trials_over_trials, trial)
            except IndexError:
                print("")

        #avg_firststop = np.nanmean(firststop_over_trials)
        #sd_firststop = np.nanstd(firststop_over_trials)
        spike_data.at[cluster, 'first_stop_locations'] = firststop_over_trials
        spike_data.at[cluster, 'first_stop_trials'] = trials_over_trials

    return spike_data

