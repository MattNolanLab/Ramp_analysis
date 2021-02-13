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


def extract_first_stop_rewarded(spike_data, prm):
    spike_data["FirstStop"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_rewarded[4], dtype= np.int32)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()

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

    behaviour_data = spike_data[["session_id", "Mouse", "Day", "FirstStop"]]

    return behaviour_data


