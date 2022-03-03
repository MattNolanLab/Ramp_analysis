import os
import matplotlib.pylab as plt
import numpy as np
import cmath
import pandas as pd
from scipy import signal



"""

## Instantaneous acceleration calculation

The following functions calculates instantaneous acceleration from speed 

- load rewarded spike rates in time
- calculate diff in speed
- store in dataframe


"""


def generate_acceleration_rewarded_trials(spike_data, recording_folder):
    print('I am calculating acceleration...')
    spike_data["spikes_in_time"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1])
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[0])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4], dtype= np.int32)

        try:
            window = signal.gaussian(2, std=2)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except (ValueError, TypeError):
                continue


        if speed.size > 1:
            acceleration = np.diff(np.array(speed)) # calculate acceleration
            acceleration = np.hstack((0, acceleration))

            # remove acceleration outliers outside 3 SD from the mean
            mean_speed = np.nanmean(acceleration)
            sd_speed = np.nanstd(acceleration)
            upper_speed_sd = mean_speed+(sd_speed*3)

            data = np.vstack((rates, speed, acceleration, position, trials, types))
            data=data.transpose()
            data = data[data[:,2] < upper_speed_sd,:] # remove acceleration outliers outside 3 SD from the mean
            rates = data[:,0]
            speed = data[:,1]
            acceleration = data[:,2]
            position = data[:,3]
            trials = data[:,4]
            types = data[:,5]
        else:
            acceleration = np.zeros((speed.size))
        # save data
        spike_data = store_acceleration_for_rewaded(spike_data, cluster, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration), np.asarray(trials), np.asarray(types))
    return spike_data



def store_acceleration_for_rewaded(spike_data,cluster_index, rates, position, speed, acceleration,  trials, types):
    sn=[]
    sn.append(rates) # rate
    sn.append(position) # position
    sn.append(speed) # speed
    sn.append(acceleration) # acceleration
    sn.append(trials) # trials
    sn.append(types) # types
    spike_data.at[cluster_index, 'spikes_in_time'] = list(sn)
    return spike_data
