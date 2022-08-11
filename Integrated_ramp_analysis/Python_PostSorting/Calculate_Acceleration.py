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


def generate_acceleration(spike_data, rewarded=False):
    print('I am calculating acceleration...')
    spike_data["spikes_in_time"] = ""
    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0])
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4], dtype= np.int32)

        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]

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

        # remove undefined nan values and by rewarded
        rate_mask = ~np.isnan(rates)
        speed_mask = ~np.isnan(speed)
        acceleration_mask = ~np.isnan(acceleration)
        position_mask = ~np.isnan(position)
        trials_mask = ~np.isnan(trials)
        types_mask = ~np.isnan(types)
        mask = rate_mask & speed_mask & acceleration_mask & position_mask & trials_mask & types_mask

        if rewarded:
            rewarded_mask = np.isin(trials, rewarded_trials)
            mask = mask & rewarded_mask

        # save data
        spike_data = store_acceleration(spike_data, cluster, np.asarray(rates)[mask],
                                                             np.asarray(position)[mask],
                                                             np.asarray(speed)[mask],
                                                             np.asarray(acceleration)[mask],
                                                             np.asarray(trials)[mask],
                                                             np.asarray(types)[mask])
    return spike_data


def store_acceleration(spike_data, cluster_index, rates, position, speed, acceleration, trials, types):
    sn=[]
    sn.append(rates) # rate
    sn.append(position) # position
    sn.append(speed) # speed
    sn.append(acceleration) # acceleration
    sn.append(trials) # trials
    sn.append(types) # types
    spike_data.at[cluster_index, 'spikes_in_time'] = list(sn)
    return spike_data
