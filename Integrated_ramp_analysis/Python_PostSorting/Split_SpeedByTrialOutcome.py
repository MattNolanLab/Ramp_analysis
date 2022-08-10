import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats


def split_and_save_speed_data(spike_data):
    spike_data["speed_in_time_reward"] = ""
    spike_data["speed_in_time_try"] = ""
    spike_data["speed_in_time_run"] = ""
    for cluster in range(len(spike_data)):
        try_trials = np.array(spike_data.loc[cluster, 'try_trialid'])
        runthru_trials = np.array(spike_data.loc[cluster, 'run_through_trialid'])
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])

        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        # remove outliers
        mean_speed = np.nanmean(speed)
        sd_speed = np.nanstd(speed)
        upper_speed_sd = mean_speed+(sd_speed*3)

        data = np.vstack((speed, position, trials))
        data=data.transpose()
        data = data[data[:,0] < upper_speed_sd,:]
        speed = data[:,0]
        position = data[:,1]
        trials = data[:,2]

        rewarded_speed = speed[np.isin(trials,rewarded_trials)]
        rewarded_position = position[np.isin(trials,rewarded_trials)]
        failed_speed = speed[np.isin(trials,runthru_trials)]
        failed_position = position[np.isin(trials,runthru_trials)]
        try_speed = speed[np.isin(trials,try_trials)]
        try_position = position[np.isin(trials,try_trials)]

        spike_data = drop_data_into_frame(spike_data, cluster, rewarded_speed , rewarded_position, failed_speed, failed_position, try_speed, try_position)

    return spike_data



def drop_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f):
    sn=[]
    sn.append(a) # speed
    sn.append(b) # position
    spike_data.at[cluster_index, 'speed_in_time_reward'] = list(sn)

    sn=[]
    sn.append(c) # speed
    sn.append(d) # position
    spike_data.at[cluster_index, 'speed_in_time_run'] = list(sn)

    sn=[]
    sn.append(e) # speed
    sn.append(f) # position
    spike_data.at[cluster_index, 'speed_in_time_try'] = list(sn)

    return spike_data


def extract_time_binned_speed_by_outcome(spike_data):
    print("calculating mean speed for trial outcomes...")
    spike_data["Speed_mean_try"] = ""
    spike_data["Speed_mean_run"] = ""
    spike_data["Speed_mean_rewarded"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].speed_in_time_try[0])
        position=np.array(spike_data.iloc[cluster].speed_in_time_try[1])
        speed = convolve_with_scipy(speed)

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape))
        binned_speed_sd = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            speed_in_position = np.take(speed, np.where(np.logical_and(position >= rowcount, position <= rowcount+1)))
            average_speed = np.nanmean(speed_in_position)
            sd_speed = np.nanstd(speed_in_position)
            binned_speed[rowcount] = average_speed
            binned_speed_sd[rowcount] = sd_speed
        binned_speed = convolve_with_scipy(binned_speed)
        spike_data.at[cluster, 'Speed_mean_try'] = list(binned_speed)


        speed=np.array(spike_data.iloc[cluster].speed_in_time_run[0])
        position=np.array(spike_data.iloc[cluster].speed_in_time_run[1])
        speed = convolve_with_scipy(speed)

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape))
        binned_speed_sd = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            speed_in_position = np.take(speed, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
            average_speed = np.nanmean(speed_in_position)
            sd_speed = np.nanstd(speed_in_position)
            binned_speed[rowcount] = average_speed
            binned_speed_sd[rowcount] = sd_speed
        binned_speed = convolve_with_scipy(binned_speed)
        spike_data.at[cluster, 'Speed_mean_run'] = list(binned_speed)


        speed=np.array(spike_data.iloc[cluster].speed_in_time_reward[0])
        position=np.array(spike_data.iloc[cluster].speed_in_time_reward[1])
        speed = convolve_with_scipy(speed)

        position_array = np.arange(1,201,1)
        binned_speed = np.zeros((position_array.shape))
        binned_speed_sd = np.zeros((position_array.shape))
        for rowcount, row in enumerate(position_array):
            speed_in_position = np.take(speed, np.where(np.logical_and(position >= rowcount, position < rowcount+1)))
            average_speed = np.nanmean(speed_in_position)
            sd_speed = np.nanstd(speed_in_position)
            binned_speed[rowcount] = average_speed
            binned_speed_sd[rowcount] = sd_speed
        binned_speed = convolve_with_scipy(binned_speed)
        spike_data.at[cluster, 'Speed_mean_rewarded'] = list(binned_speed)

    return spike_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=3)
    convolved_rate = signal.convolve(rate, window, mode='same')
    return (convolved_rate/ sum(window))




