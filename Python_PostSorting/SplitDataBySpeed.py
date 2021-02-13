import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats


def extract_data_from_frame(spike_data, cluster):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

    window = signal.gaussian(2, std=2)
    speed = signal.convolve(speed, window, mode='same')/ sum(window)
    data = np.vstack((rates, speed, position, trials, types))
    data=data.transpose()
    return rewarded_trials, data


def split_trials(data, rewarded_trials):
    rates = data[:,0]
    speed = data[:,1]
    position = data[:,2]
    trials = data[:,3]
    types = data[:,4]

    rewarded_rates = rates[np.isin(trials,rewarded_trials)]
    rewarded_speed = speed[np.isin(trials,rewarded_trials)]
    rewarded_position = position[np.isin(trials,rewarded_trials)]
    reward_trials = trials[np.isin(trials,rewarded_trials)]
    reward_types = types[np.isin(trials,rewarded_trials)]
    failed_rates = rates[np.isin(trials,rewarded_trials, invert=True)]
    failed_speed = speed[np.isin(trials,rewarded_trials, invert=True)]
    failed_position = position[np.isin(trials,rewarded_trials, invert=True)]
    failed_trials = trials[np.isin(trials,rewarded_trials, invert=True)]
    failed_types = types[np.isin(trials,rewarded_trials, invert=True)]

    data = np.vstack((rewarded_rates, rewarded_speed, rewarded_position, reward_trials, reward_types))
    hit_data = data.transpose()

    data = np.vstack((failed_rates, failed_speed, failed_position, failed_trials , failed_types))
    miss_data=data.transpose()

    return hit_data, miss_data


def convolve_with_scipy(rate):
    window = signal.gaussian(2, std=2)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def find_avg_speed_in_reward_zone(data):
    trialids = np.array(np.unique(np.array(data[:,3])))
    speeds = np.zeros((trialids.shape[0]))
    for trialcount, trial in enumerate(trialids):
        trial_data = data[data[:,3] == trial,:] # get data only for each trial
        data_in_position = trial_data[trial_data[:,2] >= 90,:]
        data_in_position = data_in_position[data_in_position[:,2] <= 110,:]
        speeds[trialcount] = np.nanmean(data_in_position[:,1])
    return speeds, trialids


def find_confidence_interval(speeds):
    mean, sigma = np.nanmean(speeds), np.nanstd(speeds)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]
    return upper, lower


def catagorise_failed_trials(speed_array_failed, trialids_failed, upper, lower):
    trial_id_runthrough = []
    trial_id_try = []

    for rowcount, row in enumerate(speed_array_failed):
        speed = speed_array_failed[rowcount]
        trial = trialids_failed[rowcount]

        if speed < upper:
            trial_id_try = np.append(trial_id_try, trial)
        elif speed >= upper:
            trial_id_runthrough = np.append(trial_id_runthrough, trial)
    return trial_id_try, trial_id_runthrough


def extract_rewarded_confidence_intervals(data):
    speeds_rewarded, trialids_rewarded = find_avg_speed_in_reward_zone(data)
    upper, lower = find_confidence_interval(speeds_rewarded)
    rewarded_ci = np.array((upper, lower))
    return rewarded_ci


def split_time_data_by_speed(spike_data, prm):
    spike_data["run_through_trialid"] = ""
    spike_data["try_trialid"] = ""
    spike_data["spikes_in_time_try_b"] = ""
    spike_data["spikes_in_time_runthru_b"] = ""
    spike_data["speed_CI_rewarded"] = ""
    spike_data["speed_CI_try"] = ""
    spike_data["speed_CI_run"] = ""

    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)
        hit_data, miss_data = split_trials(data, rewarded_trials)
        rewarded_ci = extract_rewarded_confidence_intervals(hit_data)
        speed_array_failed, trialids_failed = find_avg_speed_in_reward_zone(miss_data)

        trial_id_try, trial_id_run = catagorise_failed_trials(speed_array_failed, trialids_failed, rewarded_ci[0], rewarded_ci[1])

        spike_data.at[cluster,"run_through_trialid"] = pd.Series(trial_id_run)
        spike_data.at[cluster,"try_trialid"] = pd.Series(trial_id_try)

        spike_data, data_try, data_run = split_and_save_data(spike_data)

        speed_array_try, trialids_try = find_avg_speed_in_reward_zone(data_try)
        upper, lower = find_confidence_interval(speed_array_try)
        try_ci = np.array((upper, lower))
        speed_array_run, trialids_run = find_avg_speed_in_reward_zone(data_run)
        upper, lower = find_confidence_interval(speed_array_run)
        run_ci = np.array((upper, lower))

        spike_data.at[cluster,"speed_CI_rewarded"] = rewarded_ci
        spike_data.at[cluster,"speed_CI_try"] = try_ci
        spike_data.at[cluster,"speed_CI_run"] = run_ci
    return spike_data



def split_and_save_data(spike_data):
    for cluster in range(len(spike_data)):
        try_trials = np.array(spike_data.loc[cluster, 'try_trialid'])
        runthru_trials = np.array(spike_data.loc[cluster, 'run_through_trialid'])

        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        rewarded_rates = rates[np.isin(trials,try_trials)]
        rewarded_speed = speed[np.isin(trials,try_trials)]
        rewarded_position = position[np.isin(trials,try_trials)]
        reward_trials = trials[np.isin(trials,try_trials)]
        reward_types = types[np.isin(trials,try_trials)]
        failed_rates = rates[np.isin(trials,runthru_trials)]
        failed_speed = speed[np.isin(trials,runthru_trials)]
        failed_position = position[np.isin(trials,runthru_trials)]
        failed_trials = trials[np.isin(trials,runthru_trials)]
        failed_types = types[np.isin(trials,runthru_trials)]

        spike_data = drop_runthru_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types)

        data = np.vstack((rewarded_rates, rewarded_speed, rewarded_position, reward_trials, reward_types))
        data_try= data.transpose()
        data = np.vstack((failed_rates, failed_speed, failed_position, failed_trials, failed_types))
        data_run= data.transpose()
    return spike_data, data_try, data_run



def drop_runthru_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j):

    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_try_b'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_runthru_b'] = list(sn)
    return spike_data



def extract_time_binned_firing_rate_runthru(spike_data):
    spike_data["Rates_averaged_runthru_b"] = ""
    spike_data["Rates_sd_runthru_b"] = ""

    for cluster in range(len(spike_data)):
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[0])
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[1])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_runthru_b[4], dtype= np.int32)
        window = signal.gaussian(2, std=2)
        speed = signal.convolve(speed, window, mode='same')/ sum(window)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        if len(np.unique(trials)) > 1:
            # bin data over position bins
            bins = np.arange(0,200,1)
            trial_numbers = np.arange(min(trials),max(trials), 1)
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                if trial_data.shape[0] > 0:
                    t_rates = trial_data[:,0]
                    t_pos = trial_data[:,2]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            x = np.nanmean(data_b, axis=1)
            x_sd = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'Rates_averaged_runthru_b'] = list(x)# add data to dataframe
            spike_data.at[cluster, 'Rates_sd_runthru_b'] = list(x_sd)
        else:
            spike_data.at[cluster, 'Rates_averaged_runthru_b'] = np.nan
            spike_data.at[cluster, 'Rates_sd_runthru_b'] = np.nan
    return spike_data



def extract_time_binned_firing_rate_try(spike_data):
    spike_data["Rates_averaged_try_b"] = ""
    spike_data["Rates_sd_try_b"] = ""

    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[1])
        rates=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[0])
        position=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[2])
        trials=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[3], dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spikes_in_time_try_b[4], dtype= np.int32)
        window = signal.gaussian(2, std=2)
        speed = signal.convolve(speed, window, mode='same')/ sum(window)

        # stack data
        data = np.vstack((rates,speed,position,types, trials))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]

        if len(np.unique(trials)) > 1:
            # bin data over position bins
            bins = np.arange(0,200,1)
            trial_numbers = np.arange(min(trials),max(trials), 1)
            binned_data = np.zeros((bins.shape[0], trial_numbers.shape[0])); binned_data[:, :] = np.nan
            for tcount, trial in enumerate(trial_numbers):
                trial_data = data[data[:,4] == trial,:]
                if trial_data.shape[0] > 0:
                    t_rates = trial_data[:,0]
                    t_pos = trial_data[:,2]
                    for bcount, b in enumerate(bins):
                        rate_in_position = np.take(t_rates, np.where(np.logical_and(t_pos >= bcount, t_pos < bcount+1)))
                        average_rates = np.nanmean(rate_in_position)
                        binned_data[bcount, tcount] = average_rates

            #remove nans interpolate
            data_b = pd.DataFrame(binned_data[:,:], dtype=None, copy=False)
            data_b = data_b.dropna(axis = 1, how = "all")
            data_b.reset_index(drop=True, inplace=True)
            data_b = data_b.interpolate(method='linear', limit=None, limit_direction='both')
            data_b = np.asarray(data_b)
            x = np.reshape(data_b, (data_b.shape[0]*data_b.shape[1]))
            x = signal.convolve(x, window, mode='same')/ sum(window)
            data_b = np.reshape(x, (data_b.shape[0], data_b.shape[1]))
            x = np.nanmean(data_b, axis=1)
            x_sd = np.nanstd(data_b, axis=1)
            spike_data.at[cluster, 'Rates_averaged_try_b'] = list(x)# add data to dataframe
            spike_data.at[cluster, 'Rates_sd_try_b'] = list(x_sd)
        else:
            spike_data.at[cluster, 'Rates_averaged_try_b'] = np.nan
            spike_data.at[cluster, 'Rates_sd_try_b'] = np.nan
    return spike_data

