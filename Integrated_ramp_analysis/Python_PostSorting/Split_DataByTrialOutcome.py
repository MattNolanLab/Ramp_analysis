import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel


def extract_data_from_frame(spike_data, cluster):
    rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
    rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]

    rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0])
    speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1])
    position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2])
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4], dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3], dtype= np.int32)
    speed = convolve_speed_data(speed)

    data = np.vstack((rates, speed, position, trials, types))
    data=data.transpose()
    return rewarded_trials, data


def convolve_speed_data(speed):
    window = signal.gaussian(2, std=3)
    speed = signal.convolve(speed, window, mode='same')/ sum(window)
    return speed


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
    window = signal.gaussian(2, std=3)
    convolved_rate = signal.convolve(rate, window, mode='same')/ sum(window)
    return convolved_rate


def find_avg_speed_in_reward_zone(data):
    trialids = np.array(np.unique(np.array(data[:,3])))
    speeds = np.zeros((trialids.shape[0]))
    for trialcount, trial in enumerate(trialids):
        trial_data = data[data[:,3] == trial,:] # get data only for each trial

        time_binned_position = trial_data[:,2]
        time_binned_speed = trial_data[:,1]

        if len(time_binned_position)>2:
            f = interp1d(time_binned_position, time_binned_speed)
            interpolated_time_binned_position = np.linspace(min(time_binned_position), max(time_binned_position), num=200, endpoint=True)
            interpolated_time_binned_speed = f(interpolated_time_binned_position)
            rz_speeds = interpolated_time_binned_speed[(interpolated_time_binned_position>=90) & (interpolated_time_binned_position<=110)]
            speeds[trialcount] = np.nanmean(rz_speeds)
        else:
            speeds[trialcount] = np.nan

    return speeds, trialids


def find_confidence_interval(speeds):
    mean, sigma = np.nanmean(speeds), np.nanstd(speeds)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]
    return upper, lower


def catagorise_failed_trials(speed_array_failed, trialids_failed, upper):
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


def split_time_data_by_trial_outcome(spike_data):
    print("Splitting data based on trial outcome...")

    spike_data["run_through_trialid"] = ""
    spike_data["try_trialid"] = ""
    spike_data["spikes_in_time_reward"] = ""
    spike_data["spikes_in_time_try"] = ""
    spike_data["spikes_in_time_run"] = ""


    for cluster in range(len(spike_data)):
        rewarded_trials, data = extract_data_from_frame(spike_data, cluster)  #load all data
        hit_data, miss_data = split_trials(data, rewarded_trials)   #split on hit/mmiss

        speeds_rewarded, trialids_rewarded = find_avg_speed_in_reward_zone(hit_data)
        speed_array_failed, trialids_failed = find_avg_speed_in_reward_zone(miss_data)
        upper, lower = find_confidence_interval(speeds_rewarded)

        trial_id_try, trial_id_run = catagorise_failed_trials(speed_array_failed, trialids_failed, upper)

        spike_data.at[cluster,"run_through_trialid"] = list(trial_id_run)
        spike_data.at[cluster,"try_trialid"] = list(trial_id_try)

    spike_data = split_and_save_data_with_all_speeds(spike_data)

    return spike_data





def split_and_save_data_with_all_speeds(spike_data):
    for cluster in range(len(spike_data)):
        try_trials = np.array(spike_data.loc[cluster, 'try_trialid'])
        runthru_trials = np.array(spike_data.loc[cluster, 'run_through_trialid'])
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])

        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real, dtype=np.float32)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real, dtype=np.float32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)

        speed = convolve_speed_data(speed)

        #speed filter first
        data = np.vstack((rates,speed,position, trials, types))
        data=data.transpose()
        data = data[data[:,1] >= 3,:]
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
        failed_rates = rates[np.isin(trials,runthru_trials)]
        failed_speed = speed[np.isin(trials,runthru_trials)]
        failed_position = position[np.isin(trials,runthru_trials)]
        failed_trials = trials[np.isin(trials,runthru_trials)]
        failed_types = types[np.isin(trials,runthru_trials)]
        try_rates = rates[np.isin(trials,try_trials)]
        try_speed = speed[np.isin(trials,try_trials)]
        try_position = position[np.isin(trials,try_trials)]
        trying_trials = trials[np.isin(trials,try_trials)]
        try_types = types[np.isin(trials,try_trials)]

        spike_data = drop_data_into_frame(spike_data, cluster, rewarded_rates, rewarded_speed , rewarded_position, reward_trials, reward_types, failed_rates, failed_speed, failed_position, failed_trials , failed_types, try_rates, try_speed, try_position, trying_trials , try_types)
    return spike_data


def drop_data_into_frame(spike_data, cluster_index, a,b, c, d, e, f,  g, h, i, j, k, l, n, m, o):

    sn=[]
    sn.append(a) # rate
    sn.append(b) # speed
    sn.append(c) # position
    sn.append(d) # trials
    sn.append(e) # trials
    spike_data.at[cluster_index, 'spikes_in_time_reward'] = list(sn)

    sn=[]
    sn.append(f) # rate
    sn.append(g) # speed
    sn.append(h) # position
    sn.append(i) # trials
    sn.append(j) # trials
    spike_data.at[cluster_index, 'spikes_in_time_run'] = list(sn)

    sn=[]
    sn.append(k) # rate
    sn.append(l) # speed
    sn.append(n) # position
    sn.append(m) # trials
    sn.append(o) # trials
    spike_data.at[cluster_index, 'spikes_in_time_try'] = list(sn)

    return spike_data


###------------------------------------------------------------------------------------------------------------------###


###------------------------------------------------------------------------------------------------------------------###


def calculate_rate_map_by_hit_try_run(cluster_row, tt, htr, smoothed):
    bin=200

    # subset first by trial type
    rates_in_space = cluster_row["spike_rate_on_trials_smoothed"].iloc[0][0]
    tn_in_space = cluster_row["spike_rate_on_trials_smoothed"].iloc[0][1]
    tt_in_space = cluster_row["spike_rate_on_trials_smoothed"].iloc[0][2]

    if smoothed:
        gauss_kernel = Gaussian1DKernel(2)
        rates_in_space = convolve(rates_in_space, gauss_kernel)

    rates_in_space = rates_in_space[tt_in_space == tt]
    tn_in_space = tn_in_space[tt_in_space == tt]

    # then get the trial numbers to use to extract the rates by hit/try/run
    if htr == "Hit":
        column = "rewarded_trials"
    elif htr == "Try":
        column = "try_trialid"
    elif htr == "SlowTry":
        column = "try_trialid_slow"
    elif htr == "FastTry":
        column = "try_trialid_fast"
    elif htr == "Run":
        column = "run_through_trialid"
    htr_trial_numbers = cluster_row[column].iloc[0]
    htr_mask = np.isin(tn_in_space, htr_trial_numbers)
    rates_in_space = rates_in_space[htr_mask]

    # reshape rates_in_space and compute a trial_averaged_rate_map
    trial_rate_map = np.reshape(rates_in_space, (int(len(rates_in_space)/bin), bin))
    rate_map = np.nanmean(trial_rate_map, axis=0)
    sem_rate_map = stats.sem(trial_rate_map, axis=0, nan_policy="omit")
    if len(rate_map[np.isnan(rate_map)]) > 0:
        return np.nan, np.nan, np.nan
    else:
        return trial_rate_map.tolist(), rate_map.tolist(), sem_rate_map.tolist()


### Calculate average firing rate for trial outcomes
def extract_firing_rate_map_by_hit_try_run(spike_data):

    new_spike_data =pd.DataFrame()
    for index, cluster_row in spike_data.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)

        # beaconed and non beaconed
        for tt, tt_suffix in zip([0, 1], ["", "_nb"]):
            for htr in ["Hit", "Try", "Run"]:
                for smoothed, smooth_suffix in zip([True, False], ["_smoothed", ""]):
                    trial_rate_map, rate_map, sem_rate_map = calculate_rate_map_by_hit_try_run(cluster_row, tt=tt, htr=htr, smoothed=smoothed)
                    cluster_row["Avg_FiringRate_"+htr+"Trials"+tt_suffix+smooth_suffix] = [rate_map]
                    cluster_row["SD_FiringRate_"+htr+"Trials"+tt_suffix+smooth_suffix] = [sem_rate_map]
                    cluster_row["FiringRate_"+htr+"Trials_trials"+tt_suffix+smooth_suffix] = [trial_rate_map]


        new_spike_data = pd.concat([new_spike_data, cluster_row], ignore_index=True)

    return new_spike_data



