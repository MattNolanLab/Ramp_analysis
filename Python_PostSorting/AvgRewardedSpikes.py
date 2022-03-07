import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
import math
import matplotlib.pyplot as plt


def extract_smoothed_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0], 'trial_number' :  np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype=np.int16), 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings


def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==0)
    nbeaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==1)
    probe_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==2)
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def split_firing_data_by_reward(cluster_firings, rewarded_trials):
    rewarded_cluster_firings = cluster_firings.loc[cluster_firings['trial_number'].isin(rewarded_trials)]
    rewarded_cluster_firings.reset_index(drop=True, inplace=True)
    return rewarded_cluster_firings


def reshape_and_average_over_trials2(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings, max_trial_number):
    window = signal.gaussian(2, std=3)
    bin=199

    beaconed_spike_rate = signal.convolve(beaconed_cluster_firings, window, mode='same')/ sum(window)
    nonbeaconed_spike_rate = signal.convolve(nonbeaconed_cluster_firings, window, mode='same')/ sum(window)
    probe_spike_rate = signal.convolve(probe_cluster_firings, window, mode='same')/ sum(window)

    beaconed_reshaped_hist = np.reshape(beaconed_spike_rate, (int(beaconed_cluster_firings.size/bin),bin))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_spike_rate, (int(nonbeaconed_cluster_firings.size/bin), bin))
    probe_reshaped_hist = np.reshape(probe_spike_rate, (int(probe_cluster_firings.size/bin), bin))

    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)
    average_beaconed_sd = np.nanstd(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_sd = np.nanstd(nonbeaconed_reshaped_hist, axis=0)
    average_probe_sd = np.nanstd(probe_reshaped_hist, axis=0)

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16), average_beaconed_sd, average_nonbeaconed_sd, average_probe_sd



def reshape_and_average_over_trials(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings, max_trial_number):
    bin=199
    probe_cluster_firings[probe_cluster_firings==0] = np.nan
    beaconed_cluster_firings[beaconed_cluster_firings==0] = np.nan

    data_b = pd.DataFrame(beaconed_cluster_firings, dtype=None, copy=False)
    data_b = data_b.interpolate(method='pad')
    beaconed_cluster_firings = np.asarray(data_b)

    data_nb = pd.DataFrame(nonbeaconed_cluster_firings, dtype=None, copy=False)
    data_nb = data_nb.interpolate(method='pad')
    nonbeaconed_cluster_firings = np.asarray(data_nb)

    data_p = pd.DataFrame(probe_cluster_firings, dtype=None, copy=False)
    data_p = data_p.interpolate(method='pad')
    probe_cluster_firings = np.asarray(data_p)

    beaconed_reshaped_hist = np.reshape(beaconed_cluster_firings, (int(beaconed_cluster_firings.size/bin),bin))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_cluster_firings, (int(nonbeaconed_cluster_firings.size/bin), bin))
    probe_reshaped_hist = np.reshape(probe_cluster_firings, (int(probe_cluster_firings.size/bin), bin))

    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_beaconed_spike_rate = np.hstack((np.nan,np.nan,np.nan, average_beaconed_spike_rate[3:196],np.nan,np.nan,np.nan,)) ## get rid of edges but really this needs to be done properly
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.hstack((np.nan,np.nan,np.nan, average_nonbeaconed_spike_rate[3:196],np.nan,np.nan,np.nan,))
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)
    average_probe_spike_rate = np.hstack((np.nan,np.nan,np.nan, average_probe_spike_rate[3:196],np.nan,np.nan,np.nan,))
    average_beaconed_sd = np.nanstd(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_sd = np.nanstd(nonbeaconed_reshaped_hist, axis=0)
    average_probe_sd = np.nanstd(probe_reshaped_hist, axis=0)
    plt.plot(average_beaconed_spike_rate)
    plt.close()

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16), average_beaconed_sd, average_nonbeaconed_sd, average_probe_sd



def extract_smoothed_average_firing_rate_data(spike_data):
    spike_data["Rates_averaged_rewarded_b"] = ""
    spike_data["Rates_averaged_rewarded_nb"] = ""
    spike_data["Rates_averaged_rewarded_p"] = ""
    spike_data["Rates_sd_rewarded_b"] = ""
    spike_data["Rates_sd_rewarded_nb"] = ""
    spike_data["Rates_sd_rewarded_p"] = ""
    for cluster in range(len(spike_data)):
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        rewarded_trials = rewarded_trials[rewarded_trials >0]

        cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster)
        cluster_firings = split_firing_data_by_reward(cluster_firings, rewarded_trials)
        beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
        avg_b, avg_nb, avg_p, sd, s_nb, sd_p = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))

        spike_data.at[cluster, 'Rates_averaged_rewarded_b'] = list(avg_b)
        spike_data.at[cluster, 'Rates_averaged_rewarded_nb'] = list(avg_nb)
        spike_data.at[cluster, 'Rates_averaged_rewarded_p'] = list(avg_p)
        spike_data.at[cluster, 'Rates_sd_rewarded_b'] = list(sd)
        spike_data.at[cluster, 'Rates_sd_rewarded_nb'] = list(s_nb)
        spike_data.at[cluster, 'Rates_sd_rewarded_p'] = list(sd_p)

    return spike_data




def rewrite_smoothed_average_firing_rate_data(spike_data):
    spike_data["spike_rate_on_trials_smoothed"] = ""

    for cluster in range(len(spike_data)):
        rates = np.array(spike_data.at[cluster, 'Rates_bytrial'])

        spike_data.at[cluster, 'spike_rate_on_trials_smoothed'] = list(rates)


    spike_data

    return spike_data


