import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
import math
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel

def extract_firing_rate_data(spike_data, cluster_index, smoothen):
    firing_rates = np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0])
    if smoothen:
        gauss_kernel = Gaussian1DKernel(2)
        firing_rates = convolve(firing_rates, gauss_kernel)
    trial_numbers = np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype=np.int16)
    trial_types = np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2])

    cluster_firings = pd.DataFrame({'firing_rate' : firing_rates,
                                    'trial_number': trial_numbers,
                                    'trial_type'  : trial_types})
    return cluster_firings


def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==0]
    nbeaconed_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==1]
    probe_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==2]
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def split_firing_data_by_reward(cluster_firings, rewarded_trials):
    rewarded_cluster_firings = cluster_firings.loc[cluster_firings['trial_number'].isin(rewarded_trials)]
    rewarded_cluster_firings.reset_index(drop=True, inplace=True)
    return rewarded_cluster_firings


def reshape_and_average_over_trials(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings):
    bin=200
    data_b = pd.DataFrame(beaconed_cluster_firings, dtype=None, copy=False)
    beaconed_cluster_firings = np.asarray(data_b)

    data_nb = pd.DataFrame(nonbeaconed_cluster_firings, dtype=None, copy=False)
    nonbeaconed_cluster_firings = np.asarray(data_nb)

    data_p = pd.DataFrame(probe_cluster_firings, dtype=None, copy=False)
    probe_cluster_firings = np.asarray(data_p)

    beaconed_reshaped_hist = np.reshape(beaconed_cluster_firings, (int(beaconed_cluster_firings.size/bin),bin))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_cluster_firings, (int(nonbeaconed_cluster_firings.size/bin), bin))
    probe_reshaped_hist = np.reshape(probe_cluster_firings, (int(probe_cluster_firings.size/bin), bin))

    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)

    average_beaconed_sd = stats.sem(beaconed_reshaped_hist, axis=0, nan_policy="omit")
    average_nonbeaconed_sd = stats.sem(nonbeaconed_reshaped_hist, axis=0, nan_policy="omit")
    average_probe_sd = stats.sem(probe_reshaped_hist, axis=0, nan_policy="omit")
    plt.plot(average_beaconed_spike_rate)
    plt.close()

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16), average_beaconed_sd, average_nonbeaconed_sd, average_probe_sd



def extract_smoothed_average_firing_rate_data(spike_data, rewarded=True, smoothen=True):
    suffix = ""
    if rewarded:
        suffix += "_rewarded"
    if smoothen:
        suffix += "_smoothed"

    spike_data["Rates_averaged"+suffix+"_b"] = ""
    spike_data["Rates_averaged"+suffix+"_nb"] = ""
    spike_data["Rates_averaged"+suffix+"_p"] = ""
    spike_data["Rates_sd"+suffix+"_b"] = ""
    spike_data["Rates_sd"+suffix+"_nb"] = ""
    spike_data["Rates_sd"+suffix+"_p"] = ""
    for cluster in range(len(spike_data)):
        rewarded_trials = np.array(spike_data.at[cluster, 'rewarded_trials'], dtype=np.int16)
        rewarded_trials = rewarded_trials[rewarded_trials >0]
        cluster_firings = extract_firing_rate_data(spike_data, cluster, smoothen=smoothen)
        if rewarded:
            cluster_firings = split_firing_data_by_reward(cluster_firings, rewarded_trials)
        beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
        avg_b, avg_nb, avg_p, sd, s_nb, sd_p = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]))

        spike_data.at[cluster, 'Rates_averaged'+suffix+'_b'] = list(avg_b)
        spike_data.at[cluster, 'Rates_averaged'+suffix+'_nb'] = list(avg_nb)
        spike_data.at[cluster, 'Rates_averaged'+suffix+'_p'] = list(avg_p)
        spike_data.at[cluster, 'Rates_sd'+suffix+'_b'] = list(sd)
        spike_data.at[cluster, 'Rates_sd'+suffix+'_nb'] = list(s_nb)
        spike_data.at[cluster, 'Rates_sd'+suffix+'_p'] = list(sd_p)

    return spike_data



