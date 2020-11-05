import elephant as elephant
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms
import numpy as np
import matplotlib.pylab as plt
from elephant.statistics import isi, cv
import os
import Python_PostSorting.plot_utility
import scipy.stats


"""

## Here we convolve spikes in time trial by trial basis on clusters

1. load data:
    data is firing times in ms
2. bin spikes into 250 ms windows

"""


def generate_time_bins(spike_times):
    time_bins = np.arange(0,np.max(spike_times),7500)
    number_of_bins = time_bins.shape[0]
    return time_bins


def bin_spike_times(spike_times, number_of_bins):
    binned_spikes_in_time = create_histogram(spike_times, number_of_bins)
    return binned_spikes_in_time


def create_histogram(spike_times, number_of_bins):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=max(number_of_bins)+1)
    values = np.array([[posrange[0], posrange[-1]]])
    H, bins = np.histogram(spike_times, bins=(posrange), range=values)
    return H


def convolve_binned_spikes(binned_spike_times):
    convolved_spikes=[]
    convolved_spikes = elephant.statistics.fftkernel(binned_spike_times, 2)
    return convolved_spikes


def convolve_spikes_in_time(spike_data):
    print('I am convolving spikes in time...')
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        spike_times = np.array(spike_data.at[cluster, "firing_times"])
        number_of_bins = generate_time_bins(spike_times)
        binned_spike_times = bin_spike_times(spike_times, number_of_bins)
        convolved_spikes = convolve_binned_spikes(binned_spike_times)
    return spike_data


def convolve_speed_in_time(spike_data):
    print('I am convolving spikes in time...')
    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        spike_times = np.array(spike_data.at[cluster, "speed_per200ms"])
        number_of_bins = generate_time_bins(spike_times)
        binned_speed = bin_spike_times(spike_times, number_of_bins)
        convolved_speed = convolve_binned_spikes(binned_speed)

    return spike_data

