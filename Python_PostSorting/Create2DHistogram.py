import numpy as np


def create_2dhistogram(trials, locations, number_of_bins, trialrange):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=max(number_of_bins)+1)
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(trials, locations, bins=(trialrange, posrange), range=values)
    return H


def create_histogram(spike_times, number_of_bins):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=max(number_of_bins)+1)
    values = np.array([[posrange[0], posrange[-1]]])
    H, bins = np.histogram(spike_times, bins=(posrange), range=values)
    return H
