import Python_PostSorting.ExtractFiringData
import numpy as np
import pandas as pd
import matplotlib.pylab as plt



def add_columns_to_dataframe(spike_data):
    spike_data["monotonicity_increasing"] = ""
    spike_data["monotonicity_decreasing"] = ""
    return spike_data


def extract_time_binned_data(spike_data, cluster_index):
    rates_b, rates, rates, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster_index)
    rate = pd.Index(rates_b[20:90])
    return rate


def calculate_monotonicity_score(rate):
    try:
        is_increasing = rate.is_monotonic_increasing
        is_decreasing = rate.is_monotonic_decreasing
        print(is_increasing,is_decreasing)
        if is_decreasing == True:
            plt. plot(rate)
            plt.show()
            plt.close()
        if is_increasing == True:
            plt. plot(rate)
            plt.show()
            plt.close()
        plt. plot(rate)
        plt.show()
        plt.close()
    except TypeError:
        return False, False
    return is_increasing, is_decreasing


def calculate_monotonicity(spike_data):
    print('generating monotonicity score')
    spike_data=add_columns_to_dataframe(spike_data)

    for cluster in range(len(spike_data)):
        cluster_firings = extract_time_binned_data(spike_data, cluster)
        is_increasing, is_decreasing = calculate_monotonicity_score(cluster_firings)
        spike_data = add_data_to_dataframe(cluster, is_increasing, is_decreasing, spike_data)
    print(len(spike_data[spike_data.monotonicity_increasing == True]))
    print(len(spike_data[spike_data.monotonicity_decreasing == True]))
    return spike_data



def add_data_to_dataframe(cluster_index, is_increasing, is_decreasing, spike_data):
    spike_data.at[cluster_index, 'monotonicity_increasing'] = is_increasing
    spike_data.at[cluster_index, 'monotonicity_decreasing'] = is_decreasing
    return spike_data
