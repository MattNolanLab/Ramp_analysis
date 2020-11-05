import Python_PostSorting.ExtractFiringData
import numpy as np


### -------------------------------------------------------------- ###

## Calculates location along the track in cm with maximum firing rate


def find_max_firing_loc(spike_data):
    print('calculating max firing location')
    spike_data["max_firing_location_outbound"] = ""
    spike_data["max_firing_location_homebound"] = ""

    for cluster in range(len(spike_data)):
        beaconed, nonbeaconed, probe = extract_firing_rate_data(spike_data, cluster)

        max_location_outbound=np.argmax(beaconed[20:90])
        max_location_outbound = max_location_outbound+20
        max_location_homebound=np.argmax(beaconed[110:180])
        max_location_homebound = max_location_homebound+110

        spike_data.at[cluster, "max_firing_location_outbound"] = max_location_outbound
        spike_data.at[cluster, "max_firing_location_homebound"] = max_location_homebound
    return spike_data




### -------------------------------------------------------------- ###

# calculates min and max firing rate for each cluster

def extract_firing_rate_data(spike_data, cluster):
    beaconed, nonbeaconed, probe, sd = Python_PostSorting.ExtractFiringData.extract_smoothed_average_firing_rate_data(spike_data, cluster)
    return beaconed, nonbeaconed, probe


def find_min_max_FR(spike_data):
    print('calculating min and max firing rate')
    spike_data["min_rate"] = ""
    spike_data["max_rate"] = ""
    for cluster in range(len(spike_data)):
        beaconed, nonbeaconed, probe = extract_firing_rate_data(spike_data, cluster)
        spike_data.at[cluster, "max_rate"] = np.max(beaconed[30:170])
        spike_data.at[cluster, "min_rate"] = np.min(beaconed[30:170])
    return spike_data




### -------------------------------------------------------------- ###



def find_min_firing_loc(spike_data):
    print('calculating min firing location')
    spike_data["min_firing_location_outbound"] = ""
    spike_data["min_firing_location_homebound"] = ""

    for cluster in range(len(spike_data)):
        beaconed, nonbeaconed, probe = extract_firing_rate_data(spike_data, cluster)

        max_location_outbound=np.argmin(beaconed[20:90])
        max_location_outbound = max_location_outbound+20
        max_location_homebound=np.argmin(beaconed[110:180])
        max_location_homebound = max_location_homebound+110

        spike_data.at[cluster, "min_firing_location_outbound"] = max_location_outbound
        spike_data.at[cluster, "min_firing_location_homebound"] = max_location_homebound
    return spike_data


def find_ramp_length(spike_data):
    print('calculating ramp length')
    spike_data["ramp_length_outbound"] = ""
    spike_data["ramp_length_homebound"] = ""

    for cluster in range(len(spike_data)):
        max = spike_data.loc[cluster, "max_firing_location_outbound"]
        min = spike_data.loc[cluster, "min_firing_location_outbound"]
        outbound = max-min
        max = spike_data.loc[cluster, "max_firing_location_homebound"]
        min = spike_data.loc[cluster, "min_firing_location_homebound"]
        homebound = max-min

        spike_data.at[cluster, "ramp_length_outbound"] = outbound
        spike_data.at[cluster, "ramp_length_homebound"] = homebound
    return spike_data


### -------------------------------------------------------------- ###

"""
### old code - needs segmenting

def find_max_firing_location(spike_data, cluster):
    beaconed, nonbeaconed, probe = extract_firing_rate_data(spike_data, cluster)

    max_location=np.argmax(beaconed[20:180])
    max_location = max_location+20

    return max_location


def find_fr_change(firing_rate, spike_data, cluster):
    rate_min = np.min(firing_rate[30:170])
    rate_max = np.max(firing_rate[30:170])
    change_in_fr = rate_max-rate_min
    spike_data.at[cluster, "change_in_fr"] = change_in_fr
    return change_in_fr


def find_above_error_increase(change_in_fr):
    above_error_increase = change_in_fr/20
    return above_error_increase


def find_rate_for_bin(firing_rate, b):
    bin_sum = np.nanmean(firing_rate[b:b+10])
    return bin_sum


def find_min_firing_location(firing_rate,max_location, above_error_increase):
    bins = np.arange(1,201,10)
    ramp_start=100
    if max_location >110: # if end of track firing cell
        for bin_count, b in enumerate(bins):
            bin_sum = find_rate_for_bin(firing_rate, b)
            next_bin_sum = find_rate_for_bin(firing_rate, b+10)
            the_next_bin_sum = find_rate_for_bin(firing_rate, b+20)
            if next_bin_sum - bin_sum > above_error_increase:
                if the_next_bin_sum - next_bin_sum > above_error_increase:
                    ramp_start = b
                    ramp_length =max_location-ramp_start

                    return ramp_length, ramp_start

    elif max_location < 90: # if start of track firing cell
        for bin_count, b in enumerate(reversed(bins)):
            bin_sum = find_rate_for_bin(firing_rate, b)
            next_bin_sum = find_rate_for_bin(firing_rate, b-10)
            the_next_bin_sum = find_rate_for_bin(firing_rate, b-20)
            if next_bin_sum - bin_sum > above_error_increase:
                if the_next_bin_sum - next_bin_sum > above_error_increase:
                    ramp_start = b
                    ramp_length =ramp_start-max_location
                    return ramp_length, ramp_start
    return ramp_start, 0


def calculate_ramp_length(firing_rate, spike_data, cluster):
    max_location = find_max_firing_location(spike_data, cluster)
    change_in_fr = find_fr_change(firing_rate, spike_data, cluster)
    above_error_increase = find_above_error_increase(change_in_fr)
    ramp_length, ramp_start = find_min_firing_location(firing_rate,max_location, above_error_increase)
    return ramp_length, ramp_start


def ramp_analysis(spike_data):
    print('calculating ramp length')
    spike_data["ramp_length"] = ""
    spike_data["min_firing_location"] = ""

    for cluster in range(len(spike_data)):
        beaconed, nonbeaconed, probe = extract_firing_rate_data(spike_data, cluster)
        ramp_length_b, ramp_start = calculate_ramp_length(beaconed, spike_data, cluster)

        spike_data.at[cluster, "ramp_length"] = ramp_length_b
        spike_data.at[cluster, "min_firing_location"] = ramp_start

    return spike_data


"""
