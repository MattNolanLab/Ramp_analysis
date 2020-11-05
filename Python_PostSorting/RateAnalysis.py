
import os
import numpy as np





def remove_low_speeds(rates, speed, position ):
    data = np.vstack((rates, speed, position))
    data=data.transpose()
    data_filtered = data[data[:,1] > 2.4,:]
    rates = data_filtered[:,0]
    speed = data_filtered[:,1]
    position = data_filtered[:,2]
    return rates, speed, position




def calculate_track_rate(rates, speed, position ):
    data = np.vstack((rates, speed, position))
    data=data.transpose()
    data_filtered = data[data[:,1] > 2,:]

    #track data
    data_track = data_filtered[data_filtered[:,2] > 30,:]
    data_track = data_track[data_track[:,2] < 170,:]

    #blackbox data
    data_bb1 = data_filtered[data_filtered[:,2] < 30,:]
    data_bb2 = data_filtered[data_filtered[:,2] > 170,:]


    rates_track = np.nanmean(data_track[:,0])
    rates_bb1 = np.nanmean(data_bb1[:,0])
    rates_bb2 = np.nanmean(data_bb2[:,0])
    rates_bb = (rates_bb1+rates_bb2)/2

    return rates_track , rates_bb



def extract_firing_rate(spike_data):
    print('I am calculating track firing rate...')
    spike_data["track_rates"] = ""
    spike_data["bb_rates"] = ""

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        rates =  np.array(spike_data.loc[cluster, 'spike_rate_in_time'], dtype=np.float)
        speed = np.array(spike_data.loc[cluster, 'speed_rate_in_time'],dtype=np.float)
        position = np.array(spike_data.loc[cluster, 'position_rate_in_time'],dtype=np.float)

        rates_track , rates_bb = calculate_track_rate(rates, speed, position )

        spike_data.at[cluster, "track_rates"] = rates_track
        spike_data.at[cluster, "bb_rates"] = rates_bb
    return spike_data

