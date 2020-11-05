import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Python_PostSorting.interspike_interval import *

def create_average_isi_plot(spiking_data, number_of_bins, track_length):

    '''
    This functions averages interspike intervals over trials and plots the average interspike interval
    against track position for every cluster ID in dataframe for a given track length and number of location bins

    :param spike_data: pandas dataframe with "inter_spike_interval" collumn
    :param number_of_bins: number of bins to use for histogram
    :param track_length; track length
    '''

    bins_edges = np.arange(0, track_length, track_length/number_of_bins)

    for cluster_id in spiking_data["cluster_id"]:

        location_bins = x = [[] for i in range(number_of_bins)]

        cluster_data_trials = np.asarray(spiking_data.loc[spiking_data.loc[:, "cluster_id"] == cluster_id, :]["trial_number"].values[0])
        cluster_data_x_pos =  np.asarray(spiking_data.loc[spiking_data.loc[:, "cluster_id"] == cluster_id, :]["x_position_cm"].values[0])
        cluster_data_isi =    np.asarray(spiking_data.loc[spiking_data.loc[:, "cluster_id"] == cluster_id, :]["inter_spike_interval"].values[0])
        trial_numbers = np.unique(cluster_data_trials)


        for trial_number in trial_numbers: # selects from list of unique trial numbers

            trial_indices = np.asarray(np.where(cluster_data_trials == trial_number)[0]) # indices for trial numbers specific to trial in this loop
            trial_interspike_intervals = cluster_data_isi[trial_indices]
            trial_x_position_cm = cluster_data_x_pos[trial_indices]

            # loop over spike locations
            for pos_idx, pos_x_cm in np.ndenumerate(trial_x_position_cm):

                bin_idx = np.digitize(pos_x_cm, bins_edges)-1

                if not np.isnan(trial_interspike_intervals[pos_idx]):
                    location_bins[bin_idx].append(trial_interspike_intervals[pos_idx])#

        average_isi_bins = np.zeros(len(location_bins))

        for i in range(len(location_bins)):
            if len(location_bins[i]) is not 0:  # doesnt calculate average for those without entries
                average_isi_bins[i] = np.mean(location_bins[i])

        # now plot for all clusters
        ax = plt.subplot(111)
        ax.bar(bins_edges[0:number_of_bins], average_isi_bins, width=1, color='k')
        ax.set_xlabel('Track Position (cm)')
        ax.set_ylabel('Average Interspike Interval (cm)')
        ax.set_title("title here")
        #f.savefig(save_path + title)

        plt.xlim([0, bins_edges.size])
        plt.show()


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #test_create_isi_histogram()

    example_spike_data = pd.read_pickle('/home/harry/Downloads/spatial_firing.pkl')
    example_spike_data = calculate_spike_interval(example_spike_data)

    create_average_isi_plot(example_spike_data, 200, 200)

if __name__ == '__main__':
    main()