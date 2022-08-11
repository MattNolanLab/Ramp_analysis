import os
import matplotlib.pylab as plt
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility
import numpy as np
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour


"""

## cluster spatial firing properties
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate plots

"""


def plot_firing_rate_maps_for_trials(recording_folder, spike_data, rewarded, smoothen):
    suffix=""
    if rewarded:
        suffix += "_rewarded"
    if smoothen:
        suffix += "_smoothed"

    print('I am plotting firing rate maps for trials...')

    save_path = recording_folder + '/Figures/average_firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        avg_beaconed_spike_rate=np.array(spike_data.loc[cluster, 'Rates_averaged'+suffix+'_b'])
        average_beaconed_sd=np.array(spike_data.loc[cluster, 'Rates_sd'+suffix+'_b'])

        avg_nonbeaconed_spike_rate=np.array(spike_data.loc[cluster, 'Rates_averaged'+suffix+'_nb'])
        average_nonbeaconed_sd=np.array(spike_data.loc[cluster, 'Rates_sd'+suffix+'_nb'])

        avg_probe_spike_rate=np.array(spike_data.loc[cluster, 'Rates_averaged'+suffix+'_p'])
        average_probe_sd=np.array(spike_data.loc[cluster, 'Rates_sd'+suffix+'_p'])

        bins=np.arange(0.5,199.5+1,1)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) +suffix+'.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_probe_spike_rate, '-', color=(31/255, 181/255, 178/255), alpha=0.7)
        ax.fill_between(bins, avg_probe_spike_rate-average_probe_sd,avg_probe_spike_rate+average_probe_sd, facecolor = (31/255, 181/255, 178/255), alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) +suffix+'_p.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, avg_nonbeaconed_spike_rate, '-', color='Blue', alpha=0.7)
        ax.fill_between(bins, avg_nonbeaconed_spike_rate-average_nonbeaconed_sd,avg_nonbeaconed_spike_rate+average_nonbeaconed_sd, facecolor = 'Blue', alpha = 0.2)
        ax.plot(bins, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bins, avg_beaconed_spike_rate-average_beaconed_sd,avg_beaconed_spike_rate+average_beaconed_sd, facecolor = 'Black', alpha = 0.2)
        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_ylim(0)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) +suffix+'_nb.png', dpi=200)
        plt.close()





