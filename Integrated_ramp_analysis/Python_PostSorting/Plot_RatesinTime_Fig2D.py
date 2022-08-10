import os
import matplotlib.pylab as plt
import numpy as np
import math
from astropy.convolution import convolve, Gaussian1DKernel
import pandas as pd
from scipy import stats
from scipy import signal



"""

## Instantaneous firing rate comparisons 

The following functions plots instantaneous firing rate aginast location and speed

Variables:
> distance
> speed
> firing rate

"""



def remove_low_speeds_and_segment_beaconed(rates, speed, acceleration, position, types ):
    data = np.vstack((rates, speed, acceleration, position, types))
    data=data.transpose()
    data_filtered = data[data[:,1] > 3,:]
    data_filtered = data_filtered[data_filtered[:,4] == 0,:]

    data_filtered = data_filtered[data_filtered[:,3] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,3] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,3] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,3] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    accel_outbound = data_outbound[:,2]
    position_outbound = data_outbound[:,3]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    accel_homebound = data_homebound[:,2]
    position_homebound = data_homebound[:,3]
    return rates_outbound , speed_outbound , position_outbound , rates_homebound, speed_homebound, position_homebound, accel_outbound, accel_homebound


def plot_color_coded_instant_rates_according_to_segment(recording_folder, spike_data, smoothen=True):
    gauss_kernel = Gaussian1DKernel(1)

    print('I am plotting instant rate against location ...')
    save_path = recording_folder + '/Figures/instant_firing_rates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        session_id = spike_data.session_id.values[cluster]

        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real

        if smoothen:
            rates = convolve(rates, gauss_kernel)
            speed = convolve(speed, gauss_kernel)

        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real
        acceleration = np.diff(np.array(speed)) # calculate acceleration
        acceleration = np.hstack((0, acceleration))

        # remove outliers
        rates =  pd.Series(rates)
        speed =  pd.Series(speed)
        position =  pd.Series(position)
        acceleration =  pd.Series(acceleration)
        types =  pd.Series(types)
        rates = rates[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        position = position[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        acceleration = acceleration[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        types = types[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers
        speed = speed[speed.between(speed.quantile(.05), speed.quantile(.95))] # without outliers

        rates_o, speed_o, position_o,rates_h, speed_h, position_h, accel_o, accel_h = remove_low_speeds_and_segment_beaconed(rates, speed, acceleration, position, types )
        max_fr_to_show = np.nanpercentile(rates_o, 99.9)

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(speed_o, rates_o, s=1, c=position_o, cmap='viridis')
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Speed (cm/s)', fontsize=16, labelpad = 10)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.locator_params(axis = 'x', nbins=3)
        #ax.set_xticklabels(['10', '30', '50'])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max_fr_to_show) # without outliers)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '_coded_outbound.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(position_o, rates_o, s=1, c=speed_o, cmap='BuPu_r')
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xticks([40,60,80])
        ax.set_xticklabels(['10', '30', '50'])
        ax.set_xlim(left=30, right=90)
        ax.set_ylim(bottom=0, top=max_fr_to_show)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_location' + '_coded_outbound.png', dpi=200)
        plt.close()

        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.scatter(accel_o, rates_o, s=1, c=position_o, cmap='viridis')
        cbar=plt.colorbar() #plt.cm.ScalarMappable(cmap='jet')
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Firing rate (Hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Acceleration $(cm/s)^2$', fontsize=16, labelpad = 10) # "meters $10^1$"
        ax.locator_params(axis = 'x', nbins=3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=True,
            labelleft=True,
            labelbottom=True,
            labelsize=16,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.locator_params(axis = 'x', nbins=3)
        ax.set_ylim(bottom=0, top=max_fr_to_show)
        ax.set_xlim(-10, 10)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_accel' + '_coded_outbound.png', dpi=200)
        plt.close()

    return spike_data

