import os
import matplotlib.pylab as plt
import numpy as np
import cmath
import pandas as pd
from scipy import signal



### --------------------------------------------------------------------------------------------------- ###

### CALCULATE ACCELERATION OVER TIME



def generate_acceleration(spike_data, recording_folder):
    print('I am calculating acceleration...')
    spike_data["spikes_in_time_all"] = ""
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        trials =  np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real)
        types =  np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real)
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)

        # filter data
        try:
            window = signal.gaussian(2, std=3)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except (ValueError, TypeError):
                continue

        # remove outliers
        rates_o =  pd.Series(rates)
        speed_o =  pd.Series(speed)
        position_o =  pd.Series(position)
        trials_o =  pd.Series(trials)
        types_o =  pd.Series(types)

        rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        trials = trials_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        types = types_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers


        if speed.size > 1:
            acceleration = np.diff(np.array(speed))
            acceleration = np.hstack((0, acceleration))
            #plot_acceleration(recording_folder, spike_data, cluster_index, speed, acceleration)
            #plot_instant_acceleration(recording_folder, spike_data, cluster_index, rates, position, speed, acceleration)
            #plot_instant_acceleration_by_segment(recording_folder, spike_data, cluster, cluster_index, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration))
        else:
            acceleration = np.zeros((speed.size))
        spike_data = store_acceleration(spike_data, cluster, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration), np.asarray(trials), np.asarray(types))
    return spike_data


def plot_acceleration(recording_folder, spike_data, cluster, speed, acceleration):
    plt.plot(speed[:1000])
    plt.plot(acceleration[:1000])
    save_path = recording_folder + 'Figures/Acceleration'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_accell_map_Cluster_' + str(cluster +1) + '_location' + '.png', dpi=200)
    plt.close()
    return



def generate_acceleration_rewarded_trials(spike_data, recording_folder):
    print('I am calculating acceleration...')
    spike_data["spikes_in_time"] = ""
    for cluster in range(len(spike_data)):
        session_id = spike_data.at[cluster, "session_id"]
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates =  np.array(spike_data.iloc[cluster].spikes_in_time_reward[0].real)
        speed = np.array(spike_data.iloc[cluster].spikes_in_time_reward[1].real)
        trials =  np.array(spike_data.iloc[cluster].spikes_in_time_reward[3].real)
        types =  np.array(spike_data.iloc[cluster].spikes_in_time_reward[4].real)
        position = np.array(spike_data.iloc[cluster].spikes_in_time_reward[2].real)

        # filter data
        try:
            window = signal.gaussian(2, std=3)
            speed = signal.convolve(speed, window, mode='same')/sum(window)
            rates = signal.convolve(rates, window, mode='same')/sum(window)
        except (ValueError, TypeError):
                continue


        """
        # remove outliers
        rates_o =  pd.Series(rates)
        speed_o =  pd.Series(speed)
        position_o =  pd.Series(position)
        trials_o =  pd.Series(trials)
        types_o =  pd.Series(types)

        rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        trials = trials_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        types = types_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))] # without outliers
        """



        if speed.size > 1:
            acceleration = np.diff(np.array(speed))
            acceleration = np.hstack((0, acceleration))

            # remove acceleration outliers
            mean_speed = np.nanmean(acceleration)
            sd_speed = np.nanstd(acceleration)
            upper_speed_sd = mean_speed+(sd_speed*3)

            data = np.vstack((rates, speed, acceleration, position, trials, types))
            data=data.transpose()
            data = data[data[:,2] < upper_speed_sd,:]
            rates = data[:,0]
            speed = data[:,1]
            acceleration = data[:,2]
            position = data[:,3]
            trials = data[:,4]
            types = data[:,5]
        else:
            acceleration = np.zeros((speed.size))
        spike_data = store_acceleration_for_rewaded(spike_data, cluster, np.asarray(rates), np.asarray(position), np.asarray(speed), np.asarray(acceleration), np.asarray(trials), np.asarray(types))
        plot_instant_acceleration_by_segment(recording_folder, spike_data, cluster, cluster_index, rates, position, speed, acceleration)
    return spike_data





def plot_instant_acceleration(recording_folder, spike_data, cluster, rates, position, speed, acceleration):
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(acceleration, rates, 'o', color='Black', markersize=1.5)
    plt.ylabel('Spike rate (hz)', fontsize=10, labelpad = 10)
    plt.xlabel('Speed (cm/s)', fontsize=10, labelpad = 10)
    ax.locator_params(axis = 'x', nbins=3)
    plt.locator_params(axis = 'y', nbins  = 4)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster +1) + '_acceleration' + '.png', dpi=200)
    plt.close()
    return


def remove_low_speeds_and_segment(rates, speed, position, acceleration ):
    data = np.vstack((rates, speed, position, acceleration))
    data=data.transpose()
    data_filtered = data[data[:,1] >= 2,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]
    data_homebound = data_filtered[data_filtered[:,2] >= 110,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]
    acceleration_outbound = data_outbound[:,3]

    rates_homebound = data_homebound[:,0]
    speed_homebound = data_homebound[:,1]
    position_homebound = data_homebound[:,2]
    acceleration_homebound = data_homebound[:,3]

    return rates_outbound , speed_outbound , position_outbound , acceleration_outbound, rates_homebound, speed_homebound, position_homebound, acceleration_homebound


def plot_instant_acceleration_by_segment(recording_folder, spike_data, cluster, cluster_index, rates, position, speed, acceleration):
    save_path = recording_folder + '/Figures/InstantRates'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    rates_o, speed_o, position_o, acceleration_o, rates_h, speed_h, position_h, acceleration_h = remove_low_speeds_and_segment(rates, speed, position, acceleration )

    rates =  pd.Series(rates_o)
    acceleration =  pd.Series(acceleration_o)
    speed =  pd.Series(speed_o)
    position =  pd.Series(position_o)
    rates_o = rates[speed.between(speed.quantile(.03), speed.quantile(.97))] # without outliers
    acceleration_o = acceleration[speed.between(speed.quantile(.03), speed.quantile(.97))] # without outliers
    position_o = position[speed.between(speed.quantile(.03), speed.quantile(.97))] # without outliers
    speed_o = speed[speed.between(speed.quantile(.03), speed.quantile(.97))] # without outliers

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(acceleration_o, rates_o, s=area, c=position_o)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Firing rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Acceleration $(cm/s^2)$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-20, 25)
    #ax.set_ylim(0)
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
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_acceleration' + '_coded_outbound.png', dpi=200)
    plt.close()

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(acceleration_h, rates_h, s=area, c=position_h)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Firing rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Acceleration $(cm/s)^2$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-100, 150)
    ax.set_ylim(0)
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
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_acceleration' + '_coded_homebound.png', dpi=200)
    plt.close()

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(speed_o, rates_o, s=area, c=position_o)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Firing rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Speed $(cm/s)$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-20, 25)
    #ax.set_ylim(0)
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
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_speed' + '_coded_outbound.png', dpi=200)
    plt.close()

    avg_spikes_on_track = plt.figure(figsize=(4,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    area = np.pi*1
    plt.scatter(position_o, rates_o, s=area, c=position_o)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.ylabel('Firing rate (hz)', fontsize=16, labelpad = 10)
    plt.xlabel('Location $(cm)$', fontsize=16, labelpad = 10) # "meters $10^1$"
    #ax.set_xlim(-20, 25)
    #ax.set_ylim(0)
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
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_position' + '_coded_outbound.png', dpi=200)
    plt.close()

    return


def store_acceleration(spike_data,cluster_index, rates, position, speed, acceleration,  trials, types):
    sn=[]
    sn.append(rates) # rate
    sn.append(position) # position
    sn.append(speed) # speed
    sn.append(acceleration) # acceleration
    sn.append(trials) # trials
    sn.append(types) # types
    spike_data.at[cluster_index, 'spikes_in_time_all'] = list(sn)
    return spike_data



def store_acceleration_for_rewaded(spike_data,cluster_index, rates, position, speed, acceleration,  trials, types):
    sn=[]
    sn.append(rates) # rate
    sn.append(position) # position
    sn.append(speed) # speed
    sn.append(acceleration) # acceleration
    sn.append(trials) # trials
    sn.append(types) # types
    spike_data.at[cluster_index, 'spikes_in_time'] = list(sn)
    return spike_data
