import os
import matplotlib.pylab as plt
import numpy as np


"""

## cluster spatial firing properties
The following functions make plots of cluster spatial firing properties:

-> spike location versus trial
-> average firing rate versus location
-> smoothed firing rate plots
-> spike number versus location

"""



def remake_trial_numbers(rewarded_beaconed_trial_numbers):
    unique_trials = np.unique(rewarded_beaconed_trial_numbers)
    new_trial_numbers = []
    trial_n = 1
    for trial in unique_trials:
        trial_data = rewarded_beaconed_trial_numbers[rewarded_beaconed_trial_numbers == trial]# get data only for each tria
        num_stops_per_trial = len(trial_data)
        new_trials = np.repeat(trial_n, num_stops_per_trial)
        new_trial_numbers = np.append(new_trial_numbers, new_trials)
        trial_n +=1
    return new_trial_numbers, unique_trials



### This plots raw data binned in time - a few trials as example data


def plot_tiny_raw(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #load raw data binned in time
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        acceleration = np.diff(np.array(speed)) # calculate acceleration
        acceleration = np.hstack((0, acceleration))
        ## stack and segment a few trials

        data = np.vstack((rates,acceleration, speed, position, trials))
        data = np.transpose(data)
        trial_data = data[350:750, :]

        # plot raw position
        avg_spikes_on_track = plt.figure(figsize=(10,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(4, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,3], '-', color='Red', linewidth = 2)
        #ax.plot(data_bb[:,3], '-', color='red', linewidth = 2)
        #ax.plot(data_bb2[:,3], '-', color='red', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
        ax.set_xticklabels(['', '', ''])
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

        # plot raw speed
        ax = avg_spikes_on_track.add_subplot(4, 1, 2)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,2], '-', color='Gold', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Speed (cm/s)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        ax.set_ylim(0, 60)
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
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)


        # plot raw acceleration
        ax = avg_spikes_on_track.add_subplot(4, 1, 3)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,1], '-', color='Blue', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Acceleration (cm/s)2', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
        ax.set_ylim(-15, 15)
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
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        #ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)

        # plot raw spikes
        ax = avg_spikes_on_track.add_subplot(4, 1, 4)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
        #ax.plot(data_bb[:,0], '-', color='Black', linewidth = 2)
        #ax.plot(data_bb2[:,0], '-', color='Black', linewidth = 2)
        ax.locator_params(axis = 'x', nbins=3)
        #plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
        #plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
            labelsize=18,
            length=5,
            width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all.png', dpi=200)
        plt.close()
    return spike_data




