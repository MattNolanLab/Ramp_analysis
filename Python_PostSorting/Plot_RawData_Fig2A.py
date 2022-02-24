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


def plot_raw_orig(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data_final'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #load raw data binned in time
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)

        trials, unique_trials = remake_trial_numbers(trials)
        ## stack and segment a few trials

        data = np.vstack((rates, position, trials, speed))
        data = np.transpose(data)

        """
        try:
            trial_data = data[data[:,2] > 26,:]
            trial_data = trial_data[trial_data[:,2] < 44,:]
        except (IndexError, ValueError):
            trial_data = data[data[:,2] > 0,:]
            trial_data = trial_data[trial_data[:,2] < 15,:]
        """

        #trial_data = data[data[:,2] > 10,:]
        #trial_data = trial_data[trial_data[:,2] < 27,:]
        #trial_data = data[6500:8500,:]
        trial_data = data[350:750, :]

        trials_points = np.diff(np.array(trial_data[:,2]))
        trial_y_coords = np.where(trials_points >0 )

        slow_data = np.where(trial_data[:,3] < 3)
        reward_data = np.where(trial_data[:,1] == 100)

        # plot raw position
        avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
        ymax = np.nanmax(trial_data[:,1])
        ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
        ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
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


        # plot raw spikes
        ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
        ymax = np.nanmax(trial_data[:,0])
        ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
        ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 2, alpha=0.2)
        ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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


        try:
            trial_data = data[750:1150, :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points >0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
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


            # plot raw spikes
            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 2, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
            plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_2.png', dpi=200)
            plt.close()
        except (ValueError, IndexError):
            print("meh")

        try:
            trial_data = data[1150:1550, :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points >0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
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
            # plot raw spikes
            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 2, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
            plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_3.png', dpi=200)
            plt.close()
        except (ValueError, IndexError):
            print("meh")


        try:
            trial_data = data[1550:1950, :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points >0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
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
            # plot raw spikes
            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 2, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
            plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_4.png', dpi=200)
            plt.close()
        except (ValueError, IndexError):
            print("meh")

        try:
            trial_data = data[1950:2350, :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points >0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
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
            # plot raw spikes
            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 2, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
            plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
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
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_5.png', dpi=200)
            plt.close()
        except (ValueError, IndexError):
            print("meh")
    return spike_data




def plot_raw(recording_folder, spike_data):
    print('I am plotting a few trials of raw data...')
    save_path = recording_folder + '/Figures/raw_data_final'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        #load raw data binned in time
        rates=np.array(spike_data.iloc[cluster].spike_rate_in_time[0].real)*10
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time[2].real)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
        rewarded_trials = np.array(spike_data.loc[cluster, 'rewarded_trials'])
        rewarded_trials = rewarded_trials[~np.isnan(rewarded_trials)]
        rewarded_trials = np.where(trials[np.isin(trials,rewarded_trials)])

        #trials, unique_trials = remake_trial_numbers(reward_trials)
        ## stack and segment a few trials

        data = np.vstack((rates, position, trials, speed))
        data = np.transpose(data)

        #trial_data = data[data[:,2] > 10,:]
        #trial_data = trial_data[trial_data[:,2] < 27,:]
        data_length = len(rates)
        bits = data_length/10

        trial_data = data[:int(bits), :]
        trials_points = np.diff(np.array(trial_data[:,2]))
        trial_y_coords = np.where(trials_points > 0 )
        slow_data = np.where(trial_data[:,3] < 3)
        reward_data = np.where(trial_data[:,1] == 100)
        stop_data = np.where(trial_data[:,3] < 4.7)
        failed_data = np.where(np.isin(trial_data[:,2], rewarded_trials, invert=True))

        # plot raw position
        avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
        ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
        ymax = np.nanmax(trial_data[:,1])
        ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
        ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('cm', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticklabels(['', '', ''])
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis


        # plot raw spikes
        ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
        ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
        ymax = np.nanmax(trial_data[:,0])
        ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
        ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
        ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
        ax.locator_params(axis = 'x', nbins=3)
        plt.ylabel('Hz', fontsize=14, labelpad = 10)
        plt.xlabel('Seconds', fontsize=14, labelpad = 10)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all.png', dpi=200)
        plt.close()


        try:
            trial_data = data[int(bits):int(bits*2), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis


            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
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
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_2.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")


        try:
            trial_data = data[int(bits*2):int(bits*3), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis


            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_3.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*3):int(bits*4), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_4.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*4):int(bits*5), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Position (cm)', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis


            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Firing rate (Hz)', fontsize=14, labelpad = 10)
            plt.xlabel('Time (sec)', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_5.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")


        try:
            trial_data = data[int(bits*5):int(bits*6), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_6.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*6):int(bits*7), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_7.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*7):int(bits*8), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_8.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*8):int(bits*9), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 4.7)

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_9.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")

        try:
            trial_data = data[int(bits*9):int(bits*10), :]
            trials_points = np.diff(np.array(trial_data[:,2]))
            trial_y_coords = np.where(trials_points > 0 )
            slow_data = np.where(trial_data[:,3] < 3)
            reward_data = np.where(trial_data[:,1] == 100)
            stop_data = np.where(trial_data[:,3] < 3)
            failed_data = np.where(np.isin(trial_data[:,2], rewarded_trials, invert=True))

            # plot raw position
            avg_spikes_on_track = plt.figure(figsize=(25,3)) # width, height?
            ax = avg_spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,1], '-', color='red', linewidth = 1, alpha=0.5)
            ymax = np.nanmax(trial_data[:,1])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('cm', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_xticklabels(['', '', ''])
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis

            ax = avg_spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
            ax.plot(trial_data[:,0], '-', color='Black', linewidth = 2, alpha=0.5)
            ymax = np.nanmax(trial_data[:,0])
            ax.vlines(trial_y_coords, 0,ymax, color='black', linewidth = 1)
            ax.vlines(slow_data,0, ymax, color='yellow', linewidth = 1, alpha=0.2)
            ax.vlines(reward_data, 0,ymax, color='green', linewidth = 1)
            ax.locator_params(axis = 'x', nbins=3)
            plt.ylabel('Hz', fontsize=14, labelpad = 10)
            plt.xlabel('Seconds', fontsize=14, labelpad = 10)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.tick_params(axis='both',  which='both',   bottom=True, top=False, right=False, left=True, labelleft=True, labelbottom=True, labelsize=18,length=5, width=1.5)  # labels along the bottom edge are off
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
            ax.set_xlim(0)
            ax.set_ylim(0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.1, right = 0.9, top = 0.9)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_raw_data_' + str(cluster_index +1) + '_all_10.png', dpi=200)
            plt.close()
        except(IndexError, ValueError):
            print("")
    return spike_data
