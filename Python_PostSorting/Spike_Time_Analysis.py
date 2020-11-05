import os
import matplotlib.pylab as plt
import numpy as np
import cmath



### --------------------------------------------------------------------------------------------------- ###

###

def find_indices_trial_start(data):
    # find index when trial increases
    new_trial_indices = []

    for rowcount, row in enumerate(data):
        if rowcount > 0:
            current_trial = data[data[rowcount,1]]
            previous_trial = data[data[rowcount-1,1]]
            if (current_trial-previous_trial) != 0:
                index = rowcount
                new_trial_indices = np.append(new_trial_indices, index)
            elif (current_trial-previous_trial) == 0:
                continue
    new_trial_indices = np.hstack((0, new_trial_indices))
    #data = np.vstack((rates, trials, rates, new_trial_indices))
    return new_trial_indices



def calculate_time_from_trial_start(data):
    # make time from trial
    trial_time = np.zeros((data.shape[0]))
    ticker = 0
    for rowcount, row in enumerate(data):
        if rowcount > 0:
            ticker += 250

            current_trial = data[rowcount,1]
            previous_trial = data[rowcount-1,1]
            if (current_trial-previous_trial) != 0:
                ticker = 0

            trial_time[rowcount] = ticker
    return trial_time


def run_spike_time_analysis(spike_data, recording_folder):
    print('I am plotting rates versus time from trial start ...')
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        rates =  np.array(spike_data.iloc[cluster].spike_rate_in_time[0]).real
        trials =  np.array(spike_data.iloc[cluster].spike_rate_in_time[3]).real
        position = np.array(spike_data.iloc[cluster].spike_rate_in_time[2]).real
        speed = np.array(spike_data.iloc[cluster].spike_rate_in_time[1]).real/1000

        data = np.vstack((rates, trials, rates))
        data = np.transpose(data)

        trial_time = calculate_time_from_trial_start(data)
        trial_time = np.transpose(trial_time)

        data = np.vstack((rates, speed, position, trial_time))
        data = np.transpose(data)

        rates_outbound , speed_outbound , position_outbound, time_outbound = remove_low_speeds_and_segment(data )

        avg_spikes_on_track = plt.figure(figsize=(4,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        area = np.pi*1
        plt.scatter(time_outbound, rates_outbound*4, s=area, c=position_outbound)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        plt.ylabel('Spike rate (hz)', fontsize=16, labelpad = 10)
        plt.xlabel('Time since trial start (ms)', fontsize=16, labelpad = 10)
        x_max = np.nanmax(rates)
        ax.set_xlim(0)
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
        ax.axvline(0, linewidth = 1.5, color = 'black') # bold line on the y axis
        ax.axhline(0, linewidth = 1.5, color = 'black') # bold line on the x axis
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #plt.text(0.7,0.7, "r-value" + str(round(r_value,2)) + "p-value" + str(round(p_value, 2)))

        save_path = recording_folder + 'Figures/InstantRates'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + spike_data.session_id.values[cluster] + '_rate_map_Cluster_' + str(cluster_index +1) + '_time' + '_coded_outbound.png', dpi=200)
        plt.close()


    return spike_data




def remove_low_speeds_and_segment(data ):

    data_filtered = data[data[:,1] > 2.4,:]

    data_filtered = data_filtered[data_filtered[:,2] >= 30,:]
    data_filtered = data_filtered[data_filtered[:,2] <= 170,:]

    data_outbound = data_filtered[data_filtered[:,2] <= 90,:]

    rates_outbound = data_outbound[:,0]
    speed_outbound = data_outbound[:,1]
    position_outbound = data_outbound[:,2]
    time_outbound = data_outbound[:,3]

    return rates_outbound , speed_outbound , position_outbound, time_outbound
