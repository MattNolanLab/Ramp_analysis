import os
import matplotlib.pylab as plt
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility
import numpy as np
from scipy import signal
import scipy.stats as stats
from astropy.convolution import convolve, Gaussian1DKernel

def calculate_average_stops(spike_data):
    spike_data["average_stops"] = ""
    spike_data["position_bins"] = ""
    spike_data["average_stops_se"] = ""
    spike_data["average_first_stops"] = ""
    for cluster in range(len(spike_data)):
        stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
        max_trial_number = int(spike_data.at[cluster, 'max_trial_number'])

        number_of_bins = 200
        number_of_trials = max_trial_number
        stop_counts = np.zeros((number_of_trials, number_of_bins))
        first_stop_counts = np.zeros((number_of_trials, number_of_bins))
        for i in range(number_of_trials):
            stop_locations_on_trial = stop_locations[stop_trials == i+1]
            first_stop_locations_on_trial = stop_locations_on_trial[stop_locations_on_trial>30]
            stop_in_trial_bins, bin_edges = np.histogram(stop_locations_on_trial, bins=200, range=[0,200])
            if len(first_stop_locations_on_trial)>0:
                first_stop_in_trial_bins, bin_edges = np.histogram(first_stop_locations_on_trial[0], bins=200, range=[0,200])
            else:
                first_stop_in_trial_bins = np.zeros(len(stop_in_trial_bins))

            stop_counts[i,:] = stop_in_trial_bins
            first_stop_counts[i,:] = first_stop_in_trial_bins

        average_stops = np.nanmean(stop_counts, axis=0)
        average_first_stops = np.nanmean(first_stop_counts, axis=0)
        se_stops = stats.sem(stop_counts, axis=0, nan_policy="omit")
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

        #window = signal.gaussian(2, std=3)
        #stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
        spike_data.at[cluster,'average_stops'] = average_stops.tolist()
        spike_data.at[cluster,'position_bins'] = bin_centres.tolist()
        spike_data.at[cluster, 'average_stops_se'] = se_stops.tolist()
        spike_data.at[cluster, 'average_first_stops'] = average_first_stops.tolist()

    return spike_data

def get_trial_numbers_from_time_binned_data(nested_time_binned_data, tt):
    trial_numbers=np.array(nested_time_binned_data[3])
    trial_types=np.array(nested_time_binned_data[4])

    trial_numbers = trial_numbers[trial_types == tt]
    number_of_trials = len(np.unique(trial_numbers))
    return number_of_trials, np.unique(trial_numbers)

def calculate_average_nonbeaconed_stops(spike_data):
    spike_data["average_stops_nb"] = ""
    spike_data["average_stops_se_nb"] = ""
    spike_data["average_first_stops_nb"] = ""
    spike_data["average_stops_p"] = ""
    spike_data["average_stops_se_p"] = ""
    spike_data["average_first_stops_p"] = ""

    for trial_type, tt_numeric in zip(["_nb", "_p"], [1,2]):
        for cluster in range(len(spike_data)):
            stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
            stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
            stop_trial_types = np.array(spike_data.at[cluster, 'stop_trial_types'], dtype=np.int16)
            nested_time_binned_data = spike_data.at[cluster, 'spike_rate_in_time']

            # subset by the trial type
            stop_locations = stop_locations[stop_trial_types == tt_numeric]
            stop_trials = stop_trials[stop_trial_types == tt_numeric]

            # get the number of trial type specific trial numbers
            # we need to load this from the nested time binned data as no other spike_data column has this information
            number_of_trials, trial_numbers = get_trial_numbers_from_time_binned_data(nested_time_binned_data, tt=tt_numeric)
            number_of_bins = 200

            stop_counts = np.zeros((number_of_trials, number_of_bins))
            first_stop_counts = np.zeros((number_of_trials, number_of_bins))
            for i, tn in enumerate(trial_numbers):
                stop_locations_on_trial = stop_locations[stop_trials == tn]
                first_stop_locations_on_trial = stop_locations_on_trial[stop_locations_on_trial>30]
                stop_in_trial_bins, bin_edges = np.histogram(stop_locations_on_trial, bins=200, range=[0,200])

                if len(first_stop_locations_on_trial)>0:
                    first_stop_in_trial_bins, bin_edges = np.histogram(first_stop_locations_on_trial[0], bins=200, range=[0,200])
                else:
                    first_stop_in_trial_bins = np.zeros(len(stop_in_trial_bins))

                stop_counts[i,:] = stop_in_trial_bins
                first_stop_counts[i,:] = first_stop_in_trial_bins

            average_stops = np.nanmean(stop_counts, axis=0)
            average_first_stops = np.nanmean(first_stop_counts, axis=0)
            se_stops = stats.sem(stop_counts, axis=0, nan_policy="omit")
            spike_data.at[cluster,'average_stops'+trial_type] = average_stops.tolist()
            spike_data.at[cluster, 'average_stops_se'+trial_type] = se_stops.tolist()
            spike_data.at[cluster, 'average_first_stops'+trial_type] = average_first_stops.tolist()

    return spike_data


def calculate_average_speed(spike_data):
    spike_data["average_speed"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)
        window = signal.gaussian(2, std=3)

        data = np.vstack((speed,position,types, trials))
        data=data.transpose()
        #data = data[data[:,0] > 3,:]
        data = data[data[:,2] == 0,:]

        stop_speed = data[:,0]
        stop_locations = data[:,1]
        stop_trials = data[:,3]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = np.nanmax(stop_trials) # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = np.nansum(stop_speed[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
            spike_data.at[cluster,'average_speed'] = list(stops_in_bins)
    return spike_data


def calculate_average_nonbeaconed_speed(spike_data):
    spike_data["average_speed_nb"] = ""
    for cluster in range(len(spike_data)):
        speed=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[1].real)
        position=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[2].real)
        types=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[4].real, dtype= np.int32)
        trials=np.array(spike_data.iloc[cluster].spike_rate_in_time_rewarded[3].real, dtype= np.int32)
        window = signal.gaussian(2, std=3)

        data = np.vstack((speed,position,types, trials))
        data=data.transpose()
        #data = data[data[:,0] > 3,:]
        data = data[data[:,2] != 0,:]

        stop_speed = data[:,0]
        stop_locations = data[:,1]
        stop_trials = data[:,3]

        if len(stop_trials) > 1:
            number_of_bins = 200
            number_of_trials = np.nanmax(stop_trials)/3 # total number of trials
            stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
            for loc in range(int(number_of_bins)-1):
                stops_in_bin = np.nansum(stop_speed[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
                stops_in_bins[loc] = stops_in_bin

            stops_in_bins = signal.convolve(stops_in_bins, window, mode='same')/ sum(window)
            spike_data.at[cluster,'average_speed_nb'] = list(stops_in_bins)
    return spike_data


def plot_stop_histogram_per_cluster_probe(spike_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stop_histogram_probe'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    gauss_kernel = Gaussian1DKernel(3)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(3.7,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        window = signal.gaussian(2, std=3)
        position_bins = np.array(spike_data.at[cluster, 'position_bins'])
        average_stops = np.array(spike_data.at[cluster, 'average_stops'])
        average_stops_se = np.array(spike_data.at[cluster, 'average_stops_se'])
        average_stops = convolve(average_stops, gauss_kernel)
        average_stops_se = convolve(average_stops_se, gauss_kernel)
        ax.fill_between(position_bins,average_stops-average_stops_se, average_stops+average_stops_se, alpha=0.3, color='Black', edgecolor="none")
        ax.plot(position_bins, average_stops, color='Black')

        average_stops_nb =  np.array(spike_data.at[cluster, 'average_stops_nb'])
        average_stops_nb_se = np.array(spike_data.at[cluster, 'average_stops_se_nb'])
        average_stops_nb = convolve(average_stops_nb, gauss_kernel)
        average_stops_nb_se = convolve(average_stops_nb_se, gauss_kernel)
        #ax.fill_between(position_bins,average_stops_nb-average_stops_nb_se, average_stops+average_stops_nb_se, alpha=0.3, color='Blue', edgecolor="none")
        #ax.plot(position_bins,average_stops_nb, '-', color='Blue')

        average_stops_p =  np.array(spike_data.at[cluster, 'average_stops_p'])
        average_stops_p_se = np.array(spike_data.at[cluster, 'average_stops_se_p'])
        average_stops_p = convolve(average_stops_p, gauss_kernel)
        average_stops_p_se = convolve(average_stops_p_se, gauss_kernel)
        average_stops_p = signal.convolve(average_stops_p, window, mode='same')/ sum(window)
        ax.fill_between(position_bins,average_stops_p-average_stops_p_se, average_stops+average_stops_p_se, alpha=0.3, color='Blue', edgecolor="none")
        ax.plot(position_bins,average_stops_p, '-', color='Blue')

        plt.ylabel('Stops/cm', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_average_stops_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

def plot_stop_histogram_per_cluster(spike_data, prm, plot_p=False):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stop_histogram'
    if plot_p:
        save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stop_histogram_p'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    gauss_kernel = Gaussian1DKernel(1)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(3.7,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        position_bins = np.array(spike_data.at[cluster, 'position_bins'])

        average_stops = np.array(spike_data.at[cluster, 'average_stops'])
        average_stops_se = np.array(spike_data.at[cluster, 'average_stops_se'])
        average_stops = convolve(average_stops, gauss_kernel)
        average_stops_se = convolve(average_stops_se, gauss_kernel)
        ax.fill_between(position_bins,average_stops-average_stops_se, average_stops+average_stops_se, alpha=0.3, color='Black', edgecolor="none")
        ax.plot(position_bins, average_stops, color='Black')

        #average_stops_nb =  np.array(spike_data.at[cluster, 'average_stops_nb'])
        #average_stops_nb_se = np.array(spike_data.at[cluster, 'average_stops_se_nb'])
        #average_stops_nb = convolve(average_stops_nb, gauss_kernel)
        #average_stops_nb_se = convolve(average_stops_nb_se, gauss_kernel)
        #ax.fill_between(position_bins,average_stops_nb-average_stops_nb_se, average_stops+average_stops_nb_se, alpha=0.3, color='Blue', edgecolor="none")
        #ax.plot(position_bins,average_stops_nb, '-', color='Blue')

        if plot_p:
            average_stops_p =  np.array(spike_data.at[cluster, 'average_stops_p'])
            average_stops_p_se = np.array(spike_data.at[cluster, 'average_stops_se_p'])
            average_stops_p = convolve(average_stops_p, gauss_kernel)
            average_stops_p_se = convolve(average_stops_p_se, gauss_kernel)
            ax.fill_between(position_bins,average_stops_p-average_stops_p_se, average_stops_p+average_stops_p_se, alpha=0.3, color=(31/255, 181/255, 178/255), edgecolor="none")
            ax.plot(position_bins,average_stops_p, '-', color=(31/255, 181/255, 178/255))

        plt.ylabel('Stops/cm', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_average_stops_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

def curate_stops(spike_data):
    # stops are calculated as being below the stop threshold per unit time bin,
    # this function removes successive stops

    stop_locations_clusters = []
    stop_trials_clusters = []
    stop_trial_types_clusters = []
    for index, row in spike_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        stop_locations=np.array(row["stop_locations"].iloc[0])
        stop_trials=np.array(row["stop_trials"].iloc[0])
        stop_locations_elapsed=(200*(stop_trials-1))+stop_locations
        stop_trial_types=np.array(row["stop_trial_types"].iloc[0])

        curated_stop_locations=[]
        curated_stop_trials=[]
        curated_stop_trial_types=[]
        for i, stop_loc in enumerate(stop_locations_elapsed):
            if (i==0): # take first stop always
                add_stop=True
            elif ((stop_locations_elapsed[i]-stop_locations_elapsed[i-1]) > 1): # only include stop if the last stop was at least 1cm away
                add_stop=True
            else:
                add_stop=False

            if add_stop:
                curated_stop_locations.append(stop_locations_elapsed[i])
                curated_stop_trials.append(stop_trials[i])
                curated_stop_trial_types.append(stop_trial_types[i])

        # revert back to track positions
        curated_stop_locations = (np.array(curated_stop_locations)%200).tolist()

        stop_locations_clusters.append(curated_stop_locations)
        stop_trials_clusters.append(curated_stop_trials)
        stop_trial_types_clusters.append(curated_stop_trial_types)

    spike_data["stop_locations"] = stop_locations_clusters
    spike_data["stop_trials"] = stop_trials_clusters
    spike_data["stop_trial_types"] = stop_trial_types_clusters
    return spike_data

def plot_speed_histogram(spike_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/speed'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    gauss_kernel = Gaussian1DKernel(3)

    for index, row in spike_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        speeds=np.array(row["spike_rate_in_time"].iloc[0])[1,:]
        positions=np.array(row["spike_rate_in_time"].iloc[0])[2,:]
        trial_numbers=np.array(row["spike_rate_in_time"].iloc[0])[3,:]
        elapsed_position = (200*(trial_numbers-1))+positions
        number_of_trials=len(np.unique(trial_numbers))
        session_id = row["session_id"].iloc[0]
        cluster_id = row["cluster_id"].iloc[0]

        speed_histogram = plt.figure(figsize=(4,3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        number_of_bins = 200
        spatial_bins = np.arange(0, (number_of_trials*200)+1, 1) # 1 cm bins
        speed_in_bins_numerator, bin_edges = np.histogram(elapsed_position, spatial_bins, weights=speeds)
        speed_in_bins_denominator, bin_edges = np.histogram(elapsed_position, spatial_bins)
        speed_space_bin_means = speed_in_bins_numerator/speed_in_bins_denominator
        #speed_space_bin_means = convolve(speed_space_bin_means, gauss_kernel)
        tn_space_bin_means = (((0.5*(spatial_bins[1:]+spatial_bins[:-1]))//200)+1).astype(np.int64)

        # create empty array
        speed_trials = np.zeros((number_of_trials, number_of_bins)); i=0
        for trial_number in range(1, number_of_trials+1):
            speed_trials[i, :] = speed_space_bin_means[tn_space_bin_means == trial_number]
            i+=1


        average_speeds = np.nanmean(speed_trials, axis=0)
        se_speeds = stats.sem(speed_trials, axis=0, nan_policy="omit")

        average_speeds = convolve(average_speeds, gauss_kernel)
        se_speeds = convolve(se_speeds, gauss_kernel)

        ax.plot(np.arange(0.5, 200.5, 1), average_speeds, '-', color='Black')
        ax.fill_between(np.arange(0.5, 200.5, 1), average_speeds-se_speeds, average_speeds+se_speeds, edgecolor="none", color='Black', alpha=0.3)
        plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0,100,200])
        ax.set_xticklabels(["-30", "70", "170"])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.locator_params(axis = 'x', nbins  = 3)
        plt.locator_params(axis = 'y', nbins  = 4)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/speed_histogram_' +session_id + '_' + str(cluster_id) + '.png', dpi=200)
        plt.close()



def plot_speed(recording_folder, spike_data):
    print('I am plotting speed...')
    save_path = recording_folder + '/Figures/speed_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        position_bins = np.arange(0,200,1)
        avg_spikes_on_track = plt.figure(figsize=(4,3.5))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(position_bins,np.array(spike_data.at[cluster, "average_speed"]), '-', color='Black')

        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        ax.set_ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        ax.set_xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)

        x_max = np.nanmax(np.array(spike_data.at[cluster, "average_speed"]))
        plt.locator_params(axis = 'y', nbins = 4)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax, x_max, 0)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_speed_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



def split_stops_by_trial_type(locations, trials, trial_type):
    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]==0),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed, nonbeaconed, probe


def calculate_stop_types(spike_data, cluster, stop_trials):
    stop_types = []
    types=np.array(spike_data.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32)
    trials=np.array(spike_data.iloc[cluster].spike_rate_in_time[3].real, dtype= np.int32)
    data=np.transpose(np.vstack((trials, types)))

    for tcount, trial in enumerate(stop_trials):
        trial_data = data[data[:,0] == trial,:]
        try:
            type_in_trial = int(trial_data[0,1])
            stop_types = np.append(stop_types, type_in_trial)
        except IndexError:
            type_in_trial = 0
            stop_types = np.append(stop_types, type_in_trial)

    return stop_types


def split_stops_by_reward(beaconed,nonbeaconed,probe, rewarded_trials):
    #spike_data = add_columns_to_dataframe(spike_data)
    beaconed_position_cm = beaconed[:,0]
    beaconed_trial_number = beaconed[:,1]
    nonbeaconed_position_cm = nonbeaconed[:,0]
    nonbeaconed_trial_number = nonbeaconed[:,1]
    probe_position_cm = probe[:,0]
    probe_trial_number = probe[:,1]

    #take firing locations when on rewarded trials
    rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

    #take firing trial numbers when on rewarded trials
    rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
    rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
    rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]

    return rewarded_beaconed_position_cm, rewarded_nonbeaconed_position_cm, rewarded_probe_position_cm, rewarded_beaconed_trial_numbers, rewarded_nonbeaconed_trial_numbers, rewarded_probe_trial_numbers


def renumber_stop_trials_based_on_renumbered(unique_trial_numbers, old_unique_trial_numbers, stop_trials):
    new_trial_array = np.zeros((stop_trials.shape[0]))
    for rowcount, row in enumerate(stop_trials):
        current_trial = row
        new_trial = unique_trial_numbers[old_unique_trial_numbers[:] == current_trial]
        new_trial_array[rowcount] = new_trial
    return new_trial_array


def plot_stops_on_track_per_cluster(spike_data, prm, plot_p=True):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stops_on_trials'
    if plot_p:
        save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour/Stops_on_trials_p'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        stops_on_track = plt.figure(figsize=(3.7,3))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        stop_locations = np.array(spike_data.at[cluster, 'stop_locations'], dtype=np.int16)
        stop_trials = np.array(spike_data.at[cluster, 'stop_trials'], dtype=np.int16)
        stop_trial_types = np.array(spike_data.at[cluster, 'stop_trial_types'], dtype=np.int16)

        if plot_p:
            ax.plot(stop_locations[stop_trial_types==0], stop_trials[stop_trial_types==0], 'o', color='black', markersize=1)
            ax.plot(stop_locations[stop_trial_types==2], stop_trials[stop_trial_types==2], 'o', color='blue', markersize=1)
        else:
            ax.plot(stop_locations[stop_trial_types==0], stop_trials[stop_trial_types==0], 'o', color="Black", markersize=1)

        plt.ylabel('Stops on trials', fontsize=18, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        plt.xlim(0,200)
        plt.ylim(0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        ax.set_xticklabels(['-30', '70', '170'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_stop_raster_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()
