import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from scipy import stats
import os
import sys
import settings
import Ramp_analysis.Integrated_ramp_analysis.Concatenate_vr_shuffle_analysis as Concatenate_vr_shuffle_analysis
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility
from astropy.convolution import convolve, Gaussian1DKernel

def add_firing_rate_maps_by_trial_type(spatial_firing, b_trial_numbers, nb_trial_numbers, p_trial_numbers):

    beaconed_maps = []; non_beaconed_maps = []; probe_maps = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_firing = spatial_firing[spatial_firing["cluster_id"]==cluster_id]
        fr_binned_in_space = np.array(cluster_firing["fr_binned_in_space"].iloc[0])

        b_fr_binned_in_space = fr_binned_in_space[b_trial_numbers-1]
        nb_fr_binned_in_space = fr_binned_in_space[nb_trial_numbers-1]
        p_fr_binned_in_space = fr_binned_in_space[p_trial_numbers-1]

        beaconed_maps.append(np.nanmean(b_fr_binned_in_space, axis=0).tolist())
        non_beaconed_maps.append(np.nanmean(nb_fr_binned_in_space, axis=0).tolist())
        probe_maps.append(np.nanmean(p_fr_binned_in_space, axis=0).tolist())

    spatial_firing["beaconed_map"] = beaconed_maps
    spatial_firing["non_beaconed_map"] = non_beaconed_maps
    spatial_firing["probe_map"] = probe_maps
    return spatial_firing

def plot_shuffles(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded, output_path, track_region):
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/shuffle_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if by_rewarded:
        processed_position_data = processed_position_data[processed_position_data["rewarded"] == 1]
    b_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 0]["trial_number"])
    nb_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 1]["trial_number"])
    p_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 2]["trial_number"])
    spike_data = add_firing_rate_maps_by_trial_type(spike_data, b_trial_numbers, nb_trial_numbers, p_trial_numbers)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data.cluster_id == cluster_id]
        shuffle_cluster_data = shuffled_data[shuffled_data.cluster_id == cluster_id]

        # plot beaconed rate map of the measured rate
        cluster_b_rate_map = np.array(cluster_spike_data["beaconed_map"].iloc[0])
        bins=np.arange(0.5,199.5+1,1)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,cluster_b_rate_map, '-', color='Black')
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
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_rewarded.png', dpi=200)
        plt.close()

        # plot overlapping beaconed rate map of the shuffled
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for i in range(n_shuffle_examples):
            shuffled_beaconed_rate_map = shuffle_cluster_data['beaconed_map'].iloc[i]
            ax.plot(bins,shuffled_beaconed_rate_map, '-', color='Black', alpha=0.3)
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
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_shuffled_rewarded.png', dpi=200)
        plt.close()

        # plot the fits of the measured and shuffled results
        shuffle_cluster_data = Concatenate_vr_shuffle_analysis.calculation_slopes(shuffle_cluster_data, track_region=track_region)
        cluster_spike_data = Concatenate_vr_shuffle_analysis.calculation_slopes(cluster_spike_data, track_region=track_region)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        x_vals = np.array([0,60])
        for i in range(n_shuffle_examples):
            shuffled_slope = shuffle_cluster_data["beaconed_slope_"+track_region].iloc[i]
            shuffled_intercept = shuffle_cluster_data["beaconed_intercept_"+track_region].iloc[i]
            y_vals = shuffled_intercept + (shuffled_slope*x_vals)
            ax.plot(x_vals,y_vals, '-', color='Grey')
        cluster_slope = cluster_spike_data["beaconed_slope_"+track_region].iloc[0]
        cluster_intercept = cluster_spike_data["beaconed_intercept_"+track_region].iloc[0]
        cluster_p_val = cluster_spike_data["beaconed_p_val_"+track_region].iloc[0]
        cluster_r2 = cluster_spike_data["beaconed_r2_"+track_region].iloc[0]
        y_vals = cluster_intercept + (cluster_slope*x_vals)
        ax.plot(x_vals, y_vals, '-', color='Red')
        plt.xlim(-3,63)
        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        ax.set_xticks([0,60])
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_slopes.png', dpi=200)

        # print out the measured fit
        print("cluster id: ", str(cluster_id), " has these linear regression fits, R2: ", str(cluster_r2), ", slope: ", str(cluster_slope), ", p: ", str(cluster_p_val))


        # plot R2 by slope of shuffled and measured and show percentile marks
        plot_slopes_and_R2s = True
        if plot_slopes_and_R2s:
            percentile_5 = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 5)
            percentile_95 = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 95)
            avg_spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            ax.scatter(shuffle_cluster_data["beaconed_slope_"+track_region], shuffle_cluster_data["beaconed_r2_"+track_region], c="Black", marker="o")
            ax.scatter(cluster_spike_data["beaconed_slope_"+track_region], cluster_spike_data["beaconed_r2_"+track_region], c="Red", marker="o")
            ax.axvline(x=percentile_5, linewidth=2, color="Blue", linestyle="dotted")
            ax.axvline(x=percentile_95, linewidth=2, color="Blue", linestyle="dotted")

            plt.ylabel('R2', fontsize=19, labelpad = 0)
            plt.xlabel('Slope (Hz/cm)', fontsize=18, labelpad = 10)
            Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
            plt.locator_params(axis = 'y', nbins  = 5)
            plt.locator_params(axis = 'x', nbins  = 3)
            ax.set_ylim([-0.05, 1])
            ax.set_yticks([0, 0.5, 1])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_slopes_vs_r2.png', dpi=200)
    return

def extract_smoothed_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[0], 'trial_number' :  np.array(spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], dtype=np.int16), 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings
def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==0]
    nbeaconed_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==1]
    probe_cluster_firings = cluster_firings[cluster_firings["trial_type"] ==2]
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings
def split_firing_data_by_reward(cluster_firings, rewarded_trials):
    rewarded_cluster_firings = cluster_firings.loc[cluster_firings['trial_number'].isin(rewarded_trials)]
    rewarded_cluster_firings.reset_index(drop=True, inplace=True)
    return rewarded_cluster_firings
def reshape_and_average_over_trials(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings):
    bin=200
    data_b = pd.DataFrame(beaconed_cluster_firings, dtype=None, copy=False)
    beaconed_cluster_firings = np.asarray(data_b)

    data_nb = pd.DataFrame(nonbeaconed_cluster_firings, dtype=None, copy=False)
    nonbeaconed_cluster_firings = np.asarray(data_nb)

    data_p = pd.DataFrame(probe_cluster_firings, dtype=None, copy=False)
    probe_cluster_firings = np.asarray(data_p)

    beaconed_reshaped_hist = np.reshape(beaconed_cluster_firings, (int(beaconed_cluster_firings.size/bin),bin))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_cluster_firings, (int(nonbeaconed_cluster_firings.size/bin), bin))
    probe_reshaped_hist = np.reshape(probe_cluster_firings, (int(probe_cluster_firings.size/bin), bin))

    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)

    average_beaconed_sd = stats.sem(beaconed_reshaped_hist, axis=0, nan_policy="omit")
    average_nonbeaconed_sd = stats.sem(nonbeaconed_reshaped_hist, axis=0, nan_policy="omit")
    average_probe_sd = stats.sem(probe_reshaped_hist, axis=0, nan_policy="omit")
    plt.plot(average_beaconed_spike_rate)
    plt.close()

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16), average_beaconed_sd, average_nonbeaconed_sd, average_probe_sd

def get_cell_class_color(p_val, slope, min_slope, max_slope):
    if (p_val<0.01):
        if (slope>max_slope):
            c = "lime"
        elif (slope<min_slope):
            c = "magenta"
        else:
            c = "grey"
    else:
        c = "grey"
    return c

def plot_shuffles_test(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded, output_path, track_region, shuffle_method):
    print('I am plotting firing rate maps...')
    if shuffle_method == "cyclic":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic'
    elif shuffle_method == "cyclic2":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic2'
    elif shuffle_method == "cyclic_unsmoothened":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic_unsmoothened'
    elif shuffle_method == "cyclic_unsmoothened2":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic_unsmoothened2'
    elif shuffle_method == "space_scramble":
        save_path = output_path + '/Figures/shuffle_rate_map_space_bin_scramble'
    elif shuffle_method == "cyclic_by_trial":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic_by_trial'
    elif shuffle_method == "cyclic_by_trial_unsmoothened":
        save_path = output_path + '/Figures/shuffle_rate_map_cyclic_by_trial_unsmoothened'
    elif shuffle_method == "space_scramble_at_trial_level":
        save_path = output_path + '/Figures/shuffle_rate_map_space_bin_scramble_at_trial_level'
    elif shuffle_method == "space_scramble_at_trial_level_unsmoothened":
        save_path = output_path + '/Figures/shuffle_rate_map_space_bin_scramble_at_trial_level_unsmoothened'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    gauss_kernel = Gaussian1DKernel(2)
    x_vals = np.array([30,90])
    if by_rewarded:
        processed_position_data = processed_position_data[processed_position_data["rewarded"] == 1]
    b_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 0]["trial_number"])
    nb_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 1]["trial_number"])
    p_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 2]["trial_number"])
    spike_data = add_firing_rate_maps_by_trial_type(spike_data, b_trial_numbers, nb_trial_numbers, p_trial_numbers)

    p_values_shuffles = []
    p_values_measured =[]
    slope_shuffled = []
    r2_shuffled = []
    class_shuffled = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data.cluster_id == cluster_id]
        shuffle_cluster_data = shuffled_data[shuffled_data.cluster_id == cluster_id]

        if shuffle_method == "space_scramble":
            # remake shuffle by spatial scramble
            beaconed_maps = []
            for i in range(len(shuffle_cluster_data)):
                cluster_b_rate_map = np.array(cluster_spike_data["beaconed_map"].iloc[0])
                np.random.shuffle(cluster_b_rate_map)
                beaconed_maps.append(cluster_b_rate_map)
            shuffle_cluster_data["beaconed_map"] = beaconed_maps

        elif shuffle_method == "space_scramble_at_trial_level":
            # remake shuffle by spatial scramble at the trial level
            beaconed_maps = []
            for i in range(len(shuffle_cluster_data)):
                firing_rates_trials = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])[b_trial_numbers-1]
                # shuffle each trial
                for j in range(len(firing_rates_trials)):
                    np.random.shuffle(firing_rates_trials[j])
                flattened_rate_map = np.array(firing_rates_trials).flatten()
                flattened_rate_map = convolve(flattened_rate_map, gauss_kernel)
                flattened_rate_map[np.isnan(flattened_rate_map)] = 0
                smoothened_rate_map = flattened_rate_map.reshape((int(len(flattened_rate_map)/200)), 200)
                # average rates across trials
                beaconed_map = np.nanmean(smoothened_rate_map, axis=0)
                beaconed_maps.append(beaconed_map)
            shuffle_cluster_data["beaconed_map"] = beaconed_maps

        elif shuffle_method == "space_scramble_at_trial_level_unsmoothened":
            # remake shuffle by spatial scramble at the trial level
            beaconed_maps = []
            for i in range(len(shuffle_cluster_data)):
                firing_rates_trials = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])[b_trial_numbers-1]
                # shuffle each trial
                for j in range(len(firing_rates_trials)):
                    np.random.shuffle(firing_rates_trials[j])
                flattened_rate_map = np.array(firing_rates_trials).flatten()
                #flattened_rate_map = convolve(flattened_rate_map, gauss_kernel)
                flattened_rate_map[np.isnan(flattened_rate_map)] = 0
                smoothened_rate_map = flattened_rate_map.reshape((int(len(flattened_rate_map)/200)), 200)
                # average rates across trials
                beaconed_map = np.nanmean(smoothened_rate_map, axis=0)
                beaconed_maps.append(beaconed_map)
            shuffle_cluster_data["beaconed_map"] = beaconed_maps

        else:
            print("were keeping the shuffled rate maps as is")

        shuffle_cluster_data = Concatenate_vr_shuffle_analysis.calculation_slopes(shuffle_cluster_data, track_region=track_region)
        min_slope = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 5)
        max_slope = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 95)

        # plot beaconed rate map of the measured rate
        cluster_spike_data_subset = Concatenate_vr_shuffle_analysis.calculation_slopes(cluster_spike_data, track_region=track_region)
        cluster_r2 = cluster_spike_data_subset["beaconed_r2_"+track_region].iloc[0]
        cluster_slope = cluster_spike_data_subset["beaconed_slope_"+track_region].iloc[0]
        cluster_p_val = cluster_spike_data_subset["beaconed_p_val_"+track_region].iloc[0]
        cluster_intercept = cluster_spike_data_subset["beaconed_intercept_"+track_region].iloc[0]
        p_values_measured.append(cluster_p_val)
        cluster_scientific_p="{:.2e}".format(cluster_p_val)
        c = get_cell_class_color(cluster_p_val, cluster_slope, min_slope, max_slope)

        cluster_b_rate_map = np.array(cluster_spike_data["beaconed_map"].iloc[0])
        bins=np.arange(0.5,199.5+1,1)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins,cluster_b_rate_map, '-', color="black")
        y_vals = cluster_intercept + (cluster_slope*x_vals)
        ax.plot(x_vals, y_vals, '-', color=c)
        ax.text(x=0.05, y=0.05, s=f"R2: {np.round(cluster_r2, decimals=2)}", fontsize=12, transform=ax.transAxes)
        ax.text(x=0.05, y=0.15, s=f"Slope: {np.round(cluster_slope, decimals=2)}", fontsize=12, transform=ax.transAxes)
        ax.text(x=0.05, y=0.25, s=f"P: {cluster_scientific_p}", fontsize=12, transform=ax.transAxes)
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
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_rewarded.png', dpi=200)
        plt.close()

        for i in range(len(shuffle_cluster_data)):
            r2 = shuffle_cluster_data["beaconed_r2_"+track_region].iloc[i]
            slope = shuffle_cluster_data["beaconed_slope_"+track_region].iloc[i]
            p_val = shuffle_cluster_data["beaconed_p_val_"+track_region].iloc[i]
            p_values_shuffles.append(p_val)
            slope_shuffled.append(slope)
            r2_shuffled.append(r2)
            class_shuffled.append(get_cell_class_color(p_val, slope, min_slope, max_slope))

        #shuffle_cluster_data_subset = shuffle_cluster_data[shuffle_cluster_data["beaconed_p_val_"+track_region] < 0.05]
        shuffle_cluster_data_subset = shuffle_cluster_data.copy()
        for i in range(20):
            avg_spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            shuffled_beaconed_rate_map = shuffle_cluster_data_subset['beaconed_map'].iloc[i]
            r2 = shuffle_cluster_data_subset["beaconed_r2_"+track_region].iloc[i]
            slope = shuffle_cluster_data_subset["beaconed_slope_"+track_region].iloc[i]
            p_val = shuffle_cluster_data_subset["beaconed_p_val_"+track_region].iloc[i]
            intercept = shuffle_cluster_data_subset["beaconed_intercept_"+track_region].iloc[i]
            scientific_p="{:.2e}".format(p_val)
            c = get_cell_class_color(p_val, slope, min_slope, max_slope)
            ax.plot(bins,shuffled_beaconed_rate_map, '-', color="grey")
            y_vals = intercept + (slope*x_vals)
            ax.plot(x_vals,y_vals, '-', color=c)
            ax.text(x=0.05, y=0.05, s=f"R2: {np.round(r2, decimals=2)}", fontsize=12, transform=ax.transAxes)
            ax.text(x=0.05, y=0.15, s=f"Slope: {np.round(slope, decimals=2)}", fontsize=12, transform=ax.transAxes)
            ax.text(x=0.05, y=0.25, s=f"P: {scientific_p}", fontsize=12, transform=ax.transAxes)
            plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
            plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
            plt.xlim(0,200); ax.set_ylim(0)
            Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, 200)
            Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
            plt.locator_params(axis = 'x', nbins  = 3)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.set_xticklabels(['-30', '70', '170'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_shuffled_'+str(i)+'_rewarded.png', dpi=200)
            plt.close()

        # plot the fits of the measured and shuffled results
        shuffle_cluster_data = Concatenate_vr_shuffle_analysis.calculation_slopes(shuffle_cluster_data, track_region=track_region)
        cluster_spike_data = Concatenate_vr_shuffle_analysis.calculation_slopes(cluster_spike_data, track_region=track_region)
        avg_spikes_on_track = plt.figure(figsize=(3.7,3))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for i in range(n_shuffle_examples):
            shuffled_slope = shuffle_cluster_data["beaconed_slope_"+track_region].iloc[i]
            shuffled_intercept = shuffle_cluster_data["beaconed_intercept_"+track_region].iloc[i]
            y_vals = shuffled_intercept + (shuffled_slope*x_vals)
            ax.plot(x_vals-30,y_vals, '-', color='Grey')
        cluster_slope = cluster_spike_data["beaconed_slope_"+track_region].iloc[0]
        cluster_intercept = cluster_spike_data["beaconed_intercept_"+track_region].iloc[0]
        cluster_p_val = cluster_spike_data["beaconed_p_val_"+track_region].iloc[0]
        cluster_r2 = cluster_spike_data["beaconed_r2_"+track_region].iloc[0]
        y_vals = cluster_intercept + (cluster_slope*x_vals)
        ax.plot(x_vals-30, y_vals, '-', color='Red')
        plt.xlim(-3,63)
        plt.ylabel('Firing rate (Hz)', fontsize=19, labelpad = 0)
        plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
        ax.set_xticks([0,60])
        Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_slopes.png', dpi=200)

        # print out the measured fit
        print("cluster id: ", str(cluster_id), " has these linear regression fits, R2: ", str(cluster_r2), ", slope: ", str(cluster_slope), ", p: ", str(cluster_p_val))


        # plot R2 by slope of shuffled and measured and show percentile marks
        plot_slopes_and_R2s = True
        if plot_slopes_and_R2s:
            percentile_5 = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 5)
            percentile_95 = np.nanpercentile(shuffle_cluster_data["beaconed_slope_"+track_region], 95)
            avg_spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            ax.scatter(shuffle_cluster_data["beaconed_slope_"+track_region], shuffle_cluster_data["beaconed_r2_"+track_region], c="Black", marker="o")
            ax.scatter(cluster_spike_data["beaconed_slope_"+track_region], cluster_spike_data["beaconed_r2_"+track_region], c="Red", marker="o")
            ax.axvline(x=percentile_5, linewidth=2, color="Blue", linestyle="dotted")
            ax.axvline(x=percentile_95, linewidth=2, color="Blue", linestyle="dotted")

            plt.ylabel('R2', fontsize=19, labelpad = 0)
            plt.xlabel('Slope (Hz/cm)', fontsize=18, labelpad = 10)
            Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
            plt.locator_params(axis = 'y', nbins  = 5)
            plt.locator_params(axis = 'x', nbins  = 3)
            ax.set_ylim([-0.05, 1])
            ax.set_yticks([0, 0.5, 1])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_b_rate_map_Cluster_' + str(cluster_id) + '_slopes_vs_r2.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(np.array(p_values_shuffles), bins=50)
    plt.ylabel('Count', fontsize=19, labelpad = 0)
    plt.xlabel('P', fontsize=18, labelpad = 10)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    #ax.set_xlim([0, 1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_P_histogram.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(np.array(r2_shuffled), bins=50)
    plt.ylabel('Count', fontsize=19, labelpad = 0)
    plt.xlabel('R2', fontsize=18, labelpad = 10)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    #ax.set_xlim([0, 1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_R2_histogram.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(np.array(slope_shuffled), bins=50)
    plt.ylabel('Count', fontsize=19, labelpad = 0)
    plt.xlabel('Slope', fontsize=18, labelpad = 10)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    #ax.set_xlim([0, 1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_Slope_histogram.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.scatter(np.array(slope_shuffled), np.array(p_values_shuffles), marker="x", c=np.array(class_shuffled), alpha=0.5)
    plt.ylabel('P', fontsize=19, labelpad = 0)
    plt.xlabel('Slope', fontsize=18, labelpad = 10)
    #ax.set_yscale('log')
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    #ax.set_ylim([0, 1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_p_vs_slope_scatter.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.scatter(np.array(slope_shuffled), np.array(r2_shuffled), marker="x", c=np.array(class_shuffled), alpha=0.5)
    plt.ylabel('R2', fontsize=19, labelpad = 0)
    plt.xlabel('Slope', fontsize=18, labelpad = 10)
    #ax.set_ylim([0, 1])
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_r2_vs_slope_scatter.png', dpi=200)

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.scatter(np.array(p_values_shuffles), np.array(r2_shuffled), marker="x", c=np.array(class_shuffled), alpha=0.5)
    plt.ylabel('R2', fontsize=19, labelpad = 0)
    plt.xlabel('P', fontsize=18, labelpad = 10)
    ax.set_xscale('log')
    #ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/_r2_vs_p_scatter.png', dpi=200)
    return

def process_recordings(vr_recording_path_list, n_shuffle_examples=10, by_rewarded=True, track_region="ob"):
    suffix = ""
    if by_rewarded:
        suffix = "_rewarded"

    vr_recording_path_list.sort()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            if os.path.exists(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+".pkl"):
                shuffled_data_by_trial_unsmoothened = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_by_trial_unsmoothened.pkl")
                shuffled_data_by_trial = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_by_trial.pkl")
                shuffled_data = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+".pkl")
                shuffled_data_unsmoothened = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_unsmoothened.pkl")
                shuffled_data2 = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"2.pkl")
                shuffled_data_unsmoothened2 = pd.read_pickle(recording+"/MountainSort/DataFrames/shuffled_firing_rate_maps"+suffix+"_unsmoothened2.pkl")
                spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
                processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
                #plot_shuffles(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region)

                # for test cyclic vs space scramble shuffle
                plot_shuffles_test(spike_data, shuffled_data_by_trial_unsmoothened, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic_by_trial_unsmoothened")
                plot_shuffles_test(spike_data, shuffled_data_by_trial, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic_by_trial")
                plot_shuffles_test(spike_data, shuffled_data_unsmoothened, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic_unsmoothened")
                plot_shuffles_test(spike_data, shuffled_data_unsmoothened2, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic_unsmoothened2")
                plot_shuffles_test(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="space_scramble_at_trial_level")
                plot_shuffles_test(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="space_scramble_at_trial_level_unsmoothened")
                plot_shuffles_test(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic")
                plot_shuffles_test(spike_data, shuffled_data2, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="cyclic2")
                plot_shuffles_test(spike_data, shuffled_data, processed_position_data, n_shuffle_examples, by_rewarded=by_rewarded, output_path=output_path, track_region=track_region, shuffle_method="space_scramble")
            print("successfully processed on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process analysis on "+recording)


#  for testing
def main():
    print('-------------------------------------------------------------')
    by_rewarded = True
    n_shuffle_examples = 50

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffle_examples, by_rewarded)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffle_examples, by_rewarded)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffle_examples, by_rewarded)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffle_examples, by_rewarded)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()]
    process_recordings(vr_path_list, n_shuffle_examples, by_rewarded)

    print("shuffle_data dataframes have been remade")

if __name__ == '__main__':
    main()
