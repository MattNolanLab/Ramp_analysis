

import seaborn as sns


import numpy as np
import os
import matplotlib.pylab as plt
import Python_PostSorting.plot_utility
from scipy import signal
import pandas as pd
from Python_PostSorting import FirstStopAnalysis_behaviour


def plot_heatmaps(spike_data):
    return spike_data


def plot_heatmap_by_trial(spike_data, prm):
    print("I am plotting heatmaps of trial data...")
    save_path = prm.get_local_recording_folder_path() + '/Figures/heatmaps_per_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        try:
            rates=np.array(spike_data.loc[cluster, 'FiringRate_RunTrials_trials'])
            speed_histogram = plt.figure(figsize=(5,12))
            ax = sns.heatmap(np.transpose(rates))
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.axvline(100, linewidth = 1.5, color = 'black') # bold line on the y axis
            plt.locator_params(axis = 'x', nbins  = 4)
            plt.savefig(save_path + '/heatmap_Run_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
            plt.close()

            rates=np.array(spike_data.loc[cluster, 'FiringRate_TryTrials_trials'])
            speed_histogram = plt.figure(figsize=(5,12))
            ax = sns.heatmap(np.transpose(rates))
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.axvline(100, linewidth = 1.5, color = 'black') # bold line on the y axis
            plt.locator_params(axis = 'x', nbins  = 4)
            plt.savefig(save_path + '/heatmap_Try_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
            plt.close()

            rates=np.array(spike_data.loc[cluster, 'FiringRate_HitTrials_trials'])
            speed_histogram = plt.figure(figsize=(5,12))
            ax = sns.heatmap(np.transpose(rates))
            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.axvline(100, linewidth = 1.5, color = 'black') # bold line on the y axis
            plt.locator_params(axis = 'x', nbins  = 4)
            plt.savefig(save_path + '/heatmap_Hit_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + '.png', dpi=200)
            plt.close()
        except ValueError:
            continue

    return spike_data





def plot_average(spike_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/Trisl_Outcome_FR'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        position_array=np.arange(0,200,1)

        rates=np.array(spike_data.loc[cluster, 'Avg_FiringRate_RunTrials'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FiringRate_RunTrials'])

        if rates.size > 5:
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Python_PostSorting.plot_utility.style_track_plot(ax, 200)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
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

            ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)
            ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'Run.png', dpi=200)
            plt.close()

        rates=np.array(spike_data.loc[cluster, 'Avg_FiringRate_TryTrials'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FiringRate_TryTrials'])

        if rates.size > 5:
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Python_PostSorting.plot_utility.style_track_plot(ax, 200)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
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

            ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)
            ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'Try.png', dpi=200)
            plt.close()

        rates=np.array(spike_data.loc[cluster, 'Avg_FiringRate_HitTrials'])
        sd_rates=np.array(spike_data.loc[cluster, 'SD_FiringRate_HitTrials'])

        if rates.size > 5:
            speed_histogram = plt.figure(figsize=(4,3))
            ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(position_array,rates, '-', color='Black')
            ax.fill_between(position_array, rates-sd_rates,rates+sd_rates, facecolor = 'Black', alpha = 0.2)

            plt.ylabel('Firing rates (Hz)', fontsize=16, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=16, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Python_PostSorting.plot_utility.style_track_plot(ax, 200)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
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

            ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
            ax.set_ylim(0)
            plt.locator_params(axis = 'x', nbins  = 4)
            ax.set_xticklabels(['10', '30', '50'])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/time_binned_Rates_histogram_' + spike_data.session_id[cluster] + '_' + str(cluster_index +1) + 'Hit.png', dpi=200)
            plt.close()


    return spike_data

