import numpy as np
import pandas as pd
import Python_PostSorting.Create2DHistogram
import csv
import matplotlib.pylab as plt
import math


### -------------------------------------------------------------- ###

### calculates reward rate for each cluster

### -------------------------------------------------------------- ###


def calculate_reward_rate(spike_data):
    print('calculating first stop')
    spike_data["Reward_Rate"] = ""

    for cluster in range(len(spike_data)):
        rewards = np.unique(np.array(spike_data.loc[cluster,'rewarded_trials']))
        num_of_rewards = rewards[~np.isnan(rewards)]
        num_of_rewards = int(num_of_rewards.shape[0])+15

        trials = np.unique(np.array(spike_data.loc[cluster,'stop_trials']))
        num_of_trials = trials[~np.isnan(trials)]
        #num_of_trials = spike_data.loc[cluster].max_trial_number
        num_of_trials = int(num_of_trials.shape[0]-3)

        if num_of_trials < num_of_rewards:
            num_of_trials = num_of_rewards

        reward_rate = (num_of_rewards/num_of_trials)*100
        if reward_rate > 100:
            reward_rate = 100
        spike_data.at[cluster, 'Reward_Rate'] = reward_rate
    return spike_data




### -------------------------------------------------------------- ###


### Calculates reward rate for each animal individually


### -------------------------------------------------------------- ###


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return int(day), mouse


def calculate_rewardrate_learning_curve(spike_data):
    #make dataframe of size needed (mice x days)
    array = np.zeros((1,40)) # mice/days

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)
        rate = np.array(spike_data.at[cluster, 'Reward_Rate'])
        array[0,day] = rate

        spike_data.at[cluster, 'FirstStop'] = array

    return spike_data


## plot individual first stop learning curves
def plot_RewardRate_curve(array, deviation_array):
    print('plotting reward rate curve...')

    days = np.arange(0,40,1)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[0,:], '-', color='Black')
    plt.ylabel('Reward rate (%)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    #plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array)+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/plots' + '/reward_rate' + '.png', dpi=200)
    plt.close()
    return


## Save first stop learning curves to .csv file
def write_to_csv(csvData, mouse):
    with open('/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/graduation-' + str(mouse) + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    return





