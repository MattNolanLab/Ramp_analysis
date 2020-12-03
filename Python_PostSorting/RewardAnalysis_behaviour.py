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
        num_of_rewards = rewards[~np.isnan(rewards)] +1
        num_of_rewards = int(num_of_rewards.shape[0])

        trials = np.unique(np.array(spike_data.loc[cluster,'stop_trial_number']))
        num_of_trials = trials[~np.isnan(trials)]
        #num_of_trials = spike_data.loc[cluster].max_trial_number
        num_of_trials = int(num_of_trials.shape[0]-1)

        reward_rate = (num_of_rewards*num_of_trials)/100
        if reward_rate > 100:
            reward_rate = 100
        spike_data.at[cluster, 'Reward_Rate'] = reward_rate
    return spike_data




### -------------------------------------------------------------- ###


### Calculates reward rate for each animal individually


### -------------------------------------------------------------- ###


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    #mouse1 = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return int(day)


def calculate_rewardrate_learning_curve(spike_data):
    #make dataframe of size needed (mice x days)
    #extract mice and day as loop from each cluster
    array = np.zeros((1,40)) # mice/days
    deviation_array = np.zeros((1,40)) # mice/days

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day = extract_mouse_and_day(session_id)
        rate = np.array(spike_data.at[cluster, 'Reward_Rate'])
        #sd_firststop = np.array(spike_data.at[cluster, 'SD_FirstStopcm'])
        array[0,day] = rate
        #deviation_array[0,day] = sd_firststop

        # plot graph
    plot_RewardRate_curve(array,deviation_array)
    write_to_csv(array,deviation_array)

    return


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
def write_to_csv(array, deviation_array):
    csvData = np.vstack((array,deviation_array))
    with open('/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c3_m2.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    return







### -------------------------------------------------------------- ###


### Loads individual first stop csv's and plots (from above code)


### -------------------------------------------------------------- ###



def load_csv_data():
    #with open('/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c4_m2.csv', newline='') as csvfile:
    #    m1 = csv.reader(csvfile, delimiter=' ', quotechar='|')
    m1 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c1_m1.csv", delimiter=None, header=None)
    m1=m1.transpose()
    m2 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c1_m6.csv", delimiter=None, header=None)
    m2=m2.transpose()
    m3 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c2_m1.csv", delimiter=None, header=None)
    m3=m3.transpose()
    m4 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c2_m2.csv", delimiter=None, header=None)
    m4=m4.transpose()
    m5 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c3_m1.csv", delimiter=None, header=None)
    m5=m5.transpose()
    m6 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c3_m2.csv", delimiter=None, header=None)
    m6=m6.transpose()
    m7 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c4_m1.csv", delimiter=None, header=None)
    m7=m7.transpose()
    m8 = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/RewardRate_c4_m2.csv", delimiter=None, header=None)
    m8=m8.transpose()
    #print(m1)

    return np.asarray(m1), np.asarray(m2), np.asarray(m3), np.asarray(m4), np.asarray(m5), np.asarray(m6), np.asarray(m7), np.asarray(m8)


def multimouse_rewardrate_plot():
    m1, m2, m3, m4, m5, m6, m7, m8 = load_csv_data()

    array = np.vstack((m1[1:30,0],np.flip(m2[1:30,0]),m3[1:30,0],m4[1:30,0],m5[1:30,0],m6[1:30,0],m7[1:30,0],m8[1:30,0]))
    sdarray_mean = np.nanstd(array, axis=0)/math.sqrt(8)

    array[array == 0] = 'nan'

    #array = np.flip(array)
    #sdarray = np.vstack((m1sd[0:25],m2sd[0:25],m3sd[0:25],m4sd[0:25],m5sd[0:25],m6sd[0:25],m7sd[0:25],m8sd[0:25]))
    #sdarray = np.flip(sdarray)

    array_mean = np.nanmean(array, axis=0)
    #sdarray_mean = np.std(array, axis=0)

    #array_mean = np.flip(array_mean)
    days = np.arange(1,30,1)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array_mean, '-', color='Black')
    ax.fill_between(days, array_mean-sdarray_mean,array_mean+sdarray_mean, facecolor = 'Black', alpha = 0.2)
    plt.ylabel('Reward rate (%)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    ax.set_ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array_mean)+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/allmice_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    plot_individual_mice_together(array)
    plot_individual_mice(array)
    return




def plot_individual_mice_together(array):
    #array_mean = np.flip(array_mean)
    days = np.arange(1,30,1)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[0,:], '-', color='Black')
    ax.plot(days,array[1,:], '-', color='purple')
    ax.plot(days,array[2,:], '-', color='blue')
    ax.plot(days,array[3,:], '-', color='green')
    ax.plot(days,array[4,:], '-', color='yellow')
    ax.plot(days,array[5,:], '-', color='orange')
    ax.plot(days,array[6,:], '-', color='red')
    ax.plot(days,array[7,:], '-', color='pink')
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    plt.xlim(0,25)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/AllMice_together_RewardRate_curve' + '.png', dpi=200)
    plt.close()




def plot_individual_mice(array):
    #array_mean = np.flip(array_mean)
    days = np.arange(1,30,1)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[0,:], '-', color='Black')
    #ax.fill_between(days, array[0,:]-sdarray[0,:],array[0,:]+sdarray[0,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M1_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[1,:], '-', color='Black')
    #ax.fill_between(days, array[1,:]-sdarray[1,:],array[1,:]+sdarray[1,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[1,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M2_RewardRate_curve' + '.png', dpi=200)
    plt.close()

    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[2,:], '-', color='Black')
    #ax.fill_between(days, array[2,:]-sdarray[2,:],array[2,:]+sdarray[2,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M3_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[3,:], '-', color='Black')
    #ax.fill_between(days, array[3,:]-sdarray[3,:],array[3,:]+sdarray[3,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M4_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[5,:], '-', color='Black')
    #ax.fill_between(days, array[5,:]-sdarray[5,:],array[5,:]+sdarray[5,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M5_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[6,:], '-', color='Black')
    #ax.fill_between(days, array[6,:]-sdarray[6,:],array[6,:]+sdarray[6,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M6_RewardRate_curve' + '.png', dpi=200)
    plt.close()


    #array_mean = np.flip(array_mean)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(days,array[7,:], '-', color='Black')
    #ax.fill_between(days, array[7,:]-sdarray[7,:],array[7,:]+sdarray[7,:], facecolor = 'Black', alpha = 0.2)
    plt.ylabel('First stop location (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Training day', fontsize=12, labelpad = 10)
    ax.set_xlim(1,30)
    plt.ylim(0,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #Python_PostSorting.plot_utility.style_track_plot(ax, 200)
    x_max = max(array[0,:])+0.1
    #Python_PostSorting.plot_utility.style_vr_plot(ax, x_max)
    ax.axvline(0, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
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
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/M7_RewardRate_curve' + '.png', dpi=200)
    plt.close()

