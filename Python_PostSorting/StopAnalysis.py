import numpy as np
import os
import pandas as pd
import math
import gc
import Python_PostSorting.LoadDataFrames
import matplotlib.pylab as plt
import csv
from scipy import stats

def check_stop_threshold(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    try:
        param_file_reader = open(parameters_path, 'r')
        parameters = param_file_reader.readlines()
        parameters = list([x.strip() for x in parameters])
        threshold = parameters[2]

    except Exception as ex:
        print('There is a problem with the parameter file.')
        print(ex)
    return np.float(threshold)


def keep_first_from_close_series(array, threshold):
    num_delete = 1
    while num_delete > 0:
        diff = np.ediff1d(array, to_begin= threshold + 1)
        to_delete = np.where(diff <= threshold)
        num_delete = len(to_delete[0])

        if num_delete > 0:
            array = np.delete(array, to_delete)
    return array


def get_beginning_of_track_positions(raw_position_data):
    location = np.array(raw_position_data['x_position_cm']) # Get the raw location from the movement channel
    position = 0
    beginning_of_track = np.where((location >= position) & (location <= position + 4))
    beginning_of_track = np.asanyarray(beginning_of_track)
    beginning_plus_one = beginning_of_track + 1
    beginning_plus_one = np.asanyarray(beginning_plus_one)
    track_beginnings = np.setdiff1d(beginning_of_track, beginning_plus_one)

    track_beginnings = keep_first_from_close_series(track_beginnings, 30000)
    return track_beginnings


def remove_extra_stops(min_distance, stops):
    to_remove = []
    for stop in range(len(stops) - 1):
        current_stop = stops[stop]
        next_stop = stops[stop + 1]
        if 0 <= (next_stop - current_stop) <= min_distance:
            to_remove.append(stop+1)

    filtered_stops = np.asanyarray(stops)
    np.delete(filtered_stops, to_remove)
    return filtered_stops


def get_stop_times(raw_position_data, stop_threshold):
    stops = np.array([])
    speed = np.array(raw_position_data['speed_per200ms'].tolist())

    threshold = stop_threshold
    low_speed = np.where(speed < threshold)
    low_speed = np.asanyarray(low_speed)
    low_speed_plus_one = low_speed + 1
    intersect = np.intersect1d(low_speed, low_speed_plus_one)
    stops = np.setdiff1d(low_speed, intersect)

    stops = remove_extra_stops(5, stops)
    return stops


def get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings):
    print('extracting stops...')
    stop_locations = []
    stop_trials = []
    stop_trial_types = []
    location = np.array(raw_position_data['x_position_cm'].tolist())
    trial_type = np.array(raw_position_data['trial_type'].tolist())
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    all_stops = np.asanyarray(all_stops)
    track_beginnings = np.asanyarray(track_beginnings)
    try:
        for trial in range(1,int(number_of_trials)-1):
            beginning = track_beginnings[trial]
            end = track_beginnings[trial + 1]
            all_stops = np.asanyarray(all_stops)
            stops_on_trial_indices = (np.where((beginning <= all_stops) & (all_stops <= end)))

            stops_on_trial = np.take(all_stops, stops_on_trial_indices)
            if len(stops_on_trial) > 0:
                stops = np.take(location, stops_on_trial)
                trial_types = np.take(trial_type, stops_on_trial)

                stop_locations=np.append(stop_locations,stops[0])
                stop_trial_types=np.append(stop_trial_types,trial_types[0])
                stop_trials=np.append(stop_trials,np.repeat(trial, len(stops[0])))
    except IndexError:
        print('indexerror')

    print('stops extracted')

    processed_position_data['stop_location_cm'] = pd.Series(stop_locations)
    processed_position_data['stop_trial_number'] = pd.Series(stop_trials)
    processed_position_data['stop_trial_type'] = pd.Series(stop_trial_types)
    return processed_position_data


def calculate_stops(raw_position_data,processed_position_data, threshold):
    all_stops = get_stop_times(raw_position_data,threshold)
    track_beginnings = get_beginning_of_track_positions(raw_position_data)
    processed_position_data = get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings)
    return processed_position_data


def calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory):
    stop_threshold = check_stop_threshold(recording_directory)
    stop_locations, stop_trials, stop_trial_types = calculate_stops(raw_position_data, processed_position_data, stop_threshold)
    processed_position_data['stop_location_cm'] = pd.Series(stop_locations)
    processed_position_data['stop_trial_number'] = pd.Series(stop_trials)
    processed_position_data['stop_trial_type'] = pd.Series(stop_trial_types)
    return processed_position_data


def find_first_stop_in_series(processed_position_data):
    stop_difference = np.array(processed_position_data['stop_location_cm'].diff())
    first_in_series_indices = np.where(stop_difference > 1)[0]
    print('Finding first stops in series')
    processed_position_data['first_series_location_cm'] = pd.Series(processed_position_data.stop_location_cm[first_in_series_indices].values)
    processed_position_data['first_series_trial_number'] = pd.Series(processed_position_data.stop_trial_number[first_in_series_indices].values)
    processed_position_data['first_series_trial_type'] = pd.Series(processed_position_data.stop_trial_type[first_in_series_indices].values)
    return processed_position_data


def take_first_reward_on_trial(rewarded_stop_locations,rewarded_trials):
    locations=[]
    trials=[]
    for tcount, trial in enumerate(np.unique(rewarded_trials)):
        trial_locations = np.take(rewarded_stop_locations, np.where(rewarded_trials == trial)[0])
        if len(trial_locations) ==1:
            locations = np.append(locations,trial_locations)
            trials = np.append(trials,trial)
        if len(trial_locations) >1:
            locations = np.append(locations,trial_locations[0])
            trials = np.append(trials,trial)
    return np.array(locations), np.array(trials)


def find_rewarded_positions(raw_position_data,processed_position_data):
    stop_locations = np.array(processed_position_data['first_series_location_cm'])
    stop_trials = np.array(processed_position_data['first_series_trial_number'])
    rewarded_stop_locations = np.take(stop_locations, np.where(np.logical_and(stop_locations >= 88, stop_locations < 110))[0])
    rewarded_trials = np.take(stop_trials, np.where(np.logical_and(stop_locations >= 88, stop_locations < 110))[0])

    locations, trials = take_first_reward_on_trial(rewarded_stop_locations, rewarded_trials)
    processed_position_data['rewarded_stop_locations'] = pd.Series(locations)
    processed_position_data['rewarded_trials'] = pd.Series(trials)
    return processed_position_data


def find_rewarded_positions_test(raw_position_data,processed_position_data):
    stop_locations = np.array(processed_position_data['stop_location_cm'])
    stop_trials = np.array(processed_position_data['stop_trial_number'])
    rewarded_stop_locations=[]
    rewarded_trials=[]
    for tcount, trial in enumerate(np.unique(stop_trials)):
            trial_locations = np.take(stop_locations, np.where(stop_trials == trial)[0])
            if len(trial_locations) > 0:
                for count in trial_locations:
                    if count >= 90 and count <= 110:
                        rewarded_stop_locations= np.append(rewarded_stop_locations, count)
                        rewarded_trials=np.append(rewarded_trials, trial)
                        break
    processed_position_data['rewarded_stop_locations'] = pd.Series(rewarded_stop_locations)
    processed_position_data['rewarded_trials'] = pd.Series(rewarded_trials)
    return processed_position_data


def get_bin_size(spatial_data):
    #bin_size_cm = 1
    track_length = spatial_data.x_position_cm.max()
    start_of_track = spatial_data.x_position_cm.min()
    #number_of_bins = (track_length - start_of_track)/bin_size_cm
    number_of_bins = 200
    bin_size_cm = (track_length - start_of_track)/number_of_bins
    bins = np.arange(start_of_track,track_length, 200)
    return bin_size_cm,number_of_bins, bins


def calculate_average_stops(raw_position_data,processed_position_data):
    stop_locations = processed_position_data.stop_location_cm.values
    stop_locations = stop_locations[~np.isnan(stop_locations)] #need to deal with
    bin_size_cm,number_of_bins, bins = get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
    for loc in range(int(number_of_bins)-1):
        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
        stops_in_bins[loc] = stops_in_bin

    processed_position_data['average_stops'] = pd.Series(stops_in_bins)
    processed_position_data['position_bins'] = pd.Series(range(int(number_of_bins)))
    return processed_position_data


def process_stops(raw_position_data,processed_position_data, prm):
    processed_position_data = calculate_stops(raw_position_data, processed_position_data, 10.7)
    processed_position_data = calculate_average_stops(raw_position_data,processed_position_data)
    gc.collect()
    processed_position_data = find_first_stop_in_series(processed_position_data)
    processed_position_data = find_rewarded_positions(raw_position_data,processed_position_data)
    return processed_position_data





def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25/' # test recording for plastic codes

    print('Processing ' + str(recording_folder))

    processed_position_data = pd.DataFrame()

    raw_position_data = Python_PostSorting.LoadDataFrames.process_raw_position_dir(recording_folder)

    process_stops(raw_position_data,processed_position_data)
    return raw_position_data






if __name__ == '__main__':
    main()





### --------------------------------------------------------------------------------------------------- ###

# average behaviour stops


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse1 = session_id.rsplit('_', 3)[0]
    day1 = session_id.rsplit('_', 3)[1]
    mouse = mouse1.rsplit('M', 3)[1]
    day = day1.rsplit('D', 3)[1]
    return int(day), int(mouse)


def create_histogram(spike_times, number_of_bins):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=(max(number_of_bins)/10)+1)
    values = np.array([[posrange[0], posrange[-1]]])
    H, bins = np.histogram(spike_times, bins=(posrange), range=values)
    return H


def calculate_avg_stop(spike_data):
    print('calculating first stop')
    spike_data["Average_Stop"] = ""
    bins = np.arange(0,201,1)

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)

        stop_locations=np.array(spike_data.loc[cluster].stop_location_cm)
        stop_hist = create_histogram(stop_locations, bins)
        trials=np.array(spike_data.loc[cluster].stop_trial_number)
        max_trial = np.nanmax(trials)
        averaged_stop_his = stop_hist/max_trial
        spike_data.at[cluster, 'Average_Stop'] = averaged_stop_his
        plot_stop_hist(averaged_stop_his, bins, day, mouse)

    #plot_average_stop_hist(averaged_stop_his, bins)
    #write_to_csv(averaged_stop_his)
    return spike_data



def create_2dhistogram(trials, locations, number_of_bins, trialrange):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=(max(number_of_bins)/10)+1)
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(trials, locations, bins=(trialrange, posrange), range=values)
    return H



def calculate_avg_stop_over_trials(spike_data):
    print('calculating stop hist')
    spike_data["Average_Stop"] = ""
    spike_data["Average_Stop_sd"] = ""
    bins = np.arange(0,201,1)

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)

        stop_locations=np.array(spike_data.loc[cluster].stop_location_cm)
        trials=np.array(spike_data.loc[cluster].stop_trial_number)
        unique_trials=np.unique(trials)
        stop_hist = create_2dhistogram(trials, stop_locations, bins, unique_trials)

        averaged_stop_his = np.nanmean(stop_hist, axis=0)
        sd_stop_his = stats.sem(stop_hist, axis=0)

        spike_data.at[cluster, 'Average_Stop'] = averaged_stop_his
        spike_data.at[cluster, 'Average_Stop_sd'] = sd_stop_his
        plot_stop_hist(averaged_stop_his, sd_stop_his, bins, day, mouse)

    #plot_average_stop_hist(averaged_stop_his, bins)
    #write_to_csv(averaged_stop_his)
    return spike_data



def calculate_avg_shuff_stop(spike_data):
    print('calculating first stop')
    spike_data["Average_shuffle_Stop"] = ""
    bins = np.arange(0,201,1)

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)

        stop_locations=np.array(spike_data.loc[cluster].shuffled_stops)
        stop_hist = create_histogram(stop_locations, bins)
        trials=np.array(spike_data.loc[cluster].stop_trial_number)
        max_trial = np.nanmax(trials)
        averaged_stop_his = stop_hist/max_trial
        spike_data.at[cluster, 'Average_shuffle_Stop'] = averaged_stop_his
        plot_shuff_stop_hist(spike_data, cluster, averaged_stop_his, bins, day, mouse)

    #write_to_csv(averaged_stop_his)
    return spike_data



def calculate_avg_shuff_stop_over_trials(spike_data):
    print('calculating first stop')
    spike_data["Average_shuffle_Stop"] = ""
    spike_data["Average_shuffle_Stop_sd"] = ""
    bins = np.arange(0,201,1)

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)

        stop_locations=np.array(spike_data.loc[cluster].shuffled_stops)
        trials=np.array(spike_data.loc[cluster].stop_trial_number)
        unique_trials=np.unique(trials)
        stop_hist = create_2dhistogram(trials, stop_locations, bins, unique_trials)

        averaged_stop_his = np.nanmean(stop_hist, axis=0)
        sd_stop_his = stats.sem(stop_hist, axis=0)

        spike_data.at[cluster, 'Average_shuffle_Stop'] = averaged_stop_his
        spike_data.at[cluster, 'Average_shuffle_Stop_sd'] = sd_stop_his
        plot_shuff_stop_hist(spike_data, cluster, averaged_stop_his, sd_stop_his, bins, day, mouse)

    #plot_average_stop_hist(averaged_stop_his, bins)
    #write_to_csv(averaged_stop_his)
    return spike_data


## plot individual stop histograms
def plot_stop_hist(array, sd_array, bins, day, mouse):
    print('plotting stop histogram...')

    stop_histogram = plt.figure(figsize=(4,2))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    #ax.bar(np.arange(0.5,20.5,1), array, color='Black')
    ax.plot(np.arange(0.5,20.5,1), array, '-',color='Black')
    ax.fill_between(np.arange(0.5,20.5,1), array-sd_array,array+sd_array, facecolor = 'Black', alpha = 0.2)
    plt.ylabel(' Average stops (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Location', fontsize=12, labelpad = 10)
    plt.xlim(-0.5,20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '0', '100', '200'])
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
    ax.axvspan(9, (9+2), facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(0, 3, facecolor='k', linewidth =0, alpha=.15) # black box
    ax.axvspan((20-3), 20, facecolor='k', linewidth =0, alpha=.15)# black box
    plt.ylim(0)

    ax.axvline(-0.5, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/average_stop_histogram_' + str(mouse) + '_' + str(day) + '_.png', dpi=200)
    plt.close()
    return



## plot individual stop histograms with shuffled stop
def plot_shuff_stop_hist(spike_data, cluster, array, sd_array, bins, day, mouse):
    print('plotting shuffled stop histogram...')

    real_stops = np.array(spike_data.at[cluster,"Average_Stop"])
    real_stops_sd = np.array(spike_data.at[cluster,"Average_Stop_sd"])

    stop_histogram = plt.figure(figsize=(4,2))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(np.arange(0.5,20.5,1), real_stops, '-', color='Black')
    ax.fill_between(np.arange(0.5,20.5,1), real_stops-real_stops_sd,real_stops+real_stops_sd, facecolor = 'Black', alpha = 0.2)
    ax.plot(np.arange(0.5,20.5,1), array, '-', color='Red')
    ax.fill_between(np.arange(0.5,20.5,1), array-sd_array,array+sd_array, facecolor = 'Red', alpha = 0.2)
    plt.ylabel(' Average stops (cm)', fontsize=12, labelpad = 10)
    plt.xlabel('Location', fontsize=12, labelpad = 10)
    plt.xlim(-0.5,20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '0', '100', '200'])
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
    ax.axvspan(9, (9+2), facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(0, 3, facecolor='k', linewidth =0, alpha=.15) # black box
    ax.axvspan((20-3), 20, facecolor='k', linewidth =0, alpha=.15)# black box
    plt.ylim(0)

    ax.axvline(-0.5, linewidth = 2.5, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 2.5, color = 'black') # bold line on the x axis
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig('/Users/sarahtennant/Work/Analysis/RampAnalysis/plots' + '/average_shuff_stop_histogram2_' + str(mouse) + '_' + str(day) + '_.png', dpi=200)
    plt.close()
    return


## Save average stops to .csv file
def write_to_csv(array):
    with open('/Users/sarahtennant/Work/Analysis/RampAnalysis/data/AverageStopHist_c1_M1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(array)
    csvFile.close()
    return


## Save shuffled stops to .csv file
def write_to_csv_shuffled(array):
    with open('/Users/sarahtennant/Work/Analysis/RampAnalysis/data/AverageShuffStopHist_c1_M1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(array)
    csvFile.close()
    return

