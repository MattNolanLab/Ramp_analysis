import csv
import glob
import numpy as np
import os
import pandas as pd
import Python_PostSorting.math_utility
import matplotlib.pylab as plt


import Python_PostSorting.postsorting_parameters

''' The name of bonsai output files is not standardised in all experiments, so this function checks all csv
files in the recording folder and reads the first line. Our bonsai output files start with the date and 'T'
for example: 2017-11-21T
One recording folder has no more than one bonsai file, so this is sufficient for identification.
'''


def find_bonsai_file(recording_folder):
    if os.path.isdir(recording_folder) is False:
        print('    Error in open_field_spatial_data.py : The folder you specified as recording folder does not exist, please check if the path is correct.')
    path_to_bonsai_file = ''
    is_found = False
    for name in glob.glob(recording_folder + '/*.csv'):
        if os.path.exists(name):
            with open(name, newline='') as file:
                try:
                    reader = csv.reader(file)
                    row1 = next(reader)
                    if "T" not in row1[0]:
                        continue
                    else:
                        if len(row1[0].split('T')[0]) == 10:
                            path_to_bonsai_file = name
                            is_found = True
                except Exception as ex:
                    print('Could not read csv file:')
                    print(name)
                    print(ex)

    return path_to_bonsai_file, is_found



def read_bonsai_file(recording_folder):
    if os.path.isdir(recording_folder) is False:
        print('    Error in open_field_spatial_data.py : The folder you specified as recording folder does not exist, please check if the path is correct.')
    path_to_bonsai_file = ''
    is_found = False
    for name in glob.glob(recording_folder + '/*.csv'):
        if os.path.exists(name):
            with open(name, newline='') as file:
                try:
                    reader = csv.reader(file)
                    row1 = next(reader)
                    if "T" not in row1[0]:
                        continue
                    else:
                        if len(row1[0].split('T')[0]) == 10:
                            path_to_bonsai_file = name
                            is_found = True
                except Exception as ex:
                    print('Could not read csv file:')
                    print(name)
                    print(ex)

    return path_to_bonsai_file, is_found


''' 
Read raw position data and sync LED intensity from Bonsai file amd convert time to seconds
'''


def convert_time_to_seconds(position_data):
    position_data['hours'], position_data['minutes'], position_data['seconds'] = position_data['time'].str.split(':', 2).str
    position_data['hours'] = position_data['hours'].astype(int)
    position_data['minutes'] = position_data['minutes'].astype(int)
    position_data['seconds'] = position_data['seconds'].astype(float)
    position_data['time_seconds'] = position_data['hours'] * 3600 + position_data['minutes']*60 + position_data['seconds']
    position_data['time_seconds'] = position_data['time_seconds'] - position_data['time_seconds'][0]
    return position_data


def read_position(path_to_bonsai_file):
    position_data = pd.read_csv(path_to_bonsai_file, sep=' ', header=None)
    if len(list(position_data)) > 6:
        position_data = position_data.drop([6], axis=1)  # remove column of NaNs due to extra space at end of lines
    position_data['date'], position_data['time'] = position_data[0].str.split('T', 1).str

    position_data['time'], position_data['str_to_remove'] = position_data['time'].str.split('+', 1).str
    position_data = position_data.drop([0, 'str_to_remove'], axis=1)  # remove first column that got split into date and time

    position_data.columns = ['x_left', 'y_left', 'x_right', 'y_right','syncLED', 'date', 'time']
    position_data = convert_time_to_seconds(position_data)
    return position_data


def calculate_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['x_left'].diff().pow(2))
    position_data['speed_left'] = distance_travelled / elapsed_time
    return position_data


def calculate_central_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2))
    position_data['speed'] = distance_travelled / elapsed_time
    return position_data


def remove_jumps(position_data, prm):
    max_speed = 1  # m/s, anything above this is not realistic
    pixel_ratio = prm.get_pixel_ratio()
    max_speed_pixels = max_speed * pixel_ratio
    speed_exceeded_left = position_data['speed_left'] > max_speed_pixels
    position_data['x_left_without_jumps'] = position_data.x_left[speed_exceeded_left == False]
    position_data['y_left_without_jumps'] = position_data.y_left[speed_exceeded_left == False]
    return position_data


def curate_position(position_data, prm):
    position_data = remove_jumps(position_data, prm)
    return position_data


def calculate_position(position_data):
    position_data['position_x_tmp'] = position_data['x_left_without_jumps']
    position_data['position_y_tmp'] = position_data['y_left_without_jumps']

    position_data['position_x'] = position_data['position_x_tmp'].interpolate()  # interpolate missing data
    position_data['position_y'] = position_data['position_y_tmp'].interpolate()
    return position_data



def convert_to_cm(position_data, params):
    pixel_ratio = params.get_pixel_ratio()
    position_data['position_x_pixels'] = position_data.position_x
    position_data['position_y_pixels'] = position_data.position_y
    position_data['position_x'] = position_data.position_x / pixel_ratio * 100
    position_data['position_y'] = position_data.position_y / pixel_ratio * 100
    return position_data


def shift_to_start_from_zero_at_bottom_left(position_data):
    # this is copied from MATLAB script, 0.0001 is here to 'avoid bin zero in first point'
    position_data['position_x'] = position_data.position_x - min(position_data.position_x[~np.isnan(position_data.position_x)])
    position_data['position_y'] = position_data.position_y - min(position_data.position_y[~np.isnan(position_data.position_y)])
    return position_data


def get_sides(position_data):
    left_side_edge = position_data.position_x.round().min()
    right_side_edge = position_data.position_x.round().max()
    top_side_edge = position_data.position_y.round().max()
    bottom_side_edge = position_data.position_y.round().min()
    return left_side_edge, right_side_edge, top_side_edge, bottom_side_edge


def remove_edge_from_horizontal_side(position_data, left_side_edge, right_side_edge):
    points_on_left_edge = np.where(position_data.position_x < (left_side_edge + 1))[0]
    number_of_points_on_left_edge = len(points_on_left_edge)
    points_on_right_edge = np.where(position_data.position_x > (right_side_edge - 1))[0]
    number_of_points_on_right_edge = len(points_on_right_edge)

    if number_of_points_on_left_edge > number_of_points_on_right_edge:
        # remove left edge
        position_data = position_data.drop(position_data.index[points_on_right_edge])
    else:
        # remove right edge
        position_data = position_data.drop(position_data.index[points_on_left_edge])
    return position_data


def remove_edge_from_vertical_side(position_data, top_side_edge, bottom_side_edge):
    points_on_top_edge = np.where(position_data.position_y > (top_side_edge - 1))[0]
    number_of_points_on_top_edge = len(points_on_top_edge)
    points_on_bottom_edge = np.where(position_data.position_y < (bottom_side_edge + 1))[0]
    number_of_points_on_bottom_edge = len(points_on_bottom_edge)

    if number_of_points_on_top_edge > number_of_points_on_bottom_edge:
        # remove left edge
        position_data = position_data.drop(position_data.index[points_on_bottom_edge])
    else:
        # remove right edge
        position_data = position_data.drop(position_data.index[points_on_top_edge])
    return position_data


def get_dimensions_of_arena(position_data):
    left_side_edge, right_side_edge, top_side_edge, bottom_side_edge = get_sides(position_data)
    x_length = right_side_edge - left_side_edge
    y_length = top_side_edge - bottom_side_edge
    return x_length, y_length


def remove_position_outlier_rows(position_data):
    is_square = False
    x_length, y_length = get_dimensions_of_arena(position_data)
    while is_square is False:
        left_side_edge, right_side_edge, top_side_edge, bottom_side_edge = get_sides(position_data)
        if x_length == y_length:
            is_square = True
        elif x_length > y_length:
            position_data = remove_edge_from_horizontal_side(position_data, left_side_edge, right_side_edge)
        else:
            position_data = remove_edge_from_vertical_side(position_data, top_side_edge, bottom_side_edge)
        x_length, y_length = get_dimensions_of_arena(position_data)
    return position_data


def plot_behavioural_figures(position_data, save_path):
    # plot position
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    x = position_data["position_x_pixels"]
    y = position_data["position_y_pixels"]
    ax.plot(x, y)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/position_pixels' + '.png', dpi=200)
    plt.close()
    # plot position
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    x = position_data["x_left"]
    y = position_data["y_left"]
    ax.plot(x, y)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/position' + '.png', dpi=200)
    plt.close()



def process_position_data(recording_folder, params):
    position_of_mouse = None
    path_to_position_file, is_found = read_bonsai_file(recording_folder)
    if is_found:
        position_data = read_position(path_to_position_file)  # raw position data from bonsai output
    if not is_found:
        if os.path.isfile(recording_folder + '/axona_position.pkl'):
            position_data = pd.read_pickle(recording_folder + '/axona_position.pkl')
            is_found = True
    if is_found:
        position_data = calculate_speed(position_data)
        position_data = curate_position(position_data, params)  # remove jumps from data, and when the beads are far apart
        position_data = calculate_position(position_data)  # get central position and interpolate missing data
        position_data = shift_to_start_from_zero_at_bottom_left(position_data)
        # position_data = remove_position_outlier_rows(position_data)
        position_data = convert_to_cm(position_data, params)
        position_data = calculate_central_speed(position_data)
        position_of_mouse = position_data[['time_seconds', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'syncLED', 'speed']].copy()
        plot_behavioural_figures(position_data, recording_folder)
    return position_of_mouse, is_found


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = Python_PostSorting.postsorting_parameters.Parameters()
    params.set_pixel_ratio(440)
    params.set_sorter_name('MountainSort')

    recording_folder = '/Users/sarahtennant/Work/Analysis/Bonsia/Test/' # test recording
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M0_2017-11-21_15-52-53'
    position_of_mouse = process_position_data(recording_folder, params)


if __name__ == '__main__':
    main()
