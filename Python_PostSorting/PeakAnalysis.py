
import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


def process_a_dir(df_path,output_path):

    if os.path.exists(output_path):
        print('I found the output folder.')

    os.path.isdir(df_path)
    if os.path.exists(df_path):
        print('I found a firing data frame.')
        peak_data = pd.read_pickle(df_path)
    return peak_data


def extract_lm_classified(peak_data):
    peak_data_beaconed = peak_data['trial_type'] == 'beaconed'
    peak_data_beaconed = peak_data[peak_data_beaconed]

    peak_data_outbound = peak_data_beaconed['peak_region'] == 'outbound'
    peak_data_outbound = peak_data_beaconed[peak_data_outbound]

    peak_data_max = peak_data_outbound['maxima_type'] == 'max'
    peak_data_max = peak_data_outbound[peak_data_max]

    peak_data_min = peak_data_outbound['maxima_type'] == 'min'
    peak_data_min = peak_data_outbound[ peak_data_min]

    cm_max_pos = peak_data_max['max_cm']
    cm_max_neg = peak_data_min['max_cm']

    cm_min_pos = peak_data_max['min_cm']
    cm_min_neg = peak_data_min['min_cm']
    return cm_max_pos, cm_max_neg, cm_min_pos, cm_min_neg


def plot_peak_data(peak_data, output_path):

    print('plotting stop histogram...')

    cm_max_pos, cm_max_neg, cm_min_pos, cm_min_neg = extract_lm_classified(peak_data)

    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(cm_max_pos, alpha=0.5, bins=int(15), color="DodgerBlue") #"violetred2", "chartreuse3"
    ax.hist(cm_min_pos, alpha=0.5, bins=int(15), color="Grey")
    plt.ylabel('Count', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    ax.axvspan(90, 110, facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(20, 30, facecolor='k', linewidth =0, alpha=.15) # black box
    plt.xlim(20,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True,
        labelsize=14)  # labels along the bottom edge are off

    plt.locator_params(axis = 'y', nbins  = 3)
    plt.locator_params(axis = 'x', nbins  = 3)
    ax.set_xticklabels(['20', '20', '70'])

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.17, right = 0.87, top = 0.92)
    plt.savefig(output_path + 'peak_histogram_positive' + '.png', dpi=200)
    plt.close()


    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(cm_max_neg, alpha=0.5, bins=int(15), color="DodgerBlue")
    ax.hist(cm_min_neg, alpha=0.5, bins=int(15), color="Grey")
    plt.ylabel('Count', fontsize=14, labelpad = 16)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 16)
    ax.axvspan(90, 110, facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(20, 30, facecolor='k', linewidth =0, alpha=.15) # black box
    plt.xlim(20,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True,
        labelsize=18)  # labels along the bottom edge are off
    plt.locator_params(axis = 'x', nbins  = 3)
    plt.locator_params(axis = 'y', nbins  = 3)
    ax.set_xticklabels(['20', '20', '70'])
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.16, right = 0.87, top = 0.92)
    plt.savefig(output_path + 'peak_histogram_negative' + '.png', dpi=200)
    plt.close()





def replot_peak_data(peak_data, output_path):

    print('plotting stop histogram...')

    cm_max_pos, cm_max_neg, cm_min_pos, cm_min_neg = extract_lm_classified(peak_data)

    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(cm_max_pos, alpha=0.5, bins=int(15), color="Chartreuse") #"violetred2", "chartreuse3"
    ax.hist(cm_max_neg, alpha=0.5, bins=int(15), color="HotPink")
    plt.ylabel('Count', fontsize=18, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
    ax.axvspan(90, 110, facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(20, 30, facecolor='k', linewidth =0, alpha=.15) # black box
    plt.xlim(20,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
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
        labelsize=18)  # labels along the bottom edge are off

    plt.locator_params(axis = 'y', nbins  = 3)
    plt.locator_params(axis = 'x', nbins  = 3)
    ax.set_xticklabels(['20', '20', '70'])

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.17, right = 0.87, top = 0.92)
    plt.savefig(output_path + 'peak_histogram_peak' + '.png', dpi=200)
    plt.close()


    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.hist(cm_min_pos, alpha=0.5, bins=int(15), color="Chartreuse")
    ax.hist(cm_min_neg, alpha=0.5, bins=int(15), color="HotPink")
    plt.ylabel('Count', fontsize=18, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
    ax.axvspan(90, 110, facecolor='DarkGreen', alpha=.15, linewidth =0)
    ax.axvspan(20, 30, facecolor='k', linewidth =0, alpha=.15) # black box
    plt.xlim(20,110)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
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
        labelsize=18)  # labels along the bottom edge are off
    plt.locator_params(axis = 'x', nbins  = 3)
    plt.locator_params(axis = 'y', nbins  = 3)
    ax.set_xticklabels(['20', '20', '70'])
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.16, right = 0.87, top = 0.92)
    plt.savefig(output_path + 'peak_histogram_trough' + '.png', dpi=200)
    plt.close()


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    df_path = '/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/peak_analysis-2.pkl'
    output_path = '/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/plots/'


    peak_data = process_a_dir(df_path,output_path)

    replot_peak_data(peak_data,output_path)

if __name__ == '__main__':
    main()
