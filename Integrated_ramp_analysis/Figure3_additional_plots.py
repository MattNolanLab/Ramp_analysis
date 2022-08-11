import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
import pyarrow.feather as feather
import pickle5 as pickle
import pickle
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#plt.style.use('ggplot')

def plot_histogram_by_trial_outcome(all_mice_behaviour, output_path):

    for HTR, HTR_time_binned_column, HTR_c in zip(["hit", "try", "run"], ["spikes_in_time_reward", "spikes_in_time_try", "spikes_in_time_run"], ["black", "blue", "red"]):

        speeds = []
        positions = []
        for index, row in all_mice_behaviour.iterrows():
            row = row.to_frame().T.reset_index(drop=True)
            speed=np.array(row[HTR_time_binned_column].iloc[0])[1,:]
            position=np.array(row[HTR_time_binned_column].iloc[0])[2,:]
            speeds.extend(speed)
            positions.extend(position)
        speeds = np.array(speeds)
        positions = np.array(positions)

        # subset for the bounds of the RZ
        speeds = speeds[(positions>90) & (positions<110)]

        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(4, 2.5, forward=True)
        ax = spikes_on_track.add_subplot(1, 1, 1)
        ax.hist(speeds, bins=30, range=[0,110], color=HTR_c, alpha=0.4, density=True)
        plt.xlim(-2.5, 110)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.locator_params(axis = 'x', nbins  = 5)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.savefig(output_path + "rewardzone_hist_"+HTR+".png", dpi=300)
        plt.close()

        print("there are ", str(len(speeds)), " samples in this measure for ", HTR)
    return

def plot_avg_speed_by_trial_outcome(all_mice_behaviour, output_path):
    all_mice_behaviour["cohort_mouse"] = np.char.add(np.asarray(all_mice_behaviour["cohort"]).astype(np.str0),
                                                     np.asarray(all_mice_behaviour["mouse_id"]).astype(np.str0))

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(4, 2.5, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    for HTR, HTR_time_binned_column, HTR_c in zip(["hit", "try", "run"], ["spikes_in_time_reward", "spikes_in_time_try", "spikes_in_time_run"], ["black", "blue", "red"]):
        avg_speeds = []
        for cohort_mouse in np.unique(all_mice_behaviour["cohort_mouse"]):
            mouse_behaviour = all_mice_behaviour[all_mice_behaviour["cohort_mouse"] == cohort_mouse]

            speeds = []
            positions = []
            for index, row in mouse_behaviour.iterrows():
                row = row.to_frame().T.reset_index(drop=True)
                speed=np.array(row[HTR_time_binned_column].iloc[0])[1,:]
                position=np.array(row[HTR_time_binned_column].iloc[0])[2,:]
                speeds.extend(speed)
                positions.extend(position)
            speeds = np.array(speeds)
            positions = np.array(positions)
            spatial_bins = np.arange(0, 200+1, 1) # 1 cm bins
            speed_space_bin_means = (np.histogram(positions, spatial_bins, weights = speeds)[0] /
                                     np.histogram(positions, spatial_bins)[0])
            avg_speeds.append(speed_space_bin_means.tolist())
        avg_speeds = np.array(avg_speeds)

        bin_centres = 0.5*(spatial_bins[1:]+spatial_bins[:-1])
        ax.plot(bin_centres, np.nanmean(avg_speeds, axis=0), color=HTR_c, alpha=0.7)
        ax.fill_between(bin_centres, np.nanmean(avg_speeds, axis=0)-stats.sem(avg_speeds, axis=0, nan_policy="omit"),
                        np.nanmean(avg_speeds, axis=0)+stats.sem(avg_speeds, axis=0, nan_policy="omit"), color=HTR_c, alpha=0.2, edgecolor="none")

    plt.xlim(0, 200)
    plt.ylim(0, 50)
    ax.axvspan(200-60-30-20, 200-60-30, facecolor=((69.0/255,139.0/255,0/255)), alpha=.2, linewidth =0, ymin=0, ymax=1)
    ax.axvspan(0, 30, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1) # black box
    ax.axvspan(200-30, 200, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1)# black box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,100,200])
    ax.set_xticklabels(["-30","70","170"])
    plt.locator_params(axis = 'x', nbins  = 5)
    plt.locator_params(axis = 'y', nbins  = 4)
    plt.savefig(output_path + "avg_speeds_"+HTR+".png", dpi=300)
    plt.close()


def make_behavioural_df(all_mice):
    session_ids = np.unique(all_mice.session_id)
    df = pd.DataFrame()
    for session_id in session_ids:
        session_df = all_mice[all_mice.session_id == session_id].head(1) # we only need one per session so take the first
        df = pd.concat([df, session_df], ignore_index=True)

    df = df[['session_id', 'mouse_id', 'cohort', 'spikes_in_time_reward', 'spikes_in_time_try', 'spikes_in_time_run']]
    return df

def add_unique_id(df):
    unique_ids = []
    for index, row in df.iterrows():
        data = row.to_frame().T.reset_index(drop=True)
        cluster_id = str(data["cluster_id"].iloc[0])
        session_id = data["session_id"].iloc[0]
        unique_id = session_id+"_"+cluster_id
        unique_ids.append(unique_id)
    df["unique_id"] = unique_ids
    return df

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # save all to a single dataframe
    all_mice = pd.DataFrame()
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_with_OF_unsmoothened.pkl")])
    all_mice = add_unique_id(all_mice)

    # only duplicate sessions
    all_mice = all_mice[all_mice.unique_id.isin(all_mice["unique_id"])]

    # make behavioural dataframe for plotting speed profiles
    all_mice_behaviour = make_behavioural_df(all_mice)

    # plot histogram and speed profile for speeds binned in time for hit miss and try trials
    plot_histogram_by_trial_outcome(all_mice_behaviour, output_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/behaviour/")
    plot_avg_speed_by_trial_outcome(all_mice_behaviour, output_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/behaviour/")

if __name__ == '__main__':
    main()
