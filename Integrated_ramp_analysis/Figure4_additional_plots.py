import pandas as pd
import numpy as np
import scipy.stats as stats
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour
import warnings
from astropy.convolution import convolve, Gaussian1DKernel
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#plt.style.use('ggplot')

def plot_avg_stop_histogram(all_mice_behaviour, output_path, smooth=True):
    all_mice_behaviour["cohort_mouse"] = np.char.add(np.asarray(all_mice_behaviour["cohort"]).astype(np.str0),
                                                     np.asarray(all_mice_behaviour["mouse_id"]).astype(np.str0))
    start=2; end=197
    gauss_kernel = Gaussian1DKernel(2)
    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(4, 2.5, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    for tt, tt_c, stop_column in zip([0,2], [(0/255,0/255,0/255),(31/255, 181/255, 178/255)], ["average_stops", "average_stops_p"]):
        mouse_avg_stops = []
        for cohort_mouse in np.unique(all_mice_behaviour["cohort_mouse"]):
            mouse_behaviour = all_mice_behaviour[all_mice_behaviour["cohort_mouse"] == cohort_mouse]

            avg_stops = []
            for index, row in mouse_behaviour.iterrows():
                row = row.to_frame().T.reset_index(drop=True)
                avg_stops_cluster=np.array(row[stop_column].iloc[0])
                avg_stops.append(avg_stops_cluster.tolist())

            avg_stops = np.array(avg_stops)
            avg_stops = np.nanmean(avg_stops, axis=0)
            if smooth:
                avg_stops = convolve(avg_stops, gauss_kernel)

            mouse_avg_stops.append(avg_stops.tolist())
        mouse_avg_stops = np.array(mouse_avg_stops)

        spatial_bins = np.arange(0, 200+1, 1) # 1 cm bins
        bin_centres = 0.5*(spatial_bins[1:]+spatial_bins[:-1])
        ax.plot(bin_centres[start:end], np.nanmean(mouse_avg_stops, axis=0)[start:end], color=tt_c, alpha=1)
        ax.fill_between(bin_centres[start:end], np.nanmean(mouse_avg_stops, axis=0)[start:end]-stats.sem(mouse_avg_stops, axis=0, nan_policy="omit")[start:end],
                        np.nanmean(mouse_avg_stops, axis=0)[start:end]+stats.sem(mouse_avg_stops, axis=0, nan_policy="omit")[start:end], color=tt_c, alpha=0.3, edgecolor="none")

    plt.xlim(0, 200)
    ax.axvspan(200-60-30-20, 200-60-30, facecolor=((69.0/255,139.0/255,0/255)), alpha=.2, linewidth =0, ymin=0, ymax=1)
    ax.axvspan(0, 30, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1) # black box
    ax.axvspan(200-30, 200, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1)# black box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,100,200])
    ax.set_xticklabels(["-30","70","170"])
    plt.locator_params(axis = 'x', nbins  = 5)
    plt.locator_params(axis = 'y', nbins  = 4)
    plt.savefig(output_path + "avg_stops_by_mouse.png", dpi=300)
    plt.close()

def plot_avg_first_stop_histogram(all_mice_behaviour, output_path, smooth=True):
    all_mice_behaviour["cohort_mouse"] = np.char.add(np.asarray(all_mice_behaviour["cohort"]).astype(np.str0),
                                                     np.asarray(all_mice_behaviour["mouse_id"]).astype(np.str0))
    start=2; end=197
    gauss_kernel = Gaussian1DKernel(2)
    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(4, 2.5, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    for tt, tt_c, stop_column in zip([0,2], ["Black", "Blue"], ["average_first_stops", "average_first_stops_p"]):
        mouse_avg_stops = []
        for cohort_mouse in np.unique(all_mice_behaviour["cohort_mouse"]):
            mouse_behaviour = all_mice_behaviour[all_mice_behaviour["cohort_mouse"] == cohort_mouse]

            avg_stops = []
            for index, row in mouse_behaviour.iterrows():
                row = row.to_frame().T.reset_index(drop=True)
                avg_stops_cluster=np.array(row[stop_column].iloc[0])
                avg_stops.append(avg_stops_cluster.tolist())

            avg_stops = np.array(avg_stops)
            avg_stops = np.nanmean(avg_stops, axis=0)
            if smooth:
                avg_stops = convolve(avg_stops, gauss_kernel)
            mouse_avg_stops.append(avg_stops.tolist())
        mouse_avg_stops = np.array(mouse_avg_stops)

        spatial_bins = np.arange(0, 200+1, 1) # 1 cm bins
        bin_centres = 0.5*(spatial_bins[1:]+spatial_bins[:-1])
        ax.plot(bin_centres[start:end], np.nanmean(mouse_avg_stops, axis=0)[start:end], color=tt_c, alpha=1)
        ax.fill_between(bin_centres[start:end], np.nanmean(mouse_avg_stops, axis=0)[start:end]-stats.sem(mouse_avg_stops, axis=0, nan_policy="omit")[start:end],
                        np.nanmean(mouse_avg_stops, axis=0)[start:end]+stats.sem(mouse_avg_stops, axis=0, nan_policy="omit")[start:end], color=tt_c, alpha=0.3, edgecolor="none")

    plt.xlim(0, 200)
    ax.axvspan(200-60-30-20, 200-60-30, facecolor=((69.0/255,139.0/255,0/255)), alpha=.2, linewidth =0, ymin=0, ymax=1)
    ax.axvspan(0, 30, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1) # black box
    ax.axvspan(200-30, 200, facecolor=((153.0/255,153.0/255,153.0/255)), linewidth =0, alpha=.2, ymin=0, ymax=1)# black box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,100,200])
    ax.set_xticklabels(["-30","70","170"])
    plt.locator_params(axis = 'x', nbins  = 5)
    plt.locator_params(axis = 'y', nbins  = 4)
    plt.savefig(output_path + "avg_first_stops_by_mouse.png", dpi=300)
    plt.close()

def make_behavioural_df(all_mice, take_head=True):
    session_ids = np.unique(all_mice.session_id)
    df = pd.DataFrame()
    for session_id in session_ids:
        #print(session_id)
        if take_head:
            session_df = all_mice[all_mice.session_id == session_id].head(1) # we only need one per session so take the first
        else:
            session_df = all_mice

        session_df.reset_index(drop=True, inplace=True)
        session_df = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.curate_stops(session_df)
        session_df = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_stops(session_df)
        session_df = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Plot_Behaviour.calculate_average_nonbeaconed_stops(session_df)
        df = pd.concat([df, session_df], ignore_index=True)

    df = df[['session_id', 'mouse_id', 'cohort', 'average_stops', 'average_stops_nb', 'average_stops_p', 'average_first_stops', 'average_first_stops_p', 'average_first_stops_nb']]
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

    all_mice = pd.DataFrame()
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_with_OF_unsmoothened.pkl")])
    all_mice = pd.concat([all_mice, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_with_OF_unsmoothened.pkl")])
    all_mice = add_unique_id(all_mice)

    # only duplicate sessions
    all_mice = all_mice[all_mice.unique_id.isin(all_mice["unique_id"])]

    # make behavioural dataframe for plotting stops
    all_mice_behaviour = make_behavioural_df(all_mice, take_head=False)

    # plot histogram for stop of beaconed and probe trials
    plot_avg_stop_histogram(all_mice_behaviour, output_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/behaviour/", smooth=False)
    plot_avg_first_stop_histogram(all_mice_behaviour, output_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/behaviour/", smooth=False)

if __name__ == '__main__':
    main()
