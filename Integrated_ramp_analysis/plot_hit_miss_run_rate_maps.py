import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from scipy import interpolate
from scipy import stats
import matplotlib.colors as colors
import sys
import scipy
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
from Edmond.VR_grid_analysis.vr_grid_cells import *
import settings
import matplotlib.pylab as plt
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from Edmond.utility_functions.array_manipulations import *

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Split_DataByTrialOutcome


def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length


def hmt2numeric(hmt):
    # takes numpy array of "hit", "miss" and "try" and translates them into 1, 0 and 0.5 otherwise nan
    hmt_numeric = []
    for i in range(len(hmt)):
        if hmt[i] == "hit":
            numeric = 1
        elif hmt[i] == "try":
            numeric = 0.5
        elif hmt[i] == "miss":
            numeric = 0
        else:
            numeric = np.nan
        hmt_numeric.append(numeric)
    return np.array(hmt_numeric)

def hmt2color(hmt):
    # takes numpy array of "hit", "miss" and "try" and translates them into 1, 0 and 0.5 otherwise nan
    hmt_colors = []
    for i in range(len(hmt)):
        if hmt[i] == "hit":
            color = "green"
        elif hmt[i] == "try":
            color = "orange"
        elif hmt[i] == "miss":
            color = "red"
        else:
            color = np.nan
        hmt_colors.append(color)
    return np.array(hmt_colors)

def get_trial_type_colors(trial_types):
    # takes numpy array of 0, 1 and 2 and translates them into black, red and blue
    type_colors = []
    for i in range(len(trial_types)):
        if trial_types[i] == 0: # beaconed
            color_i = "black"
        elif trial_types[i] == 1: # non-beaconed
            color_i = "red"
        elif trial_types[i] == 2: # probe
            color_i = "blue"
        else:
            print("do nothing")
        type_colors.append(color_i)

    return np.array(type_colors)



def get_hmt_color(hmt):
    if hmt == "hit":
        return "green"
    elif hmt == "miss":
        return "red"
    elif hmt == "try":
        return "orange"
    else:
        return "black"

def plot_stops_on_track(processed_position_data, output_path, track_length=200):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)
        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color="black", markersize=4)

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_raster.png', dpi=200)
    plt.close()

def plot_speed_per_trial(processed_position_data, output_path, track_length=200):
    print('plotting speed heatmap...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    x_max = len(processed_position_data)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = Edmond.plot_utility2.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])
    where_are_NaNs = np.isnan(trial_speeds)
    trial_speeds[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_speeds[0]))
    ordered = np.arange(0, len(trial_speeds), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("jet")
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14)
    cbar.mappable.set_clim(0, 100)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0,100])
    cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Speed (cm/s)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_heat_map' + '.png', dpi=200)
    plt.close()

def plot_avg_speed_in_rz_hist(processed_position_data, output_path, percentile_speed):
    print('I am plotting avg speed histogram...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    g = colors.colorConverter.to_rgb("green")
    r = colors.colorConverter.to_rgb("red")
    o = colors.colorConverter.to_rgb("orange")

    fig, axes = plt.subplots(2, 1, figsize=(6,4), sharex=True)

    hits = processed_position_data[processed_position_data["hit_miss_try"] == "hit"]
    misses = processed_position_data[processed_position_data["hit_miss_try"] == "miss"]
    tries = processed_position_data[processed_position_data["hit_miss_try"] == "try"]

    axes[0].hist(pandas_collumn_to_numpy_array(hits["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(g[0],g[1],g[2], 0.3), edgecolor=(g[0],g[1],g[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)
    axes[1].hist(pandas_collumn_to_numpy_array(tries["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(o[0],o[1],o[2], 0.3), edgecolor=(o[0],o[1],o[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)
    axes[1].hist(pandas_collumn_to_numpy_array(misses["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(r[0],r[1],r[2], 0.3), edgecolor=(r[0],r[1],r[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)

    #plt.ylabel('Trial', fontsize=20, labelpad = 10)
    plt.xlabel('Avg Speed in RZ (cm/s)', fontsize=20, labelpad = 10)
    plt.xlim(0,100)
    tick_spacing = 50
    axes[0].axvline(x=percentile_speed, color="red", linestyle="dotted", linewidth=4)
    axes[1].axvline(x=percentile_speed, color="red", linestyle="dotted", linewidth=4)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    axes[0].set_yticks([0, 15, 30])
    axes[1].set_yticks([0, 15, 30])
    fig.text(0.2, 0.5, "       Trials", va='center', rotation='vertical', fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/avg_speed_hist' + '.png', dpi=200)
    plt.close()



def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def correct_for_time_binned_teleport(trial_pos_in_time, track_length):
    # check if any of the first 5 or last 5 bins are too high or too low respectively
    first_5 = trial_pos_in_time[:5]
    last_5 = trial_pos_in_time[-5:]

    first_5[first_5>(track_length/2)] = first_5[first_5>(track_length/2)]-track_length
    last_5[last_5<(track_length/2)] = last_5[last_5<(track_length/2)]+track_length

    trial_pos_in_time[:5] = first_5
    trial_pos_in_time[-5:] = last_5
    return trial_pos_in_time

def plot_speed_histogram_with_error(processed_position_data, output_path, track_length=200, tt="", hmt=""):
    subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt) &
                                                             (processed_position_data["hit_miss_try"] == hmt)]
    if len(subset_processed_position_data)>0:
        trial_speeds = pandas_collumn_to_2d_numpy_array(subset_processed_position_data["speeds_binned_in_space"])
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
        trial_speeds_sem = scipy.stats.sem(trial_speeds, axis=0, nan_policy="omit")
        trial_speeds_avg = np.nanmean(trial_speeds, axis=0)

        print('plotting avg speeds')
        save_path = output_path + '/Figures/behaviour'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        # to plot by trial using the time binned data we need the n-1, n and n+1 trials so we can plot around the track limits
        # here we extract the n-1, n and n+1 trials, correct for any time binned teleports and concatenated the positions and speeds for each trial
        for i, tn in enumerate(processed_position_data["trial_number"]):
            trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == tn]
            tt_trial = trial_processed_position_data["trial_type"].iloc[0]
            hmt_trial = trial_processed_position_data["hit_miss_try"].iloc[0]
            trial_speeds_in_time = np.asarray(trial_processed_position_data['speeds_binned_in_time'].iloc[0])
            trial_pos_in_time = np.asarray(trial_processed_position_data['pos_binned_in_time'].iloc[0])

            # cases above trial number 1
            if tn != min(processed_position_data["trial_number"]):
                trial_processed_position_data_1down = processed_position_data[processed_position_data["trial_number"] == tn-1]
                trial_speeds_in_time_1down = np.asarray(trial_processed_position_data_1down['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1down = np.asarray(trial_processed_position_data_1down['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1down = np.array([])
                trial_pos_in_time_1down = np.array([])

            # cases below trial number n
            if tn != max(processed_position_data["trial_number"]):
                trial_processed_position_data_1up = processed_position_data[processed_position_data["trial_number"] == tn+1]
                trial_speeds_in_time_1up = np.asarray(trial_processed_position_data_1up['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1up = np.asarray(trial_processed_position_data_1up['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1up = np.array([])
                trial_pos_in_time_1up = np.array([])

            trial_pos_in_time = np.concatenate((trial_pos_in_time_1down[-2:], trial_pos_in_time, trial_pos_in_time_1up[:2]))
            trial_speeds_in_time = np.concatenate((trial_speeds_in_time_1down[-2:], trial_speeds_in_time, trial_speeds_in_time_1up[:2]))

            if tt_trial == tt and hmt_trial == hmt:
                trial_pos_in_time = correct_for_time_binned_teleport(trial_pos_in_time, track_length)
                ax.plot(trial_pos_in_time, trial_speeds_in_time, color="grey", alpha=0.4)

        ax.plot(bin_centres, trial_speeds_avg, color=get_hmt_color(hmt), linewidth=4)
        ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
        plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        plt.xlim(0,track_length)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        if tt == 0:
            style_track_plot(ax, track_length)
        else:
            style_track_plot_no_RZ(ax, track_length)
        tick_spacing = 100
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        x_max = max(trial_speeds_avg+trial_speeds_sem)
        x_max = 115
        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(output_path + '/Figures/behaviour/trial_speeds_tt_'+str(tt)+"_"+hmt+'.png', dpi=300)
        plt.close()

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

def replace_leading_NaN(Data):
    nansIndx = np.where(np.isnan(Data))[0]
    isanIndx = np.where(~np.isnan(Data))[0]
    for nan in nansIndx:
        replacementCandidates = np.where(isanIndx>nan)[0]
        if replacementCandidates.size != 0:
            replacement = Data[isanIndx[replacementCandidates[0]]]
        else:
            replacement = Data[isanIndx[np.where(isanIndx<nan)[0][-1]]]
        Data[nan] = replacement
    return Data

def plot_speed_histogram_with_error_all_trials2(processed_position_data, output_path, track_length):
    print('plotting avg speeds')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    gauss_kernel = Gaussian1DKernel(3)

    # derive trial speeds from time binned data like how hit try and run trials are classified
    speeds=[]
    positions=[]
    trial_numbers=[]
    for i, tn in enumerate(processed_position_data["trial_number"]):
        speeds.extend(processed_position_data["speeds_binned_in_time_not_smoothened"].iloc[i])
        positions.extend(processed_position_data["pos_binned_in_time_not_smoothened"].iloc[i])
        trial_numbers.extend((tn*np.ones(len(processed_position_data["speeds_binned_in_time_not_smoothened"].iloc[i]))).tolist())
    speeds = np.array(speeds)
    positions = np.array(positions)
    trial_numbers = np.array(trial_numbers)
    speeds = convolve(speeds, gauss_kernel)
    elapsed_position = (200*(trial_numbers-1))+positions
    elapsed_position = convolve(elapsed_position, gauss_kernel)
    number_of_trials=len(np.unique(trial_numbers))
    number_of_bins = 200
    spatial_bins = np.arange(0, (number_of_trials*200)+1, 1) # 1 cm bins
    speed_in_bins_numerator, bin_edges = np.histogram(elapsed_position, spatial_bins, weights=speeds)
    speed_in_bins_denominator, bin_edges = np.histogram(elapsed_position, spatial_bins)
    speed_space_bin_means = speed_in_bins_numerator/speed_in_bins_denominator
    tn_space_bin_means = (((0.5*(spatial_bins[1:]+spatial_bins[:-1]))//200)+1).astype(np.int64)
    # create empty array
    speed_trials = np.zeros((number_of_trials, number_of_bins)); i=0
    for trial_number in range(1, number_of_trials+1):
        speed_trials[i, :] = speed_space_bin_means[tn_space_bin_means == trial_number]
        i+=1

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for hmt, c in zip(["hit", "try", "miss"], ["black", "blue", "red"]):
        hmt_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) & (processed_position_data["trial_type"] == 0)]
        hmt_trial_numbers = np.array(hmt_processed_position_data["trial_number"])
        if len(hmt_processed_position_data)>1:
            hmt_speed_trials = speed_trials[hmt_trial_numbers-1]

            #for i in range(len(hmt_speed_trials)):
            #    ax.plot(np.arange(0.5, 200.5, 1), hmt_speed_trials[i], '-', color=c, alpha=0.3)

            hmt_average_speeds = np.nanmean(hmt_speed_trials, axis=0)
            hmt_se_speeds = stats.sem(hmt_speed_trials, axis=0, nan_policy="omit")
            ax.plot(np.arange(0.5, 200.5, 1), hmt_average_speeds, '-', color=c)
            ax.fill_between(np.arange(0.5, 200.5, 1), hmt_average_speeds-hmt_se_speeds, hmt_average_speeds+hmt_se_speeds, edgecolor="none", color=c, alpha=0.2)

    plt.ylabel('Speed (cm/s)', fontsize=19, labelpad = 0)
    plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
    plt.xlim(0,track_length)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, track_length)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    ax.set_ylim(0)
    plt.locator_params(axis = 'x', nbins  = 3)
    plt.locator_params(axis = 'y', nbins  = 4)
    ax.set_xticklabels(['-30', '70', '170'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(output_path + '/Figures/behaviour/trial_speeds_hmt_beaconed.png', dpi=300)
    plt.close()

def plot_speed_histogram_with_error_all_trials(processed_position_data, output_path, track_length):
    print('plotting avg speeds')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    start_i = 1
    end_i = track_length-1
    trial_speeds = pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])

    avg_spikes_on_track = plt.figure(figsize=(3.7,3))
    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for hmt, c in zip(["hit", "try", "miss"], ["black", "blue", "red"]):
        hmt_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) & (processed_position_data["trial_type"] == 0)]
        hmt_trial_numbers = np.array(hmt_processed_position_data["trial_number"])

        if len(hmt_processed_position_data)>1:
            hmt_trial_speeds = trial_speeds[hmt_trial_numbers-1]

            bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
            trial_speeds_sem = scipy.stats.sem(hmt_trial_speeds, axis=0, nan_policy="omit")
            trial_speeds_avg = np.nanmean(hmt_trial_speeds, axis=0)

            ax.plot(bin_centres[start_i:end_i], trial_speeds_avg[start_i:end_i], color=c, linewidth=1)
            ax.fill_between(bin_centres[start_i:end_i], trial_speeds_avg[start_i:end_i]+trial_speeds_sem[start_i:end_i], trial_speeds_avg[start_i:end_i]-trial_speeds_sem[start_i:end_i], color=c, alpha=0.2, linewidth=0.0)

    ax.axhline(y=4.7, linewidth=2, linestyle="dashed", color="black")
    plt.ylabel('Speed (cm/s)', fontsize=19, labelpad = 0)
    plt.xlabel('Location (cm)', fontsize=18, labelpad = 10)
    plt.xlim(0,track_length)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_track_plot(ax, track_length)
    Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.plot_utility.style_vr_plot(ax)
    ax.set_ylim(bottom=0, top=50)
    plt.locator_params(axis = 'x', nbins  = 3)
    plt.locator_params(axis = 'y', nbins  = 4)
    ax.set_xticklabels(['-30', '70', '170'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(output_path + '/Figures/behaviour/trial_speeds_hmt_beaconed.png', dpi=300)
    plt.close()

def add_session_number(df):
    session_number = []
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        D = str(session_id.split("_")[1].split("D")[-1])
        session_number.append(D)
    df["session_number"] = session_number
    return df


def plot_firing_rate_maps_hmt(spike_data, processed_position_data, output_path, track_length, trial_type=0):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_hmr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == trial_type)]

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            fr_binned_in_space = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
            fr_binned_in_space[np.isnan(fr_binned_in_space)] = 0
            fr_binned_in_space[np.isinf(fr_binned_in_space)] = 0
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]

            avg_spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            for hmt, c in zip(["hit", "try", "miss"], ["black", "blue", "red"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                hmt_trial_numbers = np.asarray(hmt_processed_position_data["trial_number"])
                hmt_fr_binned_in_space = fr_binned_in_space[hmt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(hmt_fr_binned_in_space, axis=0)-stats.sem(hmt_fr_binned_in_space, axis=0), np.nanmean(hmt_fr_binned_in_space, axis=0)+stats.sem(hmt_fr_binned_in_space, axis=0), color=c, alpha=0.2)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(hmt_fr_binned_in_space, axis=0), color=c)

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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_by_trial_outcome_' + str(cluster_id) + '_tt_'+str(trial_type)+'.png', dpi=300)
            plt.close()

def plot_firing_rate_maps_tt(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_hmr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            fr_binned_in_space = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
            fr_binned_in_space[np.isnan(fr_binned_in_space)] = 0
            fr_binned_in_space[np.isinf(fr_binned_in_space)] = 0
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]

            avg_spikes_on_track = plt.figure(figsize=(3.7,3))
            ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            for tt, c in zip([0, 2], ["black", "#1FB5B2"]):
                processed_position_data_tt = processed_position_data[(processed_position_data["trial_type"] == tt)]
                tt_trial_numbers = np.asarray(processed_position_data_tt["trial_number"])
                tt_fr_binned_in_space = fr_binned_in_space[tt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0)-stats.sem(tt_fr_binned_in_space, axis=0), np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0), color=c, alpha=0.2)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0), color=c)
                #hmt_max = max(np.nanmean(hmt_fr_binned_in_space, axis=0)+stats.sem(hmt_fr_binned_in_space, axis=0))
                #y_max = max([y_max, hmt_max])
                #y_max = np.ceil(y_max)

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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_by_trial_type_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

def get_indices(hmt, tt):
    i = tt
    if hmt=="hit":
        j = 0
    elif hmt=="miss":
        j = 1
    elif hmt=="try":
        j = 2
    return i, j


def get_vmin_vmax(cluster_firing_maps, bin_cm=8):
    cluster_firing_maps_reduced = []
    for i in range(len(cluster_firing_maps)):
        cluster_firing_maps_reduced.append(block_reduce(cluster_firing_maps[i], bin_cm, func=np.mean))
    cluster_firing_maps_reduced = np.array(cluster_firing_maps_reduced)
    vmin= 0
    vmax= np.max(cluster_firing_maps_reduced)
    if vmax==0:
        print("stop here")
    return vmin, vmax

def plot_firing_rate_maps_per_trial_by_tt(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_hmr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(2, 1, figsize=(6,4.5), sharex=True)
            for ax, tt, color, ytitle in zip(axes, [0, 2], ["black", "#1FB5B2"], ["B", "P"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                if len(tt_processed_position_data)>0:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(tt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                    hmt_cluster_firing_maps = np.vstack((hmt_cluster_firing_maps, hmt_cluster_firing_maps[0]))
                    locations = np.arange(0, len(hmt_cluster_firing_maps[0]))
                    ordered = np.arange(0, len(tt_processed_position_data)+1, 1)
                    X, Y = np.meshgrid(locations, ordered)
                    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                    ax.pcolormesh(X, Y, hmt_cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
                if len(tt_processed_position_data)>0:
                    Edmond.plot_utility2.style_vr_plot(ax, len(tt_processed_position_data))
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(tt_processed_position_data)-1])
                ax.set_yticklabels([len(tt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                ax.set_xlim([0, track_length])
                ax.set_ylim([0, len(tt_processed_position_data)-1])

            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trial_by_trial_type_' + str(cluster_id) + '.png', dpi=300)
            plt.close()
    return

def plot_firing_rate_maps_per_trial_by_hmt(spike_data, processed_position_data, output_path, track_length, trial_types):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_hmr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for tt in trial_types:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["trial_type"] == tt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_tts = [str(int) for int in trial_types]
    string_tts = "-".join(string_tts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
            for ax, hmt, color, ytitle in zip(axes, ["hit", "try", "miss"], ["green", "orange", "red"], ["Hits", "Tries", "Runs"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                if len(hmt_processed_position_data)>0:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                    hmt_cluster_firing_maps = np.vstack((hmt_cluster_firing_maps, hmt_cluster_firing_maps[0]))
                    locations = np.arange(0, len(hmt_cluster_firing_maps[0]))
                    ordered = np.arange(0, len(hmt_processed_position_data)+1, 1)
                    X, Y = np.meshgrid(locations, ordered)
                    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                    ax.pcolormesh(X, Y, hmt_cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
                if len(hmt_processed_position_data)>0:
                    Edmond.plot_utility2.style_vr_plot(ax, len(hmt_processed_position_data))
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(hmt_processed_position_data)-1])
                ax.set_yticklabels([len(hmt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                ax.set_xlim([0, track_length])
                ax.set_ylim([0, len(hmt_processed_position_data)-1])

            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '_tt_'+string_tts+'.png', dpi=300)
            plt.close()
    return


def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_hmr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_99th_display = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            c = ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            plt.title(str(np.round(percentile_99th_display, decimals=1))+" Hz", fontsize=20)
            ax.tick_params(axis='both', which='both', labelsize=20)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
            ax.set_xlim([0, track_length])
            ax.set_ylim([0, len(processed_position_data)-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            tick_spacing = 100
            plt.locator_params(axis='y', nbins=3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            spikes_on_track.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
            cbar.set_ticks([0,np.max(cluster_firing_maps)])
            cbar.set_ticklabels(["0", "Max"])
            cbar.ax.tick_params(labelsize=20)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=300)
            plt.close()


def interpolate_by_trial(cluster_firing_maps, step_cm, track_length):
    x = np.arange(0, track_length)
    xnew = np.arange(step_cm/2, track_length, step_cm)

    interpolated_rate_map = []
    for i in range(len(cluster_firing_maps)):
        trial_cluster_firing_maps = cluster_firing_maps[i]
        y = trial_cluster_firing_maps
        f = interpolate.interp1d(x, y)

        ynew = f(xnew)
        interpolated_rate_map.append(ynew.tolist())

    return np.array(interpolated_rate_map)


def add_hit_miss_try(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    rewarded_processed_position_data = processed_position_data[(processed_position_data["rewarded"] == True)]
    speeds_in_rz = []
    for trial_number in np.unique(rewarded_processed_position_data["trial_number"]):
        trial_rewarded_processed_position_data = rewarded_processed_position_data[rewarded_processed_position_data["trial_number"] == trial_number]
        rewarded_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['speeds_binned_in_space'])
        rewarded_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['pos_binned_in_space'])
        in_rz_mask = (rewarded_bin_centres > reward_zone_start) & (rewarded_bin_centres <= reward_zone_end)
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space[in_rz_mask]
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space_in_reward_zone[~np.isnan(rewarded_speeds_in_space_in_reward_zone)]
        speeds_in_rz.extend(rewarded_speeds_in_space_in_reward_zone.tolist())

    speeds_in_rz = np.array(speeds_in_rz)
    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    avg_speed_in_rz =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        trial_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['speeds_binned_in_space'])
        trial_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['pos_binned_in_space'])
        in_rz_mask = (trial_bin_centres > reward_zone_start) & (trial_bin_centres <= reward_zone_end)
        trial_speeds_in_reward_zone = trial_speeds_in_space[in_rz_mask]
        trial_speeds_in_reward_zone = trial_speeds_in_reward_zone[~np.isnan(trial_speeds_in_reward_zone)]
        avg_trial_speed_in_reward_zone = np.mean(trial_speeds_in_reward_zone)

        if (trial_process_position_data["rewarded"].iloc[0] == True):
            hit_miss_try.append("hit")
        elif (avg_trial_speed_in_reward_zone >= lower) and (avg_trial_speed_in_reward_zone <= upper):
            hit_miss_try.append("try")
        else:
            hit_miss_try.append("miss")

    processed_position_data["hit_miss_try"] = hit_miss_try
    return processed_position_data, upper

def add_hmt_from_processed(processed_position_data, run_ids, try_ids):
    hmts = []
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_number = trial_row.trial_number.iloc[0]
        rewarded = trial_row.rewarded.iloc[0]
        pos_binned_in_space = trial_row.pos_binned_in_time.iloc[0]

        # eject the first and last trial and these won't have accurate hit/try/run classifications
        if trial_number == 1:
            hmt="rejected"
        elif trial_number == len(processed_position_data):
            hmt="rejected"
        elif min(pos_binned_in_space)>100:
            hmt="rejected"
        elif max(pos_binned_in_space)<100:
            hmt="rejected"
        elif trial_number in run_ids:
            hmt="miss"
        elif trial_number in try_ids:
            hmt="try"
        elif rewarded:
            hmt="hit"
        else: # if cases where the RZ bins are skipped, the mouse has ran fast enough
            hmt="miss"
        hmts.append(hmt)
    processed_position_data["hit_miss_try"] = hmts
    return processed_position_data

def process_recordings(vr_recording_path_list, processed_df):
    vr_recording_path_list.sort()
    for recording in vr_recording_path_list:

        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            track_length = get_track_length(recording)

            if (len(spike_data)>0) and (track_length==200):
                spike_data_from_processed = processed_df[processed_df["session_id"] == spike_data.session_id.iloc[0]]
                if len(spike_data_from_processed)>0:
                    run_ids = spike_data_from_processed["run_through_trialid"].iloc[0]
                    try_ids = spike_data_from_processed["try_trialid"].iloc[0]
                    processed_position_data = add_hmt_from_processed(processed_position_data, run_ids, try_ids)

                    plot_speed_histogram_with_error_all_trials(processed_position_data=processed_position_data, output_path=output_path, track_length=track_length)

                    # ANALYSIS OVER ALL TRIALS
                    plot_firing_rate_maps_per_trial(spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length)

                    # ANALYSIS BY HMT OF BEACONED TRIALS
                    plot_firing_rate_maps_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_type=0)
                    plot_firing_rate_maps_per_trial_by_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_types=[0])

                    # ANALYSIS BY HMT OF NONBEACONED TRIALS
                    #plot_firing_rate_maps_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_type=1)
                    #plot_firing_rate_maps_per_trial_by_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_types=[1])

                    # ANALYSIS BY HMT OF PROBE TRIALS
                    #plot_firing_rate_maps_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_type=2)
                    #plot_firing_rate_maps_per_trial_by_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length, trial_types=[2])

                    # ANALYSIS BY TRIAL TYPE
                    #plot_firing_rate_maps_tt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length)
                    #plot_firing_rate_maps_per_trial_by_tt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=track_length)
                    print("complete for ", recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

def main():
    print('-------------------------------------------------------------')

    processed_df = pd.DataFrame()
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_unsmoothened.pkl")], ignore_index=True)

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()])

    process_recordings(vr_path_list, processed_df)
    print("look now")

if __name__ == '__main__':
    main()