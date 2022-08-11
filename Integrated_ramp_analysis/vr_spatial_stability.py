import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.vr_sync_spatial_data
import traceback
import PostSorting.vr_spatial_firing
import control_sorting_analysis
import scipy.stats
import warnings
from scipy import stats
import plot_utility
import os
import sys
import settings
from astropy.convolution import convolve, Gaussian1DKernel

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length

def add_vr_spatial_stability_score(spike_data, processed_position_data):
    midpoint_trial = int((np.asarray(processed_position_data["trial_number"])[-1])/2)
    b_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 0]["trial_number"])
    nb_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 1]["trial_number"])
    p_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 2]["trial_number"])

    # get trial numbers for 1st and 2nd half depending on trial type
    b_trial_numbers_first_half = b_trial_numbers[b_trial_numbers < midpoint_trial]
    b_trial_numbers_second_half = b_trial_numbers[b_trial_numbers >= midpoint_trial]
    nb_trial_numbers_first_half = nb_trial_numbers[nb_trial_numbers < midpoint_trial]
    nb_trial_numbers_second_half = nb_trial_numbers[nb_trial_numbers >= midpoint_trial]
    p_trial_numbers_first_half = p_trial_numbers[p_trial_numbers < midpoint_trial]
    p_trial_numbers_second_half = p_trial_numbers[p_trial_numbers >= midpoint_trial]

    vr_spatial_score_b = []; vr_spatial_score_nb = []; vr_spatial_score_p = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]
        fr_binned_in_space = np.array(cluster_df["fr_binned_in_space"].iloc[0])

        # make rate maps for 1st and 2nd half and then make a mask for the non-nan values
        b_rate_map_1st = np.nanmean(fr_binned_in_space[b_trial_numbers_first_half-1], axis=0); mask_for_nans_b_1st = ~np.isnan(b_rate_map_1st); mask_for_infs_b_1st = ~np.isinf(b_rate_map_1st)
        nb_rate_map_1st = np.nanmean(fr_binned_in_space[nb_trial_numbers_first_half-1], axis=0); mask_for_nans_nb_1st = ~np.isnan(nb_rate_map_1st); mask_for_infs_nb_1st = ~np.isinf(nb_rate_map_1st)
        p_rate_map_1st = np.nanmean(fr_binned_in_space[p_trial_numbers_first_half-1], axis=0); mask_for_nans_p_1st = ~np.isnan(p_rate_map_1st); mask_for_infs_p_1st = ~np.isinf(p_rate_map_1st)

        b_rate_map_2nd = np.nanmean(fr_binned_in_space[b_trial_numbers_second_half-1], axis=0); mask_for_nans_b_2nd = ~np.isnan(b_rate_map_2nd); mask_for_infs_b_2nd = ~np.isinf(b_rate_map_2nd)
        nb_rate_map_2nd = np.nanmean(fr_binned_in_space[nb_trial_numbers_second_half-1], axis=0); mask_for_nans_nb_2nd = ~np.isnan(nb_rate_map_2nd); mask_for_infs_nb_2nd = ~np.isinf(nb_rate_map_2nd)
        p_rate_map_2nd = np.nanmean(fr_binned_in_space[p_trial_numbers_second_half-1], axis=0); mask_for_nans_p_2nd = ~np.isnan(p_rate_map_2nd); mask_for_infs_p_2nd = ~np.isinf(p_rate_map_2nd)

        #combine the non-nan value masks so we don't take any nan values into the pearson r calculation
        b_combined_mask = mask_for_nans_b_1st & mask_for_nans_b_2nd & mask_for_infs_b_1st & mask_for_infs_b_2nd
        nb_combined_mask = mask_for_nans_nb_1st & mask_for_nans_nb_2nd & mask_for_infs_nb_1st & mask_for_infs_nb_2nd
        p_combined_mask = mask_for_nans_p_1st & mask_for_nans_p_2nd & mask_for_infs_p_1st & mask_for_infs_p_2nd

        # calculate the pearson R for the first and 2nd half rate maps
        if (len(b_rate_map_1st[b_combined_mask]) == len(b_rate_map_2nd[b_combined_mask])) and len(b_rate_map_1st[b_combined_mask])>1:
            b_pearson_r, _ = scipy.stats.pearsonr(b_rate_map_1st[b_combined_mask], b_rate_map_2nd[b_combined_mask])
        else:
            b_pearson_r = np.nan

        if (len(nb_rate_map_1st[nb_combined_mask]) == len(nb_rate_map_2nd[nb_combined_mask])) and len(nb_rate_map_1st[nb_combined_mask])>1:
            nb_pearson_r, _ = scipy.stats.pearsonr(nb_rate_map_1st[nb_combined_mask], nb_rate_map_2nd[nb_combined_mask])
        else:
            nb_pearson_r = np.nan

        if (len(p_rate_map_1st[p_combined_mask]) == len(p_rate_map_2nd[p_combined_mask])) and len(p_rate_map_1st[p_combined_mask])>1:
            p_pearson_r, _ = scipy.stats.pearsonr(p_rate_map_1st[p_combined_mask], p_rate_map_2nd[p_combined_mask])
        else:
            p_pearson_r = np.nan

        vr_spatial_score_b.append(b_pearson_r)
        vr_spatial_score_nb.append(nb_pearson_r)
        vr_spatial_score_p.append(p_pearson_r)#

    spike_data["vr_stability_score_b"] = vr_spatial_score_b
    spike_data["vr_stability_score_nb"] = vr_spatial_score_nb
    spike_data["vr_stability_score_p"] = vr_spatial_score_p
    return spike_data

def add_half_session_slopes(spike_data, processed_position_data, track_length):
    x = np.arange(1, track_length+1)

    midpoint_trial = int((np.asarray(processed_position_data["trial_number"])[-1])/2)
    b_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 0]["trial_number"])
    nb_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 1]["trial_number"])
    p_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == 2]["trial_number"])

    # get trial numbers for 1st and 2nd half depending on trial type
    b_trial_numbers_first_half = b_trial_numbers[b_trial_numbers < midpoint_trial]
    b_trial_numbers_second_half = b_trial_numbers[b_trial_numbers >= midpoint_trial]
    nb_trial_numbers_first_half = nb_trial_numbers[nb_trial_numbers < midpoint_trial]
    nb_trial_numbers_second_half = nb_trial_numbers[nb_trial_numbers >= midpoint_trial]
    p_trial_numbers_first_half = p_trial_numbers[p_trial_numbers < midpoint_trial]
    p_trial_numbers_second_half = p_trial_numbers[p_trial_numbers >= midpoint_trial]

    all_slope_b_o1 = []; all_slope_nb_o1 = []; all_slope_p_o1 = []; all_slope_b_h1 = []; all_slope_nb_h1 = []; all_slope_p_h1 = []
    all_slope_b_o2 = []; all_slope_nb_o2 = []; all_slope_p_o2 = []; all_slope_b_h2 = []; all_slope_nb_h2 = []; all_slope_p_h2 = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]
        fr_binned_in_space = np.array(cluster_df["fr_binned_in_space"].iloc[0])

        # make rate maps for 1st and 2nd half and then make a mask for the non-nan values
        b_rate_map_1st = np.nanmean(fr_binned_in_space[b_trial_numbers_first_half-1], axis=0); mask_for_nans_b_1st = ~np.isnan(b_rate_map_1st); mask_for_infs_b_1st = ~np.isinf(b_rate_map_1st)
        nb_rate_map_1st = np.nanmean(fr_binned_in_space[nb_trial_numbers_first_half-1], axis=0); mask_for_nans_nb_1st = ~np.isnan(nb_rate_map_1st); mask_for_infs_nb_1st = ~np.isinf(nb_rate_map_1st)
        p_rate_map_1st = np.nanmean(fr_binned_in_space[p_trial_numbers_first_half-1], axis=0); mask_for_nans_p_1st = ~np.isnan(p_rate_map_1st); mask_for_infs_p_1st = ~np.isinf(p_rate_map_1st)

        b_rate_map_2nd = np.nanmean(fr_binned_in_space[b_trial_numbers_second_half-1], axis=0); mask_for_nans_b_2nd = ~np.isnan(b_rate_map_2nd); mask_for_infs_b_2nd = ~np.isinf(b_rate_map_2nd)
        nb_rate_map_2nd = np.nanmean(fr_binned_in_space[nb_trial_numbers_second_half-1], axis=0); mask_for_nans_nb_2nd = ~np.isnan(nb_rate_map_2nd); mask_for_infs_nb_2nd = ~np.isinf(nb_rate_map_2nd)
        p_rate_map_2nd = np.nanmean(fr_binned_in_space[p_trial_numbers_second_half-1], axis=0); mask_for_nans_p_2nd = ~np.isnan(p_rate_map_2nd); mask_for_infs_p_2nd = ~np.isinf(p_rate_map_2nd)

        #combine the non-nan value masks so we don't take any nan values into the pearson r calculation
        b_combined_mask1st = mask_for_nans_b_1st & mask_for_infs_b_1st
        b_combined_mask2nd = mask_for_nans_b_2nd & mask_for_infs_b_2nd
        nb_combined_mask1st = mask_for_nans_nb_1st & mask_for_infs_nb_1st
        nb_combined_mask2nd = mask_for_nans_nb_2nd & mask_for_infs_nb_2nd
        p_combined_mask1st = mask_for_nans_p_1st & mask_for_infs_p_1st
        p_combined_mask2nd = mask_for_nans_p_2nd & mask_for_infs_p_2nd

        # calculate the pearson R for the first and 2nd half rate maps
        if (len(b_rate_map_1st[b_combined_mask1st]) == len(b_rate_map_2nd[b_combined_mask2nd])) and len(b_rate_map_1st[b_combined_mask1st]) == track_length:
            slope_o_b1, _, _, _, _ = stats.linregress(x[30:90], b_rate_map_1st[b_combined_mask1st][30:90])
            slope_h_b1, _, _, _, _ = stats.linregress(x[110:170], b_rate_map_1st[b_combined_mask1st][110:170])
            slope_o_b2, _, _, _, _ = stats.linregress(x[30:90], b_rate_map_2nd[b_combined_mask2nd][30:90])
            slope_h_b2, _, _, _, _ = stats.linregress(x[110:170], b_rate_map_2nd[b_combined_mask2nd][110:170])
        else:
            slope_o_b1 = np.nan
            slope_h_b1 = np.nan
            slope_o_b2 = np.nan
            slope_h_b2 = np.nan

        # calculate the pearson R for the first and 2nd half rate maps
        if (len(nb_rate_map_1st[nb_combined_mask1st]) == len(nb_rate_map_2nd[nb_combined_mask2nd])) and len(nb_rate_map_1st[nb_combined_mask1st]) == track_length:
            slope_o_nb1, _, _, _, _ = stats.linregress(x[30:90], nb_rate_map_1st[nb_combined_mask1st][30:90])
            slope_h_nb1, _, _, _, _ = stats.linregress(x[110:170], nb_rate_map_1st[nb_combined_mask1st][110:170])
            slope_o_nb2, _, _, _, _ = stats.linregress(x[30:90], nb_rate_map_2nd[nb_combined_mask2nd][30:90])
            slope_h_nb2, _, _, _, _ = stats.linregress(x[110:170], nb_rate_map_2nd[nb_combined_mask2nd][110:170])
        else:
            slope_o_nb1 = np.nan
            slope_h_nb1 = np.nan
            slope_o_nb2 = np.nan
            slope_h_nb2 = np.nan

        # calculate the pearson R for the first and 2nd half rate maps
        if (len(p_rate_map_1st[p_combined_mask1st]) == len(p_rate_map_2nd[p_combined_mask2nd])) and len(p_rate_map_1st[p_combined_mask1st]) == track_length:
            slope_o_p1, _, _, _, _ = stats.linregress(x[30:90], p_rate_map_1st[p_combined_mask1st][30:90])
            slope_h_p1, _, _, _, _ = stats.linregress(x[110:170], p_rate_map_1st[p_combined_mask1st][110:170])
            slope_o_p2, _, _, _, _ = stats.linregress(x[30:90], p_rate_map_2nd[p_combined_mask2nd][30:90])
            slope_h_p2, _, _, _, _ = stats.linregress(x[110:170], p_rate_map_2nd[p_combined_mask2nd][110:170])
        else:
            slope_o_p1 = np.nan
            slope_h_p1 = np.nan
            slope_o_p2 = np.nan
            slope_h_p2 = np.nan

        all_slope_b_o1.append(slope_o_b1)
        all_slope_nb_o1.append(slope_o_nb1)
        all_slope_p_o1.append(slope_o_p1)
        all_slope_b_h1.append(slope_h_b1)
        all_slope_nb_h1.append(slope_h_nb1)
        all_slope_p_h1.append(slope_h_p1)
        all_slope_b_o2.append(slope_o_b2)
        all_slope_nb_o2.append(slope_o_nb2)
        all_slope_p_o2.append(slope_o_p2)
        all_slope_b_h2.append(slope_h_b2)
        all_slope_nb_h2.append(slope_h_nb2)
        all_slope_p_h2.append(slope_h_p2)

    spike_data["all_slope_b_o1"] = all_slope_b_o1
    spike_data["all_slope_nb_o1"] = all_slope_nb_o1
    spike_data["all_slope_p_o1"] = all_slope_p_o1
    spike_data["all_slope_b_h1"] = all_slope_b_h1
    spike_data["all_slope_nb_h1"] = all_slope_nb_h1
    spike_data["all_slope_p_h1"] = all_slope_p_h1
    spike_data["all_slope_b_o2"] = all_slope_b_o2
    spike_data["all_slope_nb_o2"] = all_slope_nb_o2
    spike_data["all_slope_p_o2"] = all_slope_p_o2
    spike_data["all_slope_b_h2"] = all_slope_b_h2
    spike_data["all_slope_nb_h2"] = all_slope_nb_h2
    spike_data["all_slope_p_h2"] = all_slope_p_h2
    return spike_data


def process_recordings(vr_recording_path_list, concat_frame):
    vr_recording_path_list.sort()

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing_unsmoothened.pkl")
            processed_position_data =  pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            spike_data = add_vr_spatial_stability_score(spike_data, processed_position_data)
            spike_data= add_half_session_slopes(spike_data, processed_position_data, track_length=get_track_length(recording))

            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            print("successfully processed on "+recording)

            spatial_firing_short = spike_data[["session_id", "cluster_id", "vr_stability_score_b", "vr_stability_score_nb", "vr_stability_score_p",
                                               "all_slope_b_o1", "all_slope_nb_o1", "all_slope_p_o1", "all_slope_b_h1", "all_slope_nb_h1", "all_slope_p_h1",
                                               "all_slope_b_o2", "all_slope_nb_o2", "all_slope_p_o2", "all_slope_b_h2", "all_slope_nb_h2", "all_slope_p_h2"]]

            concat_frame = pd.concat([concat_frame, spatial_firing_short], ignore_index=True)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    return concat_frame

def print_population_stability(processed_df):
    pospos = processed_df[(processed_df['lm_group_b'] == "Positive") & (processed_df['lm_group_b_h'] == "Positive")]
    posneg = processed_df[(processed_df['lm_group_b'] == "Positive") & (processed_df['lm_group_b_h'] == "Negative")]
    posun = processed_df[(processed_df['lm_group_b'] == "Positive") & (processed_df['lm_group_b_h'] == "Unclassified")]
    negpos = processed_df[(processed_df['lm_group_b'] == "Negative") & (processed_df['lm_group_b_h'] == "Positive")]
    negneg = processed_df[(processed_df['lm_group_b'] == "Negative") & (processed_df['lm_group_b_h'] == "Negative")]
    negun = processed_df[(processed_df['lm_group_b'] == "Negative") & (processed_df['lm_group_b_h'] == "Unclassified")]
    unun = processed_df[(processed_df['lm_group_b'] == "Unclassified") & (processed_df['lm_group_b_h'] == "Unclassified")]
    all_ramp =  processed_df[(processed_df['lm_group_b'] == "Positive") | (processed_df['lm_group_b'] == "Negative") ]
    print("printing half session correlations")
    print("For all ramp neurons, mean: ", str(np.nanmean(all_ramp["vr_stability_score_b"])), ", std: ", str(np.nanstd(all_ramp["vr_stability_score_b"])))
    print("For pospos neurons, mean: ", str(np.nanmean(pospos["vr_stability_score_b"])), ", std: ", str(np.nanstd(pospos["vr_stability_score_b"])))
    print("For posneg neurons, mean: ", str(np.nanmean(posneg["vr_stability_score_b"])), ", std: ", str(np.nanstd(posneg["vr_stability_score_b"])))
    print("For posun neurons, mean: ", str(np.nanmean(posun["vr_stability_score_b"])), ", std: ", str(np.nanstd(posun["vr_stability_score_b"])))
    print("For negpos neurons, mean: ", str(np.nanmean(negpos["vr_stability_score_b"])), ", std: ", str(np.nanstd(negpos["vr_stability_score_b"])))
    print("For negneg neurons, mean: ", str(np.nanmean(negneg["vr_stability_score_b"])), ", std: ", str(np.nanstd(negneg["vr_stability_score_b"])))
    print("For negun neurons, mean: ", str(np.nanmean(negun["vr_stability_score_b"])), ", std: ", str(np.nanstd(negun["vr_stability_score_b"])))
    print("For unun neurons, mean: ", str(np.nanmean(unun["vr_stability_score_b"])), ", std: ", str(np.nanstd(unun["vr_stability_score_b"])))

    t, p = scipy.stats.ttest_ind(all_ramp["vr_stability_score_b"], unun["vr_stability_score_b"], nan_policy="omit")
    degrees_of_freedom = len(all_ramp)+len(unun)-2
    print("Comparing the half session correlations between rampings and non ramping neurons, using a 2 sample t test, p=", str(p), ", t=", str(t), ", df=", str(degrees_of_freedom))


    print("printing half session r2 values for the outbound and homebound slopes")
    for ramps, ramps_str in zip([all_ramp, pospos, posneg, posun, negpos, negneg, negun, unun], ["all_ramp", "pospos", "posneg", "posun", "negpos", "negneg", "negun", "unun"]):
        _, _, b_outbound_slope_r2, _, _ = stats.linregress(ramps["all_slope_b_o1"][~np.isnan(ramps["all_slope_b_o1"])], ramps["all_slope_b_o2"][~np.isnan(ramps["all_slope_b_o2"])])
        _, _, b_homebound_slope_r2, _, _ = stats.linregress(ramps["all_slope_b_h1"][~np.isnan(ramps["all_slope_b_h1"])], ramps["all_slope_b_h2"][~np.isnan(ramps["all_slope_b_h2"])])
        _, _, nb_outbound_slope_r2, _, _ = stats.linregress(ramps["all_slope_nb_o1"][~np.isnan(ramps["all_slope_nb_o1"])], ramps["all_slope_nb_o2"][~np.isnan(ramps["all_slope_nb_o2"])])
        _, _, nb_homebound_slope_r2, _, _ = stats.linregress(ramps["all_slope_nb_h1"][~np.isnan(ramps["all_slope_nb_h1"])], ramps["all_slope_nb_h2"][~np.isnan(ramps["all_slope_nb_h2"])])
        _, _, p_outbound_slope_r2, _, _ = stats.linregress(ramps["all_slope_p_o1"][~np.isnan(ramps["all_slope_p_o1"])], ramps["all_slope_p_o2"][~np.isnan(ramps["all_slope_p_o2"])])
        _, _, p_homebound_slope_r2, _, _ = stats.linregress(ramps["all_slope_p_h1"][~np.isnan(ramps["all_slope_p_h1"])], ramps["all_slope_p_h2"][~np.isnan(ramps["all_slope_p_h2"])])

        print("for ", ramps_str, ", r2 of beaconed outbound slopes: ", str(b_outbound_slope_r2))
        print("for ", ramps_str, ", r2 of beaconed homebound slopes: ", str(b_homebound_slope_r2))
        print("for ", ramps_str, ", r2 of nonbeaconed outbound slopes: ", str(nb_outbound_slope_r2))
        print("for ", ramps_str, ", r2 of nonbeaconed homebound slopes: ", str(nb_homebound_slope_r2))
        print("for ", ramps_str, ", r2 of probe outbound slopes: ", str(p_outbound_slope_r2))
        print("for ", ramps_str, ", r2 of probe homebound slopes: ", str(p_homebound_slope_r2))

    # compare slopes before and after the reward zone
    df = pd.DataFrame()
    all_ramp["ramp_type"] = "ramping"
    unun["ramp_type"] = "not_ramping"
    df = pd.concat([df, all_ramp], ignore_index=True)
    df = pd.concat([df, unun], ignore_index=True)
    df = df[["ramp_type", "all_slope_b_o1", "all_slope_b_o2", "all_slope_b_h1", "all_slope_b_h2"]]
    df.to_csv("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/half_scores_wtih_ramp_annotation.csv")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    df = pd.DataFrame()
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()]
    df = process_recordings(vr_path_list, concat_frame=df)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()]
    df = process_recordings(vr_path_list, concat_frame=df)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()]
    df = process_recordings(vr_path_list, concat_frame=df)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()]
    df = process_recordings(vr_path_list, concat_frame=df)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()]
    df = process_recordings(vr_path_list, concat_frame=df)

    processed_df = pd.DataFrame()
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_unsmoothened.pkl")], ignore_index=True)

    processed_df = pd.merge(processed_df, df[["session_id", "cluster_id", "vr_stability_score_b", "vr_stability_score_nb", "vr_stability_score_p",
                                              "all_slope_b_o1", "all_slope_nb_o1", "all_slope_p_o1", "all_slope_b_h1", "all_slope_nb_h1", "all_slope_p_h1",
                                              "all_slope_b_o2", "all_slope_nb_o2", "all_slope_p_o2", "all_slope_b_h2", "all_slope_nb_h2", "all_slope_p_h2"]], on=["session_id", "cluster_id"])

    # add classifications df outputted from R
    classications = pd.read_csv('/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/all_results_coefficients.csv', sep="\t")
    processed_df = pd.merge(processed_df, classications, on=["session_id", "cluster_id"])

    print_population_stability(processed_df)

if __name__ == '__main__':
    main()
