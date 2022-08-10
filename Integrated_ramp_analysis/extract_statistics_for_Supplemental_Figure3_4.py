from scipy import stats
import scipy
from Edmond.VR_grid_analysis.vr_grid_cells import *
from Edmond.utility_functions.array_manipulations import *

warnings.filterwarnings('ignore')


def print_location_p_values(processed_df):
    # we would like to ask whether cells there is a difference between MEC and Pre/parasubiculum
    print("I am now print kolorov smirnov p values for location dependence")
    for slope_type in ["Positive", "Negative"]:
        processed_df_slope_type = processed_df[processed_df["lm_group_b"] == slope_type]

        MEC_data = processed_df_slope_type[processed_df_slope_type["brain_region"] == "MEC"]
        PS_data = processed_df_slope_type[processed_df_slope_type["brain_region"] == "PS"]

        # compare slope value and ramp score
        for score_type in ["ramp_score", "asr_b_o_rewarded_fit_slope"]:
            MEC_values = np.array(MEC_data[score_type], dtype=np.float64)
            PS_values = np.array(PS_data[score_type], dtype=np.float64)

            p = scipy.stats.kstest(MEC_values[~np.isnan(MEC_values)], PS_values[~np.isnan(PS_values)])[1]
            print("for slope type: ", slope_type, " and score type: ", score_type, ",  p = ", str(p))

def print_theta_p_values(processed_df):
    # we would like to ask whether cells there is a difference between theta and non theta rhythmic cells
    print("I am now print kolorov smirnov p values for theta modulation")
    for slope_type in ["Positive", "Negative"]:
        processed_df_slope_type = processed_df[processed_df["lm_group_b"] == slope_type]

        # compare slope value and ramp score
        for score_type in ["ramp_score", "asr_b_o_rewarded_fit_slope"]:
            processed_df_score_type = processed_df_slope_type[processed_df_slope_type[score_type] != "None"]

            TR_data = processed_df_score_type[processed_df_score_type["ThetaIndex"] >= 0.07]
            NR_data = processed_df_score_type[processed_df_score_type["ThetaIndex"] < 0.07]

            TR_values = np.array(TR_data[score_type], dtype=np.float64)
            NR_values = np.array(NR_data[score_type], dtype=np.float64)

            p = scipy.stats.kstest(TR_values[~np.isnan(TR_values)], NR_values[~np.isnan(NR_values)])[1]
            print("for slope type: ", slope_type, " and score type: ", score_type, ",  p = ", str(p))

def add_unique_id(processed_df):
    unique_ids = []
    for index, cluster_row in processed_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id = cluster_row["cluster_id"].iloc[0]
        session_id = cluster_row["session_id"].iloc[0]
        unique_id = session_id+"_"+str(cluster_id)
        unique_ids.append(unique_id)
    processed_df["unique_id"] = unique_ids
    return processed_df

def main():
    print('-------------------------------------------------------------')

    processed_df = pd.DataFrame()
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort4_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort5_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort7_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort2_unsmoothened.pkl")], ignore_index=True)
    processed_df = pd.concat([processed_df, pd.read_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_data/Processed_cohort3_unsmoothened.pkl")], ignore_index=True)
    processed_df = add_unique_id(processed_df)
    classifications = pd.read_csv('/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/all_results_coefficients.csv', sep="\t")
    processed_df = pd.merge(processed_df, classifications, on="unique_id", how="left")

    print_theta_p_values(processed_df)
    print_location_p_values(processed_df)

    print("look now")


if __name__ == '__main__':
    main()