import pandas as pd
import numpy as np

def remove_false_positives(df):
    print("Removing sessions with low trial numbers..")

    max_trial_numbers = []
    for index, cluster_row in df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)

        # were going to take out the last trial number in the nested space binned column
        trial_numbers = cluster_row["spike_rate_on_trials_smoothed"].iloc[0][1]
        max_trial_numbers.append(max(trial_numbers))

    df["max_trial_number"] = max_trial_numbers
    df = df.drop(df[df.max_trial_number < 30].index)
    df.reset_index(drop=True, inplace=True)
    return df

def curate_data_by_rewards(df):
    #remove sessions without reward
    print("Removing sessions with no rewarded trials..")
    df["number_of_rewards"] = ""
    for cluster in range(len(df)):
        rewards = np.unique(np.array(df.loc[cluster,'rewarded_trials']))
        rewards = rewards[~np.isnan(rewards)]
        df.at[cluster,"number_of_rewards"] = rewards.shape[0]
    df = df.drop(df[df.number_of_rewards < 15].index)
    return df



def curate_data(df):
    df = curate_data_by_rewards(df)
    df.reset_index(drop=True, inplace=True)
    return df




def make_neuron_number(spike_data):
    print('I am calculating the neuron number ...')
    spike_data["neuron_number"] = ""

    final_frame = pd.DataFrame()
    last_session_id = 0

    for cluster in range(len(spike_data)):
        #print(spike_data.at[cluster, "session_id"], cluster)
        session_id = spike_data.at[cluster, "session_id"]

        if (session_id != last_session_id or last_session_id == 0):
            session_df = spike_data['session_id'] == session_id
            session_df = spike_data[session_df]

            session_df['neuron_number'] = np.arange(len(session_df))
            clusters_in_session = len(session_df)

            final_frame = final_frame.append(session_df)
            cluster += int(clusters_in_session)
            last_session_id = session_id
    return final_frame



def load_crtieria_data_into_frame(spike_data, criteria_path):
    criteria_data = pd.read_csv(criteria_path, header=int())

    spike_data["graduation"] = ""

    for cluster in range(len(spike_data)):
        mouse = spike_data.Mouse[cluster]
        day = int(spike_data.Day_numeric.values[cluster])
        cohort = spike_data.cohort.values[cluster]

        #find data for that mouse & day
        session_fits = criteria_data['Mouse'] == mouse
        session_fits = criteria_data[session_fits]
        cohort_fits = session_fits['Cohort'] == cohort
        cohort_fits = session_fits[cohort_fits]

        # find the region
        grad_day = int(cohort_fits['Graduation day'].values)

        if day >= grad_day:
            spike_data.at[cluster,"graduation"] = 1
        elif day < grad_day:
            spike_data.at[cluster,"graduation"] = 0

    return spike_data


def remove_outlier_waveforms(all_waveforms, max_deviations=2):
    # remove snippets that have data points > 3 standard dev away from mean
    mean = all_waveforms.mean(axis=1)
    sd = all_waveforms.std(axis=1)
    distance_from_mean = all_waveforms.T - mean
    outliers = np.sum(distance_from_mean > max_deviations * sd, axis=1) > 0
    return all_waveforms[:, ~outliers]

def add_peaks_to_troughs(df):
    peak_to_troughs = []
    for index, row in df.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        primary_channel = row["primary_channel"].iloc[0]
        random_snippets = row["random_snippets"].iloc[0][primary_channel-1]
        random_snippets = remove_outlier_waveforms(random_snippets)
        troughs = np.min(random_snippets, axis=0)
        peaks = np.max(random_snippets, axis=0)
        peak_to_trough = max(peaks-troughs)
        peak_to_troughs.append(peak_to_trough)
    df["snippet_peak_to_trough"] = peak_to_troughs
    return df