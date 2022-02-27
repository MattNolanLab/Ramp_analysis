import pandas as pd
import numpy as np


def remove_false_positives(df):
    print("Removing sessions with low trial numbers..")

    df.reset_index(drop=True, inplace=True)
    df["max_trial_number"] = ""

    for cluster in range(len(df)):
        try:
            df.at[cluster,"max_trial_number"] = max(df.loc[cluster,'trial_number'])
        except ValueError:
            df.at[cluster,"max_trial_number"] = 0

    df = df.drop(df[df.max_trial_number < 30].index)
    df = df.dropna(axis=0)
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



def check_graduation(df):
    print("....")

    df.reset_index(drop=True, inplace=True)
    df["probe_trial_label"] = ""

    for cluster in range(len(df)):
        try:
            types=np.unique(np.array(df.iloc[cluster].spike_rate_in_time[4].real, dtype= np.int32))
            if len(types) == 3:
                df.at[cluster,"probe_trial_label"] = 1
            else:
                df.at[cluster,"probe_trial_label"] = 0

        except ValueError:
            df.at[cluster,"probe_trial_label"] = 0

    return df
