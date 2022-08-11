import pandas as pd
import numpy as np




def load_ramp_score_data_into_frame(spike_data, ramp_score_path):
    print('I am loading fit data into frame ...')
    spike_data["ramp_score"] = ""
    spike_data["ramp_score_shuff"] = ""
    fit_data = pd.read_csv(ramp_score_path)
    fit_data = fit_data[(fit_data["trial_type"] == "beaconed") &
                        (fit_data["ramp_region"] == "outbound")]

    for cluster in range(len(spike_data)):
        session_id=spike_data.at[cluster, "session_id"]
        cluster_id=spike_data.at[cluster, "cluster_id"]

        cluster_fit_data = fit_data[(fit_data["session_id"] == session_id) &
                                    (fit_data["cluster_id"] == cluster_id)]

        if len(cluster_fit_data)==1:
            ramp_score = cluster_fit_data.score.iloc[0]
        else:
            ramp_score = "None"

        spike_data.at[cluster,"ramp_score"] = ramp_score

    return spike_data

