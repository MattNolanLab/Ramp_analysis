import pandas as pd
import numpy as np




def load_Teris_clustering_data_into_frame(spike_data):
    print('I am loading clustering analysis into frame ...')
    spike_data["label"] = ""

    data_path = "/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/cluster_results.pkl"
    fit_data = pd.read_pickle(data_path)


    for cluster in range(len(spike_data)):
        session_id=spike_data.at[cluster, "session_id"]
        cluster_id=spike_data.at[cluster, "cluster_id"]

        session_fits = fit_data['session_id'] == session_id
        session_fits = fit_data[session_fits]

        # find that neuron
        neuron_fits = session_fits['cluster_id'] == cluster_id
        neuron_fits = session_fits[neuron_fits]

        try:
            label = int(neuron_fits['label']) # extract fit
            if label == np.nan:
                label = "None"
            spike_data.at[cluster,"label"] = label
        except (IndexError, TypeError):
            spike_data.at[cluster,"label"] = 'None'
    return spike_data

