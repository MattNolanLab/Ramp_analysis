import pandas as pd
import numpy as np




def load_Teris_ramp_score_data_into_frame(spike_data):
    print('I am loading fit data into frame ...')
    spike_data["ramp_score"] = ""
    spike_data["ramp_score_shuff"] = ""

    fit_data = pd.read_csv("/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/ramp_score_coeff_export.csv", header=int())


    for cluster in range(len(spike_data)):
        session_id=spike_data.at[cluster, "session_id"]
        cluster_id=spike_data.at[cluster, "cluster_id"]

        #find data for that neuron
        session_fits = fit_data['session_id'] == session_id
        session_fits = fit_data[session_fits]

        # get only beaconed trials
        b_fits = session_fits['trial_type'] == 'beaconed'
        b_fits = session_fits[b_fits]

        # find that neuron
        neuron_fits = b_fits['cluster_id'] == cluster_id
        neuron_fits = b_fits[neuron_fits]

        # find shuffled/not shuffled ramp score
        real_neuron_fits = neuron_fits['is_shuffled'] == False
        real_neuron_fits = neuron_fits[real_neuron_fits]

        shuff_neuron_fits = neuron_fits['is_shuffled'] == True
        shuff_neuron_fits = neuron_fits[shuff_neuron_fits]

        try:
            real_ramp_score = real_neuron_fits['ramp_score'].values # extract fit
            shuff_ramp_score = shuff_neuron_fits['ramp_score'].values # extract fit
            if real_ramp_score[0] == np.nan:
                real_ramp_score[0] = "None"
            if shuff_ramp_score[0] == np.nan:
                shuff_ramp_score[0] = "None"
            spike_data.at[cluster,"ramp_score"] = real_ramp_score
            spike_data.at[cluster,"ramp_score_shuff"] = shuff_ramp_score
        except IndexError:
            spike_data.at[cluster,"ramp_score"] = 'None'
            spike_data.at[cluster,"ramp_score_shuff"] = 'None'
    return spike_data

