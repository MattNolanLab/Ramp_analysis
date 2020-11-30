import pandas as pd




def load_fit_data_into_frame(spike_data, fit_data):
    print('I am loading fit data into frame ...')
    spike_data["meanVar"] = ""
    spike_data["Dominant_Atom"] = ""
    spike_data["Dominant_model"] = ""
    spike_data["meanCurves"] = ""

    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        session_id=spike_data.at[cluster, "session_id"]
        neuron_id=spike_data.at[cluster, "neuron_number"]

        #find data for that neuron
        session_fits = fit_data['session_id'] == session_id
        session_fits = fit_data[session_fits]

        # get only beaconed trials
        b_fits = session_fits['trial_type'] == 'beaconed'
        b_fits = session_fits[b_fits]

        # find that neuron
        neuron_fits = b_fits['neuron'] == neuron_id
        neuron_fits = b_fits[neuron_fits]

        # bimodal comparison pvalue is whether the best univariate model is significantly better to the bivariate model
        try:
            vaf = neuron_fits['meanVar'].values[0] # extract fit
            BestUni = neuron_fits['bestUniModel'].values[0] # extract fit
            uni_pval = neuron_fits['unimodel_comparison_pvalue'].values[0] # extract fit
            bi_pval = neuron_fits['bimodel_comparison_pvalue'].values[0] # extract fit
            fit_curve = neuron_fits['meanCurves'].values[0] # extract fit
            model = 'null'

            if BestUni == 0:
                if bi_pval < 0.05:
                    model = 'position-bivariate'
            if BestUni == 1:
                if bi_pval < 0.05:
                    model = 'speed-bivariate'

            if BestUni == 0:
                if bi_pval > 0.05:
                    model = 'position-univariate'
            if BestUni == 1:
                if bi_pval < 0.05:
                    model = 'speed-bivariate'

            #if uni_pval > 0.001 and bi_pval > 0.001:
             #   model = 'Both-univariate'
            #if uni_pval > 0.001 and bi_pval < 0.001:
            #    model = 'Both-bivariate'

            print(model)
            spike_data.at[cluster,"meanVar"] = vaf
            spike_data.at[cluster,"Dominant_model"] = model
            spike_data.at[cluster,"meanCurves"] = fit_curve
        except IndexError:
            spike_data.at[cluster,"meanVar"] = 'None'
            spike_data.at[cluster,"Dominant_model"] = 'None'
            spike_data.at[cluster,"meanCurves"] = 'None'
    return spike_data






def load_Teris_fit_data_into_frame(spike_data, fit_data):
    print('I am loading fit data into frame ...')
    spike_data["bestmodel"] = ""
    spike_data["bestatom"] = ""

    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
        session_id=spike_data.at[cluster, "session_id"]
        neuron_id=spike_data.at[cluster, "neuron_number"]

        #find data for that neuron
        session_fits = fit_data['session_id'] == session_id
        session_fits = fit_data[session_fits]

        # get only beaconed trials
        b_fits = session_fits['trial_type'] == 'beaconed'
        b_fits = session_fits[b_fits]

        # find that neuron
        neuron_fits = b_fits['neuron'] == neuron_id
        neuron_fits = b_fits[neuron_fits]
        try:
            best_model = neuron_fits['modelType'].values # extract fit
            best_atom = neuron_fits['bestOutboundDict'].values # extract fit
            if len(best_model) == 0:
                best_model = "None"
            if len(best_atom) == 0:
                best_atom = "None"
            print(best_model, "atom =" , best_atom, len(best_model))

            spike_data.at[cluster,"bestmodel"] = best_model
            spike_data.at[cluster,"bestatom"] = best_atom
        except IndexError:
            spike_data.at[cluster,"bestmodel"] = 'None'
            spike_data.at[cluster,"bestatom"] = 'None'

    return spike_data















def load_Teris_ramp_score_data_into_frame(spike_data):
    print('I am loading fit data into frame ...')
    spike_data["ramp_score"] = ""
    spike_data["ramp_score_shuff"] = ""

    fit_data = pd.read_csv("/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/ramp_score_coeff_export.csv", header=int())


    for cluster in range(len(spike_data)):
        print(spike_data.at[cluster, "session_id"], cluster)
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

            spike_data.at[cluster,"ramp_score"] = real_ramp_score
            spike_data.at[cluster,"ramp_score_shuff"] = shuff_ramp_score
        except IndexError:
            spike_data.at[cluster,"ramp_score"] = 'None'
            spike_data.at[cluster,"ramp_score_shuff"] = 'None'

    return spike_data

