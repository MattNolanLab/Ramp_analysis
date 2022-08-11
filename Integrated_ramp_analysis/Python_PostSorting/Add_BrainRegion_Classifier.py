import pandas as pd
import numpy as np



def load_brain_region_data_into_frame(spike_data, brain_region_path):
    print('I am loading brain region data into frame ...')
    spike_data["brain_region"] = ""
    region_data = pd.read_csv(brain_region_path, header=int())

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        #find data for that mouse & day
        session_fits = region_data['session_id'] == session_id
        session_fits = region_data[session_fits]

        # find the region
        region = session_fits['estimated_location'].values
        if len(region) == 0:
            print("brain region not there", session_id)
        spike_data.at[cluster,"brain_region"] = region
    return spike_data

