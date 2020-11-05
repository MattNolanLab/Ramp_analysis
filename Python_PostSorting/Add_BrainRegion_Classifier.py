import pandas as pd
import numpy as np


def add_mouse_to_frame(df):
    #remove sessions without reward
    print("Removing sessions with no rewarded trials..")
    df["Mouse"] = ""
    df["Day"] = ""
    for cluster in range(len(df)):
        session_id = df.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)
        df.at[cluster,"Mouse"] = mouse
        df.at[cluster,"Day"] = day
    return df


# extract what mouse and day it is from the session_id column in spatial_firing
def extract_mouse_and_day(session_id):
    mouse = session_id.rsplit('_', 3)[0]
    day = session_id.rsplit('_', 3)[1]
    #mouse = mouse1.rsplit('M', 3)[1]
    #day = day1.rsplit('D', 3)[1]
    return day, mouse


def load_brain_region_data_into_frame(spike_data):
    print('I am loading brain region data into frame ...')
    spike_data["brain_region"] = ""
    spike_data = add_mouse_to_frame(spike_data)
    #print("mouse and day columns added to frame")
    region_data = pd.read_csv("/Users/sarahtennant/Work/Analysis/in_vivo_virtual_reality/data/tetrode_locations.csv", header=int())

    for cluster in range(len(spike_data)):
        session_id = spike_data.session_id.values[cluster]
        day, mouse = extract_mouse_and_day(session_id)
        #find data for that mouse & day
        session_fits = region_data['session_id'] == session_id
        session_fits = region_data[session_fits]

        # find the region
        region = session_fits['estimated_location'].values
        if len(region) == 0:
            print("brain region not there", session_id)


        spike_data.at[cluster,"brain_region"] = region

    return spike_data

