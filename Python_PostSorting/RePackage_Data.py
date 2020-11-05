import numpy as np



def package_rates_data_for_r(spike_data):
    for cluster_index in range(len(spike_data)):
        spike_data["speed_data"] = ""
        speed = np.array(spike_data.at[cluster_index, "spike_rate_in_time"])
        trials=np.array(spike_data.loc[cluster_index].spike_num_on_trials[1], dtype= np.int32)
        types=np.array(spike_data.loc[cluster_index].spike_num_on_trials[2], dtype= np.int32)

        sr=[]
        sr.append(np.array(speed))
        sr.append(np.array(trials))
        sr.append(np.array(types))
        spike_data.at[cluster_index, 'speed_data'] = list(sr)
    return spike_data

