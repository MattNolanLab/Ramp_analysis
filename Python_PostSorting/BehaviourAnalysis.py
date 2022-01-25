import numpy as np
import csv



def calculate_graduation_day(spike_data):
    print('calculating graduation day')
    spike_data["probe_trials"] = ""

    for cluster in range(len(spike_data)):
        reward_rate = np.unique(np.array(spike_data.loc[cluster,'Reward_Rate']))
        if reward_rate >= 75:
            spike_data.at[cluster, 'probe_trials'] = 1
        elif reward_rate < 75:
            spike_data.at[cluster, 'probe_trials'] = 0
    return spike_data


def calculate_progression(spike_data):
    print('calculating graduation day')
    select_df = spike_data[["Mouse", "Day_numeric", "cohort","probe_trials", "Reward_Rate", "FirstStop"]]
    df = select_df.sort_values(['Mouse','Day_numeric'])

   # unique_mice = np.unique(np.array(select_df["Mouse"]))

    #for m, mcount in enumerate(unique_mice):
        #mouse_df = select_df.loc[select_df["Mouse"] == mcount, ["Mouse","Day_numeric", "probe_trials", "Reward_Rate"]]
        #df = mouse_df.sort_values('Day_numeric')

    write_to_csv(np.array(df))

    return spike_data



## Save to .csv file
def write_to_csv(csvData):
    with open('/Users/sarahtennant/Work/Analysis/Ramp_analysis/data/graduation-' + '7' + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    return

