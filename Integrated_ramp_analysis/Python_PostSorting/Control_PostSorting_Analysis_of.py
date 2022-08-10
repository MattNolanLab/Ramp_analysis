import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_SpikeHalfWidth
import Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters
import pandas as pd

prm = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.parameters.Parameters()


def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(4.7)
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots"
    initialize_parameters(save_path)
    print('Processing ' + str(save_path))

    #LOAD DATA
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/concatenated_dataframes/combined_Cohort7.pkl")
    spike_data.reset_index(drop=True, inplace=True)

    # add spike half width
    spike_data = Ramp_analysis.Integrated_ramp_analysis.Python_PostSorting.Calculate_SpikeHalfWidth.calculate_spike_width(spike_data, prm)

    # SAVE DATAFRAMES for R
    spike_data.to_pickle("/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Data/of_vr_combined_dataframes/combined_Cohort7_for_R.pkl")

if __name__ == '__main__':
    main()

