import Python_PostSorting.LoadDataFrames
import Python_PostSorting.Calculate_SpikeHalfWidth
import Python_PostSorting.parameters

prm = Python_PostSorting.parameters.Parameters()


def initialize_parameters(recording_folder):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_file_path(recording_folder)
    prm.set_local_recording_folder_path(recording_folder)
    prm.set_output_path(recording_folder)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path= '/Users/sarahtennant/Work/Analysis/Ramp_Plots'
    initialize_parameters(save_path)
    print('Processing ' + str(save_path))

    #LOAD DATA
    spike_data = Python_PostSorting.LoadDataFrames.process_allmice_dir_of(save_path, prm) # overall data
    #spike_data = spike_data.tail(n=20)
    spike_data.reset_index(drop=True, inplace=True)

    spike_data = Python_PostSorting.Calculate_SpikeHalfWidth.calculate_spike_width(spike_data, prm)

    # SAVE DATAFRAMES for R
    spike_data.to_pickle('/Users/sarahtennant/Work/Analysis/Data/Ramp_data/WholeFrame/combined_Cohort3_of.pkl')



if __name__ == '__main__':
    main()

