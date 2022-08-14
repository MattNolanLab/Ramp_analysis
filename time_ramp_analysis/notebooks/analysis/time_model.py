import numpy as np 
import seaborn as sns
import setting
import xarray as xr

def plotLongShortTrial(cell,speed_thres,dt,ax):
    t_short_r, spiketrain_short_r, t_long_r, spiketrain_long_r = extractTimeSpikePair(cell, dt, speed_thres)
    
    sns.lineplot(t_short_r, spiketrain_short_r,ax=ax,n_boot=100)
    sns.lineplot(t_long_r, spiketrain_long_r,ax=ax,n_boot=100)

def extractTimeSpikePair(cell, dt, speed_thres):
    short_trials = cell.short_trials
    long_trials = cell.long_trials
    short_trial_end = cell.short_trial_end
    long_trial_end = cell.long_trial_end
    speed_short = cell.speed_short
    speed_long = cell.speed_long
    spiketrain_short = cell.spiketrain_short
    spiketrain_long = cell.spiketrain_long

    t_short = np.arange(short_trial_end)*dt
    t_long = np.arange(long_trial_end)*dt

    t_short_r = np.tile(t_short,np.sum(short_trials))
    speed_short_r = speed_short.reshape(-1,1).ravel()
    short_idx = (speed_short_r>speed_thres)
    spiketrain_short_r = spiketrain_short.ravel()/dt

    t_long_r = np.tile(t_long, np.sum(long_trials))
    speed_long_r = speed_long.reshape(-1,1).ravel()
    long_idx = (speed_long_r>speed_thres)
    spiketrain_long_r = spiketrain_long.ravel()/dt
    return t_short_r, spiketrain_short_r, t_long_r, spiketrain_long_r


def extractSpikeTrainAndSpeed(totalNeuron, spiketrain, speed, trial_start_idx, reward_start_idx, trial_end):
    spiketrain_trial = np.zeros((totalNeuron,len(trial_start_idx), trial_end))
    spiketrain_trial_full = []
    speed_trial  = np.zeros((len(trial_start_idx),trial_end)) #also keep track of the speed
    speed_trial_full = []

    for n in range(totalNeuron):
        spiketrain_cell = []
        speed_cell =[]
        for i, (trial_start, reward_start) in enumerate(zip(trial_start_idx, reward_start_idx)):
            spiketrain_trial[n,i,:]=(spiketrain[trial_start:(trial_start+trial_end),n])
            spiketrain_cell.append(spiketrain[trial_start:(trial_start+reward_start),n])
            speed_cell.append(speed[trial_start:(trial_start+reward_start)])
            

        speed_trial[i,:] = (speed[trial_start:(trial_start+trial_end)])
        speed_trial_full.append(speed_cell)
        spiketrain_trial_full.append(spiketrain_cell)


    return speed_trial, spiketrain_trial, speed_trial_full, spiketrain_trial_full

def extractSpikeTrainAndSpeed2(spiketrain, speed, position, trial_start_idx, reward_start_idx, trial_end,dt):

    totalNeuron = spiketrain.shape[1]
    speed_trial =[]
    spiketrain_trial = [[]]*totalNeuron
    position_trial = []
    time_trial = []
    trial_idx =[]

    for i, (trial_start, reward_start) in enumerate(zip(trial_start_idx, reward_start_idx)):
        for n in range(totalNeuron):
            spiketrain_trial[n].append(spiketrain[trial_start:(trial_start+reward_start),n])
            
        speed_trial.append(speed[trial_start:(trial_start+reward_start)])
        pos = position[trial_start:(trial_start+reward_start)]
        position_trial.append(pos)
        time_trial.append(np.arange(len(pos))*dt)
        trial_idx.append(np.array([i]*len(pos)))

    time_trial = np.concatenate(time_trial)
    speed_trial = np.concatenate(speed_trial)
    spiketrain_trial = np.stack(spiketrain_trial)

    spiketrain_xr = xr.DataArray(data=spiketrain_trial,dims=('neuron','time'), 
        coords={'neuron':np.arange(totalNeuron), 'time':time_trial})

    trial_idx_xr = xr.DataArray(data = trial_idx, dims=('time'), coords={'time':time_trial})
    
    speed_xr = xr.DataArray(data=speed_trial, dims=('time'), coords={'time':time_trial})
    position_xr = xr.DataArray(data=position_trial, dims=('time'), coords={'time':time_trial})



    return xr.Dataset({'spiketrain': spiketrain_xr, 
        'speed':speed_xr, 'position':position_xr, 'trial_idx':trial_idx_xr})


def plotFittedTime(data, neuron_idx, length_type_to_plot, dt, ax, color,
        speed_thres = setting.speed_thres):
    # combine the array together to plot a relation plot
    # spiketrain_trial_cell: spiketrain of a single neuron

    spiketrain = data.spiketrain.data/dt
    speed = data.speed.data
    length_type = data.trial_length_type.data
    trial_num = data.trial_number
    time_relative_outbound = data.time_relative_outbound

    sel_idx = (speed > speed_thres) & (length_type==length_type_to_plot)


    sns.regplot(time_relative_outbound[sel_idx], spiketrain[sel_idx,neuron_idx],ax=ax, 
        marker='None', color = color, n_boot=50)

def plotRawTime(data, neuron_idx, length_type_to_plot, dt, ax, color,
        speed_thres = setting.speed_thres):
    # combine the array together to plot a relation plot

    spiketrain = data.spiketrain.data/dt
    speed = data.speed.data
    length_type = data.trial_length_type.data
    trial_num = data.trial_number
    time_relative_outbound = data.time_relative_outbound

    sel_idx = (speed > speed_thres) & (length_type==length_type_to_plot)


    sns.lineplot(time_relative_outbound[sel_idx], spiketrain[sel_idx,neuron_idx],ax=ax, 
        marker='None', color = color, n_boot=50)