import pickle
import numpy as np 
import matplotlib.pylab as plt 
from math import ceil

def saveDebug(d,filename):
    # save the dictionary as pickle for debugging

    with open(filename,'wb') as f:
        pickle.dump(d,f)

def take_first_reward_on_trial(rewarded_stop_locations,rewarded_trials,rewarded_sample_idx):
    unique_rewarded_trial = np.unique(rewarded_trials)
    locations=np.zeros((len(unique_rewarded_trial)))
    trials=np.zeros_like(locations)
    sample_idx = np.zeros_like(locations)

    for tcount, trial in enumerate(unique_rewarded_trial):
        idx = np.where(rewarded_trials == trial)[0]
        trial_locations = rewarded_stop_locations[idx]
        trial_stop_idx = rewarded_sample_idx[idx]

        if len(trial_locations) ==1:
            locations[tcount] = trial_locations
            trials[tcount] = trial
            sample_idx[tcount] = idx
        if len(trial_locations) >1:
            locations[tcount]  = trial_locations[0]
            trials[tcount] = trial
            sample_idx[tcount] = idx
    return np.array(locations), np.array(trials)


def getWrappedSubplots(ncol,total_n,figsize,**kwargs):
    """Create wrapped subplots

    Args:
        ncol (int): number of columns 
        total_n (int): total number of subplots
        figsize ((width,height)): tuple of figsize in each plot

    Returns:
        [(fig,axes)]: wrapped subplots axes handle
    """
    #Automatically create a wrapped subplots axis
    nrow = ceil(total_n/ncol)

    fig,ax = plt.subplots(nrow, ncol, figsize = (figsize[0]*ncol, figsize[1]*nrow), **kwargs)

    return fig, ax.ravel()