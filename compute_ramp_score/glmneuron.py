import collections as collections_abc
import itertools
import pickle
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import xarray as xr
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import pearsonr, wilcoxon
from tqdm import tqdm
import setting
import utils


def generateCombinations(variables):
    # generate different combination of variables for analysis
    x = []
    for i in range(len(variables)):
        x.append(list(itertools.combinations(variables, i + 1)))
    return x


def getVariableCombinations(dataset, variables):

    variables_name = generateCombinations(variables)
    variables_list = []
    # print(variables_name)
    for lv in variables_name:
        for v in lv:
            x = np.hstack([dataset[vv].data for vv in v])
            variables_list.append(x)

    variables_name = [' '.join(x) for y in variables_name for x in y]
    return variables_list, variables_name

def convert2xarray(data, result_dict, variable):
    # Convert dictionary of list to xarray
    xr_dict = {}
    for k in result_dict.keys():
        array = result_dict[k]

        dataArray = xr.DataArray(
            array,
            dims=("model1", "neuron", "trial_type"),
            coords={
                "model1": variable,
                "neuron": data.neuron,
                "trial_type": ["all", "beaconed", "non-beaconed", "probe"],
            },
        )
        xr_dict[k] = dataArray

    resultDataset = xr.Dataset(xr_dict)
    resultDataset.coords["pos_bins"] = data.pos_bins
    resultDataset.coords["speed_bins"] = data.speed_bins
    resultDataset.coords['accel_bins'] = data.accel_bins


    return resultDataset


def shuffle_data(dataset, shufflefield,keepfield=None):
    # shuffle the dataset while keeping the keepfield the same
    shuffle_data = dataset.copy(deep=True)

    if keepfield is not None:
        # keep the keepfield the same but shuffle other data
        pos = np.argmax(shuffle_data[keepfield].data, axis=1)
        posBin = shuffle_data[keepfield].shape[1]
        for i in range(posBin):
            oldIdx = np.where(pos == i)[0]
            for s in shufflefield:
                newIdx = np.random.permutation(oldIdx)
                shuffle_data[s].data[oldIdx, :] = shuffle_data[s].data[newIdx, :]
    else:
        #shuffle everything in the shufflefield
        for s in shufflefield:
            data2shuffle = shuffle_data[s].data
            newIdx = np.random.permutation(data2shuffle.shape[0])
            shuffle_data[s].data = data2shuffle[newIdx,:]

    return shuffle_data


def mergeSeriesOnIndex(data, index):
    # merge the data together based on the index, keep the last one for duplicate

    if len(data) > 1:
        for i, tt in enumerate(data):
            tt = np.array(tt)
            dataLength = min(tt.shape[0], index.iloc[i].shape[0])
            if i == 0:
                # sometimes the cell may stil fire after the trial end
                series = pd.Series(tt[:dataLength], index=index.iloc[i][:dataLength])
            else:
                s = pd.Series(tt[:dataLength], index=index.iloc[i][:dataLength])
                series.append(s)
                uniqueIdx = series.index.drop_duplicates(
                    keep="last"
                )  # remove duplicate time indices
                series = series[uniqueIdx]

    else:
        data = np.array(data.iloc[0])
        index = np.array(index.iloc[0])
        dataLength = min(data.shape[0], index.shape[0])
        series = pd.Series(data[:dataLength], index=index[:dataLength])

    return series


def addZeroStartTime(xDataArray):
    # add a zero start time at the beginning
    # for data alignment during resample later
    zero_start = xr.DataArray(
        [np.nan], dims=("time"), coords={"time": pd.TimedeltaIndex([0], unit="s")}
    )
    xDataArray = xr.concat([zero_start, xDataArray], dim="time")
    return xDataArray


def append2series(
    series, data: list, index: list, indexType=pd.TimedeltaIndex, unit="s"
):
    # append data to a series
    s = pd.Series(data, index=indexType(data=np.array(index), unit=unit))
    return series.append(s)


def getSpikePopulationXR(spiketrain, Fs):
    # Combine and return the spike train of a population as xarray

    # combine all spike firings together
    spiketrains = np.concatenate(list(spiketrain))
    index = np.append(
        np.zeros((1,)), np.unique(spiketrains)
    )  # add 0 to align timestamp later

    populationMatrix = np.zeros((index.size, len(spiketrain)))

    array = xr.DataArray(
        populationMatrix,
        dims=("time", "neuron"),
        coords={
            "time": pd.TimedeltaIndex(index / Fs, unit="s"),
            "neuron": np.arange(len(spiketrain))
        },
    )

    # insert the spike in the population matrix
    for i, st in enumerate(spiketrain):
        deltaIdx = pd.TimedeltaIndex(st / Fs, unit="s") #convert to proper time
        array.loc[deltaIdx, i] = 1

    return array


def getSpikeTrainPopulation(spiketrain, Fs):
    # return a pandas series containing all spikes from all cells
    spiketrainList = []

    # combine the spike trains
    for st in spiketrain:
        ps = pd.Series(
            np.ones(st.size), index=pd.TimedeltaIndex(data=st / Fs, unit="s")
        )
        ps = append2series(ps, [0], [0])
        spiketrainList.append(ps)

    if len(spiketrainList) > 0:
        df = pd.concat(spiketrainList, axis=1)  # combine to one big dataframe
        df = df.fillna(0)  # fill the nan with 0
        data = [x for x in df.to_numpy()]
        return pd.Series(data, index=df.index)
    else:
        return pd.Series()


def gaussfilter(x, mu, sigma):
    a = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    a = a / a.sum(axis=0)
    return a


def addKeyPrefix(d: dict, prefix):
    # add a prefix to the key in the dictionary
    for k in list(d.keys()):
        d[prefix + k] = d.pop(k)
    return d


def computeModelTuning(pos_param=None, hd_param=None, speed_param=None):
    # Compute the neuronal tuning based on models

    # Calculate the scale factor, assuming the other variables are at their mean rate
    scale_factor_pos = np.exp(speed_param).mean() * np.exp(hd_param).mean()
    scale_factor_hd = np.exp(speed_param).mean() * np.exp(pos_param).mean()
    scale_factor_speed = np.exp(speed_param).mean() * np.exp(pos_param).mean()

    # calculate the response curve
    pos_response = scale_factor_pos * np.exp(pos_param)
    hd_response = scale_factor_hd * np.exp(hd_param)
    speed_response = scale_factor_speed * np.exp(speed_param)

    return (pos_response, hd_response, speed_response)


def getTrainTestSet(x, trainPercent=0.7):
    # separate training set and testing set

    if x.shape[1] > x.shape[0]:
        x = x.T

    split = int(x.shape[0] * 0.7)

    if x.ndim == 1 or x.shape[1] == 1:
        return (x[:split], x[split:])
    else:
        return (x[:split, :], x[split:, :])


def getCrossValidationFold(x, totalFold, foldNo):
    # breakdown the input into totalFold number of fold, and return the training and testing set
    # The folderNo starts from 0

    # First dim is the sample axis (or time)
    if x.ndim > 1 and x.shape[1] > x.shape[0]:
        x = x.T

    testSize = x.shape[0] // totalFold
    testIdx = np.arange(foldNo * testSize, foldNo * testSize + testSize)

    trainIdx = np.setdiff1d(np.arange(x.shape[0]), testIdx)

    if x.ndim == 1 or x.shape[1] == 1:
        return (x[trainIdx], x[testIdx])
    else:
        return (x[trainIdx, :], x[testIdx, :])


def find_param(param, modelType, numPos, numHD, numSpd, numTheta):

    # model type : [position, head_direction, speed, theta]
    # TODO cater for different number of bins

    param_pos = None
    param_hd = None
    param_spd = None
    param_theta = None

    if np.all(modelType == [1, 0, 0, 0]):
        param_pos = param
    elif np.all(modelType == [0, 1, 0, 0]):
        param_hd = param
    elif np.all(modelType == [0, 0, 1, 0]):
        param_spd = param
    elif np.all(modelType == [0, 0, 0, 1]):
        param_theta = param

    elif np.all(modelType == [1, 1, 0, 0]):
        param_pos = param[:numPos]
        param_hd = param[numPos : numPos + numHD]
    elif np.all(modelType == [1, 0, 1, 0]):
        param_pos = param[:numPos]
        param_spd = param[numPos : numPos + numSpd]
    elif np.all(modelType == [1, 0, 0, 1]):
        param_pos = param[:numPos]
        param_theta = param[numPos : numPos + numTheta]
    elif np.all(modelType == [0, 1, 1, 0]):
        param_hd = param[:numHD]
        param_spd = param[numHD : numHD + numSpd]
    elif np.all(modelType == [0, 1, 0, 1]):
        param_hd = param[:numHD]
        param_theta = param[numHD + 1 : numHD + numTheta]
    elif np.all(modelType == [0, 0, 1, 1]):
        param_spd = param[:numSpd]
        param_theta = param[numSpd + 1 : numSpd + numTheta]

    elif np.all(modelType == [1, 1, 1, 0]):
        param_pos = param[:numPos]
        param_hd = param[numPos : numPos + numHD]
        param_spd = param[numPos + numHD : numPos + numHD + numSpd]
    elif np.all(modelType == [1, 1, 0, 1]):
        param_pos = param[:numPos]
        param_hd = param[numPos : numPos + numHD]
        param_theta = param[numPos + numHD : numPos + numHD + numTheta]
    elif np.all(modelType == [1, 0, 1, 1]):
        param_pos = param[:numPos]
        param_spd = param[numPos : numPos + numSpd]
        param_theta = param[numPos + numSpd : numPos + numSpd + numTheta]
    elif np.all(modelType == [0, 1, 1, 1]):
        param_hd = param[:numHD]
        param_spd = param[numHD : numHD + numSpd]
        param_theta = param[numHD + numSpd : numHD + numSpd + numTheta]
    elif np.all(modelType == [1, 1, 1, 1]):
        param_pos = param[:numPos]
        param_hd = param[numPos : numPos + numHD]
        param_spd = param[numPos + numHD : numPos + numHD + numSpd]
        param_theta = param[
            numPos + numHD + numSpd : numPos + numHD + numSpd + numTheta
        ]

    return (param_pos, param_hd, param_spd, param_theta)


def pos_map(pos, nbins, boxSize):
    # discretize the position into different bins
    # pos: nx2 matrix
    bins = np.arange(boxSize / nbins, boxSize, boxSize / nbins)
    xcoord = np.digitize(pos[:, 0], bins)
    ycoord = np.digitize(pos[:, 1], bins)
    # make x and y start form the top left corner
    coord = (xcoord) * nbins + (nbins - ycoord - 1)

    output = np.zeros((pos.shape[0], nbins ** 2))
    output[np.arange(pos.shape[0]), coord] = 1

    return (output, bins)


def pos_1d_map(pos, nbins, boxSize):
    # discretize the position into different bins
    # pos: nx2 matrix
    bins = np.arange(
        boxSize / nbins, boxSize + 1, boxSize / nbins
    )  # make sure it covers the last bin
    coord = np.digitize(pos, bins)
    output = np.zeros((pos.shape[0], nbins))
    output[np.arange(pos.shape[0]), coord] = 1

    return (output, bins)


def theta_map(phase, nbins):
    """Discretize theta phase
    
    Arguments:
        phase {np.narray} -- phase of EEG
        nbins {int} -- number of bins
    
    Returns:
        (np.narray,np.narray) -- (binned output, bin used)
    """
    bins = np.arange(2 * np.pi / nbins, np.pi, np.pi / nbins)
    coord = np.digitize(phase, bins, right=True)
    output = np.zeros((phase.shape[0], nbins))
    output[np.arange(phase.shape[0]), coord] = 1

    return (output, bins)


# filter at theta band
def extract_theta_phase(eeg, Wn, Fs):
    """Filter the EEG to theta band and extract its phase
    
    Arguments:
        eeg {np.narray} -- Raw EEG signal in form (time x channel)
        Wn {np.narray} -- upper and lower filter corner frequency
        Fs {float} -- samplig frequency
    
    Returns:
        np.narray -- phase of the signal
    """
    # Filter signal
    (b, a) = signal.butter(5, Wn / (Fs * 2), "bandpass")
    eegFilt = signal.filtfilt(b, a, eeg, axis=0)

    # Hilbert transform to find instantaneous phase
    eegHilbert = signal.hilbert(eegFilt, axis=0)
    phase = np.arctan2(np.imag(eegHilbert), np.real(eegHilbert))
    for i in range(phase.shape[1]):
        ind = np.where(phase[:, i] < 0)
        phase[ind, i] = phase[ind, i] + 2 * np.pi

    return phase


def multivariate_piecewise(x, condlist, funclist, multivariate=False, *args, **kw):
    # Stolen from numpy with some modification to work with multivariate input
    x = np.asanyarray(x)
    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if np.isscalar(condlist) or (
        not isinstance(condlist[0], (list, np.ndarray)) and x.ndim != 0
    ):
        condlist = [condlist]

    condlist = np.array(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected".format(
                n, n, n + 1
            )
        )

    if multivariate:
        y = np.zeros((x.shape[0],), x.dtype)
    else:
        y = np.zeros(x.shape, x.dtype)

    for k in range(n):
        item = funclist[k]
        if not isinstance(item, collections_abc.Callable):
            y[condlist[k]] = item
        else:
            if multivariate:
                vals = x[condlist[k], :]
            else:
                vals = x[condlist[k]]
            if vals.size > 0:
                y[condlist[k]] = item(vals, *args, **kw)

    return y


def hd_map(posx, posx2, posy, posy2, nbins):
    direction = np.arctan2(posy2 - posy, posx2 - posx) + np.pi / 2
    direction[direction < 0] = direction[direction < 0] + 2 * np.pi
    direction = direction.ravel()  # change to 1d array

    hd_grid = np.zeros((posx.shape[0], nbins))
    dirVec = np.arange(2 * np.pi / nbins, 2 * np.pi, 2 * np.pi / nbins)
    idx = np.digitize(direction, dirVec)
    hd_grid[np.arange(posx.shape[0]), idx] = 1

    return (hd_grid, dirVec, direction)


def speed_map(posx, posy, nbins, sampleRate=50, maxSpeed=50):
    velx = np.diff(np.insert(posx, 0, posx[0]))
    vely = np.diff(np.insert(posy, 0, posy[0]))
    speed = np.sqrt(velx ** 2 + vely ** 2) * sampleRate
    # send everything over 50 cm/s to 50 cm/s
    speed[speed > maxSpeed] = maxSpeed

    speedVec = np.arange(maxSpeed / nbins, maxSpeed + 1, maxSpeed / nbins)
    speed_grid = np.zeros((posx.shape[0], nbins))

    idx = np.digitize(speed, speedVec)
    speed_grid[np.arange(speed.shape[0]), idx.ravel()] = 1

    return (speed_grid, speedVec, speed, idx)


def make_1d_map(x, nbins,minBin,maxBin):
    x[x>maxBin] = maxBin #make sure everything is within range
    x[x<minBin] = minBin

    binSize = (maxBin-minBin)/nbins
    vec = np.arange(minBin+binSize, maxBin + binSize, binSize) # make sure it covers the last bin

    coord = np.digitize(x, vec,right=True)
    grid = np.zeros((x.shape[0], nbins))
    grid[np.arange(x.shape[0]), coord] = 1

    return (grid, vec, x)


def speed_map_1d(pos, nbins, sampleRate=50, maxSpeed=50, minSpeed=0, removeWrap=True):
    """bin and map the speed into one-hot vector
    
    Arguments:
        pos {np.narray} -- position of the animal, 1d array
        nbins {int} -- number of bins
    
    Keyword Arguments:
        sampleRate {int} -- sampling rate of signal (default: {50})
        maxSpeed {int} -- maximum speed (default: {50})
        removeWrap {bool} -- whether to remove the point going from the end back to the start (default: {False})
    
    Returns:
        {tuple} -- (one hot vector, bin edges, speed )
    """
    vel = np.diff(np.insert(pos, 0, pos[0]))
    speed = vel * sampleRate
    # send everything over 50 cm/s to 50 cm/s
    speed[speed > maxSpeed] = maxSpeed
    if removeWrap:
        idx = np.where(speed < minSpeed)[0]
        for i in idx:
            speed[i] = speed[
                i - 1
            ]  # assign the teleport speed right before the teleport

    speedVec = np.arange(maxSpeed / nbins, maxSpeed + 1, maxSpeed / nbins)
    speed_grid = np.zeros((pos.shape[0], nbins))

    idx = np.digitize(speed, speedVec, right=True)
    speed_grid[np.arange(speed.shape[0]), idx.ravel()] = 1

    return (speed_grid, speedVec, speed)


def calculate_track_location(recorded_location, track_length):
    print("Converting raw location input to cm...")
    recorded_startpoint = np.min(recorded_location)
    recorded_endpoint = np.max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = (
        recorded_track_length / track_length
    )  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    return location_in_cm  # fill in dataframe


def make_bin(spiketrain, binEdge):
    """
    Count spike in bin, specified by the binSize
    """
    idx = np.digitize(spiketrain, binEdge)
    spiketrain_bin = np.zeros((binEdge.shape[0], 1))
    for i in range(idx.shape[0]):
        if idx[i] < binEdge.size:  # discard the last bin
            spiketrain_bin[idx[i]] += 1
    return spiketrain_bin


def average_in_bin(position, binSize):
    l = int(
        position.size // binSize * binSize
    )  # discard data not enough to fill the binsize
    p = position[:l]
    p = p.reshape(
        int(p.shape[0] // binSize), int(binSize)
    )  # the reshape begins at the last indices
    p = p.mean(axis=1)

    return p, l


def plotRawTuningCurve(ax, vec, count, response, title=None):

    ax.plot(vec, count, "ro")
    ax.plot(vec, response)
    ax.legend(["Testing set", "Model"])
    if title:
        ax.set_title(title)


def get_smooth_fr(spiketrain, halfWidth, dt):
    # return a smoothed spike train

    f = gaussfilter(np.arange(-halfWidth + 1, halfWidth), 0, 10)
    fr = np.array(spiketrain).flatten() / dt
    fr_smooth = np.convolve(fr, f, mode="same")
    return fr_smooth


def findTeleportPoint(position, maxspeed=50):
    s = np.diff(position)
    return np.where(s < -maxspeed)[0]


def getModelPerformance(datagrid, spiketrain, spiketrain_hat, dt):
    fr_param = get_smooth_fr(spiketrain_hat, 5, dt)
    fr = get_smooth_fr(spiketrain, 5, dt)

    # variance explained
    sse = np.sum((fr_param - fr) ** 2)
    sst = np.sum((fr - np.mean(fr)) ** 2)

    varExplain = 1 - (sse / sst)

    # correlation
    (correlation, corr_p) = pearsonr(fr_param, fr)

    # log likelihood
    r_train = np.array(spiketrain_hat)  # predicted
    n_train = np.array(spiketrain)  # true spike
    meanFR_train = np.mean(n_train)
    log_llh_train_model = np.sum(
        r_train - n_train * np.log(r_train) + np.log(factorial(n_train))
    ) / np.sum(n_train)
    log_llh_train_mean = np.sum(
        meanFR_train - n_train * np.log(meanFR_train) + np.log(factorial(n_train))
    ) / np.sum(n_train)
    log_llh = (-log_llh_train_model + log_llh_train_mean) * np.log(2)

    # mse
    mse = np.mean(((fr_param - fr) ** 2))

    return {
        "spiketrain_smoothed_est": fr_param,
        "spiketrain_smoothed": fr,
        "varExplain": varExplain,
        "correlation": correlation,
        "corr_p": corr_p,
        "log_llh": log_llh,
        "mse": mse,
    }


def getVarExplain(fr_param,fr):
    # fr_param: estimated firing rate
    # fr: true firing rate
    sse = np.sum((fr_param - fr) ** 2)
    sst = np.sum((fr - np.mean(fr)) ** 2)

    varExplain = 1 - (sse / sst)

    return varExplain

def getLogLL(spiketrain_est,spiketrain):
    r_train = np.array(spiketrain_est)  # predicted
    n_train = np.array(spiketrain)  # true spike
    meanFR_train = np.mean(n_train)
    log_llh_train_model = np.sum(
        r_train - n_train * np.log(r_train) + np.log(factorial(n_train))
    ) / np.sum(n_train)
    log_llh_train_mean = np.sum(
        meanFR_train - n_train * np.log(meanFR_train) + np.log(factorial(n_train))
    ) / np.sum(n_train)
    log_llh = (-log_llh_train_model + log_llh_train_mean) * np.log(2)

    return log_llh

def getMSE(fr_param,fr):
    mse = np.mean(((fr_param - fr) ** 2))
    return mse


def compare_model_performance(param, datagrid, spiketrain_bin, dt):
    # print(param.shape)
    # print(datagrid.shape)
    # print(spiketrain_bin.shape)

    #TODO: the shape of param may affect the result significantly
    spiketrain_hat = np.exp(datagrid * np.matrix(param).T)
    fr_param = get_smooth_fr(spiketrain_hat, 5, dt) #firing estimated from model
    fr = get_smooth_fr(spiketrain_bin, 5, dt) #true firing

    # variance explained
    sse = np.sum((fr_param - fr) ** 2)
    sst = np.sum((fr - np.mean(fr)) ** 2)

    varExplain = 1 - (sse / sst)

    # correlation
    (correlation, corr_p) = pearsonr(fr_param, fr)

    # log likelihood
    r_train = np.array(spiketrain_hat)  # predicted
    n_train = np.array(spiketrain_bin)  # true spike
    meanFR_train = np.mean(n_train)
    log_llh_train_model = np.sum(
        r_train - n_train * np.log(r_train) + np.log(factorial(n_train))
        ) / np.sum(n_train)

    log_llh_train_mean = np.sum(
        meanFR_train - n_train * np.log(meanFR_train) + np.log(factorial(n_train))
        ) / np.sum(n_train)
    log_llh = (-log_llh_train_model + log_llh_train_mean) * np.log(2)

    # mse
    mse = np.mean(((fr_param - fr) ** 2))

    return {
        "spiketrain_smoothed_est": fr_param,
        "spiketrain_smoothed": fr,
        "varExplain": varExplain,
        "correlation": correlation,
        "corr_p": corr_p,
        "log_llh": log_llh,
        "mse": mse,
        "param": param,
    }


def getVariableIdx(best_model,position_bin = setting.position_bin, 
    speed_bin=setting.speed_bin, accel_bin=setting.accel_bin):
    # Find the idx of each input parameters

    variable_idx = []
    for model in best_model.split():
        # Determine where the index should start from
        if len(variable_idx) == 0:
            startIdx = 0
        else:
            startIdx = variable_idx[-1][-1]

        if "pos" in model:
            variable_idx.append(startIdx + np.arange(position_bin))
        elif "speed" in model:
            variable_idx.append(startIdx + np.arange(speed_bin))
        elif "accel" in model:
            variable_idx.append(startIdx + np.arange(accel_bin))

    return variable_idx

def findDominantVar(model,data,spkt,dt):
    # Find the dominant variable in the best model
    # The dominant variable is defined as the one that is significantly
    # better than the others according to log-likelihood
    # If no dominant variable is found, the dominant variable is undefined

    # Extract model curve
    best_model = model.best_model.data[0]
    model_curve = model.sel(
                        trial_type="all", model1=best_model
                    ).train_model_curve

    if model_curve.data.item(0) is None:
        # Early exit
        return None

    #extract raw data
    data_list = []

    for model in best_model.split():
        data_list.append(data[model].data)

    data_grid = np.hstack(data_list)

    variable_idx = getVariableIdx(best_model)
    model_param = np.log(model_curve.data.item(0) * dt)
    # Calculate scores
    valFolds = setting.valFolds
    score_list = np.zeros((valFolds, len(variable_idx)))

    for i in range(valFolds):
        (datagrid_train, datagrid_test) = getCrossValidationFold(
            data_grid, valFolds, i
        )
        (spiketrain_train, spiketrain_test) = getCrossValidationFold(
            spkt, valFolds, i
        )

        # Evaluate model performance in each variable individually
        for j, idx in enumerate(variable_idx):
            s = compare_model_performance(
                model_param[i, idx], datagrid_test[:, idx], spiketrain_test.reshape(-1,1), dt
            )
            score_list[i, j] = s["log_llh"]


    #%% Find if any variable is significantly better than the others
    nVar = score_list.shape[1]
    isDominant = np.empty((nVar))
    for i,vIdx in enumerate(range(score_list.shape[1])):
        isDominant[i] = True
        for compareIdx in np.setdiff1d(range(score_list.shape[1]),vIdx):
            r=wilcoxon(score_list[:,vIdx], score_list[:,compareIdx],alternative='greater')
            if r.pvalue > 0.05:
                isDominant[i] = False

    assert np.sum(isDominant) <= 1 #can only have 1 dominant variable

    if np.any(isDominant):
        dominantModel = best_model.split()[int(np.where(isDominant)[0])]
    else:
        dominantModel = 'Undefined'
    
    return dominantModel

def plotTuningCurves2(
    fig,
    grid,
    param,
    firing_rate,
    xtick,
    title,
    xlabel,
    ylabel,
    ste=None,
    metrics_dict: dict = {},
):
    # plot the raw firing rate and model curve
    ax = fig.add_subplot(grid)
    ax.plot(xtick, param)

    if not ste is None:
        # plot the standard error instead of raw points
        ax.fill_between(
            xtick,
            firing_rate - 1.96 * ste,
            firing_rate + 1.96 * ste,
            color="gray",
            alpha=0.2,
        )
    else:
        ax.plot(xtick, firing_rate, "ro")
        # ax.legend(['Testing set','Model'])

    info = ""

    for k, v in metrics_dict.items():
        info += f"{k}:{v:.2f}\n"


    ax.set_title(title + "\n" + info)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plotTuningCurves(
    datagrid,
    spiketrain,
    result,
    performance,
    vec,
    tuningXLabel="Position",
    figure_path=None,
):
    plt.close()

    varExplain = performance["varExplain"]
    correlation = performance["correlation"]
    log_llh_train = performance["log_llh_train"]
    mse = performance["mse"]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"(VarExplain: {varExplain:.2f}, corr: {correlation:.2f}, log_L: {log_llh_train:.2f}, MSE: {mse:.2f}) "
    )

    ax[0].plot(vec, performance["firing_rate"], "ro")
    ax[0].plot(vec, performance["model_curve"])
    ax[0].legend(["Testing set", "Model"])

    ax[0].set_ylabel("Firing rate")
    ax[0].set_xlabel(tuningXLabel)

    pltMaxIdx = min(5000, performance["spiketrain_smoothed_est"].size)
    pltIdx = (np.arange(0, pltMaxIdx),)
    ax[1].plot(performance["spiketrain_smoothed_est"][pltIdx])
    ax[1].set_title("Model firing rate")
    ax[2].plot(performance["spiketrain_smoothed"][pltIdx])
    ax[2].set_title("Real firing rate")

    ax[3].plot(np.argmax(datagrid, axis=1)[pltIdx])  # plot the raw trajectory
    ax[3].set_title(tuningXLabel)

    if figure_path:
        fig.savefig(figure_path)


def correct_for_restart(location):
    location[
        location < 0.55
    ] = 0.56  # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


def plotTuningCurveSubset(
    fig, grid, data, neuron, variable, trial_type, subset, title, selFold=0
):
    d = data.sel(neuron=neuron, model1=variable+'_grid', trial_type=trial_type)
    if not d.train_model_curve.data.item(0) is None:
        param = d.train_model_curve.data.item(0)[
            selFold, :
        ]  # only use the model from all data
        fr = d.test_firing_rate.data.item(0)[selFold, :]
        correlation = d.test_correlation.data.item(0)[selFold]
        llg = d.test_log_llh.data.item(0)[selFold]
        varExplain = d.test_varExplain.data.item(0)[selFold]
        ste = d.test_firing_rate_se.data.item(0)[selFold]

        xtick = d[variable + "_bins"]
        plotTuningCurves2(
            fig,
            grid,
            param,
            fr,
            xtick,
            title,
            variable,
            "Firing rate",
            metrics_dict={"correlation": correlation, "vae": varExplain, "llg": llg},
        )
    else:
        # there may not be enough other trial type to plot, plot the overall type instead
        if trial_type == "beaconed":
            title = variable + "-all"
            plotTuningCurveSubset(
                fig, grid, data, neuron, variable, "all", subset, title
            )


def compareModel(resultDataset):
    """Select the best model based on the log-likelihood
    
    Arguments:
        resultDataset {xarray.Dataset} -- dataset containing the fitting models
        variables_name -- name of the variable in a list, each item of the list should contain a tuple saying which variables are in the model

    Returns:
        resultDataset -- dataset containing the fitted model, with comparison matrix added
    """

    n_model = len(resultDataset.model1)
    mean_llh = np.empty((len(resultDataset.neuron),n_model))
    model_comparison_pvalue = np.empty((len(resultDataset.neuron),n_model, n_model))

    for i, n in enumerate(resultDataset.neuron):
        # only use log-likehood for model selection

        log_llh = resultDataset.test_log_llh.sel(neuron=n, trial_type="all")
        log_llh = np.stack(log_llh.data)

        if not log_llh[0] is None:
            mean_llh[i,:] = log_llh.mean(axis=1)

            # compare between all models
            for j in range(n_model):
                for k in range(n_model):
                    if j != k:
                        model_comparison_pvalue[i, j, k] = wilcoxon(
                            log_llh[j, :], log_llh[k, :]
                        ).pvalue

    # add to the analysis dataset
    resultDataset["mean_llh"] = xr.DataArray(
        mean_llh,
        dims=("neuron", "model1"),
        coords={"neuron": resultDataset.neuron, "model1": resultDataset.model1.data},
    )

    resultDataset["model_comparison_pvalue"] = xr.DataArray(
        model_comparison_pvalue,
        dims=("neuron", "model1", "model2"),
        coords={"neuron": resultDataset.neuron, 
        "model1": resultDataset.model1.data,
        'model2': resultDataset.model1.data} #xarray is having problem with dimension with the same name
    )

    return resultDataset


def addBestModel(dataset):
    # add the calculate best model to the dataset
    
    best_models = []
    #add the best model to 
    for i in dataset.neuron:
        mean_llh = dataset['mean_llh'].sel(neuron=i).data
        comparison_matrix = dataset['model_comparison_pvalue'].sel(neuron=i).data
        best_model = selectBestModel(dataset.model1.data, mean_llh, comparison_matrix)
        best_models.append(best_model)
    
    
    dataset['best_model'] = xr.DataArray(
        best_models,
        dims = ('neuron','comparison_result'),
        coords = {'neuron': dataset.neuron, 'comparison_result': ['model','isAbsoluteBest']}
    )

    return dataset


def plotHBarWithError(xtick,width, error,title,ax=None):
    # plot horizontal bar chart with error
    if ax is None:
        fig,ax = plt.subplots(1,1)
    ax.barh(xtick,width,xerr=error)
    ax.set_title(title)


def selectBestModel(variable_names, mean_llh, comparison_matrix):
    # Select the best and most simple model based on the comparison matrix
    model_order = np.array([len(v.split()) for v in variable_names])
    order_best_idx = None
    order_isAbsolute_best = None

    for order in range(model_order.max()):
        # First do comparison respectively in each order
        idx2compare = np.where(model_order == (order+1))[0]
        best_idx = idx2compare[np.argmax(mean_llh[idx2compare])]

        #find out if a single model absolutely better than the others
        idx = np.setdiff1d(idx2compare,best_idx)
        if len(idx) > 0:
            isAbsoluteBest = np.all(comparison_matrix[best_idx,idx]<0.01)
        else:
            isAbsoluteBest = None

        if order_best_idx:
            #compare between the current order and the lower order
            if comparison_matrix[best_idx,order_best_idx] >= 0.01:
                # No significant difference between more complex model and simple model, stop comparison
                    break
            else:
                order_best_idx = best_idx
                order_isAbsolute_best = isAbsoluteBest
        else:
            order_best_idx = best_idx
            order_isAbsolute_best = isAbsoluteBest

    
    return variable_names[order_best_idx],order_isAbsolute_best









def balanceSamples(classlabel, val2check=None, checkfunc=np.mean, minDiff=0.01):
    """Balance data with the supplied classslabel
    
    Arguments:
        classlabel {np.narray} -- array containing the label for each sample
    
    Keyword Arguments:
        val2check {np.narray} -- an variable to check to make sure the sampled and original population are similar (default: {None})
        checkfunc {func} -- function to check the val2check for similarity
        minDiff {float} -- minimium difference in ratio between the sampled and original population
    """

    uniq_values, counts = np.unique(classlabel, return_counts=True)
    minCount = np.min(counts)
    resampled_idx = []
    for v in uniq_values:
        idx = np.where(classlabel == v)[0]
        selIdx = np.random.choice(idx, size=(minCount,), replace=False)  # sample sample

        # Check to make sure the sampled values are similar to the original one
        if not val2check is None:
            origVal = checkfunc(val2check[idx])
            sampledVal = checkfunc(val2check[selIdx])
            retryCount = 0

            # Resample if it doesn't mean the matching criteria
            while np.abs(sampledVal - origVal) / origVal > minDiff:
                selIdx = np.random.choice(idx, size=(minCount,), replace=False)
                origVal = checkfunc(val2check[idx])
                sampledVal = checkfunc(val2check[selIdx])
                retryCount += 1
                if retryCount > 100:
                    warnings.warn("Cannot find matching sample within 100 iterations")
                    break

        resampled_idx.append(selIdx)

    resampled_idx = np.sort(np.concatenate(resampled_idx))
    return resampled_idx


if __name__ == "__main__":
    data = sio.loadmat("data_for_cell77")

    posc = np.hstack([data["posx_c"], data["posy_c"]])
    (posgrid, posVec) = pos_map(posc, 20, 100)

    (hdgrid, dirVec, direction) = hd_map(
        data["posx"], data["posx2"], data["posy"], data["posy2"], 18
    )

    (speedgrid, speedVec, speed, idx) = speed_map(data["posx_c"], data["posy_c"], 10)

    spiketrain = data["spiketrain"]
    print(spiketrain.shape)
    datagrid = np.matrix(np.hstack([hdgrid, speedgrid, posgrid]))

    glmPostProcessor = GLMPostProcessor(spiketrain, datagrid, [1, 1, 1, 0], speed)

    glmPostProcessor.run()
    glmPostProcessor.cleanup()
