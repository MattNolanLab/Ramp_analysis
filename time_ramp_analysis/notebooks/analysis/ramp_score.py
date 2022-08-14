import numpy as np 
from imblearn.over_sampling import RandomOverSampler
import tqdm
from scipy.signal import savgol_filter
import sklearn.linear_model as lm 
import matplotlib.pylab as plt
import warnings

def makeLinearGrid(gridpts):
    #girdpts should be a list, with each element a np.ndnarray or list containing the coordinate of the grid
    grid = np.meshgrid(*gridpts)
    grid = [g.ravel() for g in grid] #linearize the grid matrix
    return list(zip(*grid)) #make each combination a tupe


def findBreakPt(curve, minDist=2):
    #identify break point in a curve
    # break points include also include turning point

    d = np.sign(np.diff(curve,prepend=curve[0]))
    #need to check if any of the two points are non-zero, otherwise flat line will all be idenitifed as turning points

    bp = np.where((d[1:]*d[:-1]<=0) & (d[1:].astype(np.bool) | d[:-1].astype(np.bool)))[0]

    bp = bp[np.diff(bp,prepend=-minDist*2)>=minDist] #remove the points close together, make sure the last point is not removed
    bp = np.append(bp,len(curve)-1)

    return bp

def findBestRampScore2(spiketrain,position,pos_bins,minDist=2):
    """Attempt to find the best ramp score by performing a grid search on the break points
    return the best estimated score, and the break point used to calculate those scores
    all span shorter than the minDist will be assigned a score of 0
    the returned breakpoint index is referinng to the index of the pos_bins
    the Ramp score can be zero if the firing rate is all zeros

    Args:
        spiketrain (np.ndarray): 1D time-binned spike train
        position (np.ndarray): binned position in the same shape as spiketrain
        pos_bins (np.ndarray): position bins used to calculate the average firing rate for each position
        minDist (int, optional): minimal size of segment to consider. Defaults to 2.

    Returns:
        (tuple): a tuple of in the form of 
        (best ramp score find, 
            index of the ramp score region, 
            smoothed original curve used to find the inflection points,
            and the normalized spiketrain used to calculate the ramp score)
    """


    fr = np.zeros(pos_bins.shape)
    for i,p in enumerate(pos_bins):
        meanfr = np.nanmean(spiketrain)
        spk_train_pos = spiketrain[position==p]
        if len(spk_train_pos)>0:
            fr[i] = np.nanmean(spk_train_pos)
        else:
            # if can't find spike in a particular location
            # set the firing rate to be the meanig firing rate
            fr[i] = meanfr
            
        if np.isnan(fr[i]): #can happen if no spike found in the bin
            fr[i] = meanfr 

    # print(fr)
    meanCurve  = savgol_filter(fr, 7,1)

    bp = findBreakPt(meanCurve,minDist) #the unit of bp should be equal to the unit of position
    bp_comb = makeLinearGrid([bp,bp])

    # Balance the spike and position data
    try:
        # in some rare case there may not be enough data for resample, in that case
        # omit resampling
        sampler = RandomOverSampler()
        spiketrain_re, position_re = sampler.fit_resample(spiketrain.reshape(-1,1), position)
        spiketrain_re = spiketrain_re.ravel()
        position_re = position_re.ravel()
    except ValueError:
        warnings.warn('Error countered. Skipped performing resampling.')
        spiketrain_re = spiketrain
        position_re = position

    #Search for the best region to calculate the ramp score
    scores = np.zeros((len(bp_comb),))
    for i in range(len(bp_comb)):
        scores[i] , normCurve = getRampScore4(pos_bins[list(bp_comb[i])],
            spiketrain_re, position_re, pos_bins.max()-pos_bins.min())
        

    best_idx = np.argmax(np.abs(scores))
    best_score = scores[best_idx]
    best_bp = sorted(pos_bins[list(bp_comb[best_idx])])

    assert not np.isnan(best_score), 'Error: score is NaN'

    return (best_score,best_bp,meanCurve,normCurve)

def getRampScore4(x, curve, pos, pos_bin_lengh):
    """ Calculate the rampscore based on cocorrelation coefficient

    Take the portion of the curve into account as well

    Note: all position should be in the same unit

    score is from -1 to 1

    Args:
        x (list): list of two element, sub-region within which to calculate the ramp score. In same unit as pos
        curve (np.ndarray): time-binned firing rate of neuron
        pos (np.ndarray): time-binned position
        pos_bin_lengh (int): length of the total region 

    Returns:
        [type]: [description]
    """
    '''
    
    
     '''
    x1=int(x[0])
    x2=int(x[1])

    # Do some error check first
    if np.std(curve) ==0: #error check
        return (0,curve)

    if x2 == x1:
        return (0, curve)

    if x1>x2: #x1 must always proceed x2
        temp = x1
        x1 = x2
        x2 = temp

    # normalize the curve first
  
    curve = (curve-curve.mean())/np.std(curve) #standardization, z-score

    
    #fit with the original firing rate
    pos_range = (pos>=x1) & (pos<=x2)
    x = pos[pos_range].reshape(-1,1)
    y = curve[pos_range].reshape(-1,1)


    if len(x) > 10: 
        coeff = np.corrcoef(x.T,y.T)
        coeff= coeff[0,1]
    else: # too few data for corrcoef to be meaningful
        coeff = 0


    factor = (x2-x1)/pos_bin_lengh
    score = 2*coeff*factor/(abs(coeff)+factor) # use harmonic mean to combine the two metrics


    if np.isnan(score):
        score = 0
    
    return (score, curve)




def findBestRampScore(curve,span_threshold=2):
    # Attempt to find the best ramp score by performing a grid search on the break points
    # return the best estimated score, and the break point used to calculate those score
    # all span shorter than the span_threshold will be assigned a score of 0

    bp = findBreakPt(curve)
    bp_comb = makeLinearGrid([bp,bp])

    #Search for the best region to calculate the ramp score
    scores = np.zeros((len(bp_comb),))
    for i in range(len(bp_comb)):
        if abs(bp_comb[i][0] - bp_comb[i][1]) > span_threshold:
            scores[i] , curve = getRampScore3(bp_comb[i],curve)
        else:
            scores[i] = 0

    best_idx = np.argmax(scores)
    best_score = np.max(scores)

    return (best_score,bp_comb[best_idx],curve)


def getRampScore3(x, curve, pos=None):
    '''
    Fit the linear curve based on the maximum rather than the peak, so that a ramp without a peak can also be accepted
    Use huber loss to reduce effect of outliner
    Take the portion of the curve into account as well

    if pos is provided, then the range of the curve will be referred to the position
    
     '''
    x1=int(x[0])
    x2=int(x[1])

    # Do some error check first
    if np.std(curve) ==0: #error check
        return (-np.inf,curve)

    if x2 == x1:
        return (-np.inf, curve)

    if x1>x2: #x1 must always proceed x2
        temp = x1
        x1 = x2
        x2 = temp

    # normalize the curve first
    curve = (curve-curve.mean())/np.std(curve) #standardization, z-score

    
    if pos is not None:
        #fit with the original firing rate
        pos_range = (pos>=x1) & (pos<=x2)
        x = pos[pos_range].reshape(-1,1)
        y = curve[pos_range]
        reg  = lm.LinearRegression().fit(x,y)
        predicted = reg.predict(x)

    else:
        #fit with the provided curve only
        x = np.arange(x1,x2).reshape(-1,1)
        y = curve[x1:x2]
        reg  = lm.LinearRegression().fit(x,y)
        predicted = reg.predict(np.arange(len(curve)).reshape(-1,1))

    # loss = huberLoss(predicted[x1:x2],curve[x1:x2]) #only use the linear region
    loss = np.mean((predicted[x1:x2]-curve[x1:x2])**2)
    # print(f'loss:{loss}')
    meanLoss = np.mean(loss)
    factor = (x2-x1)/(len(curve)-1)

    score = (1-meanLoss)*factor

    
    return (score, curve)
  

def getRampScore2(curve,returnNormCurve=False):
    '''
    Fit the linear curve based on the maximum rather than the peak, so that a ramp without a peak can also be accepted
    '''

    if np.std(curve) ==0: #error check
        if returnNormCurve:
            return (-np.inf,curve)
        else:
            return -np.inf

    # print(np.std(curve))
    # normalize the curve first
    # curve = (curve-curve.min())/(curve.max()-curve.min())
    curve = (curve-curve.mean())/np.std(curve) #standardization, z-score

    # fit the best fit line between the maximum and minimum region
    x1 = np.argmin(curve)
    x2 = np.argmax(curve)
    if x1>x2: #x1 must always proceed x2
        temp = x1
        x1 = x2
        x2 = temp
    # print(x1,x2)
    
    x = np.arange(x1,x2).reshape(-1,1)
    y = curve[x1:x2]
    reg  = lm.LinearRegression().fit(x,y)
    predicted = reg.predict(np.arange(len(curve)).reshape(-1,1))
    # plt.plot(predicted)
    # plt.plot(curve)
    # print(curve)
    # print(predicted)
    error = np.abs(predicted-curve)
    score = 1-np.mean(error)

    if returnNormCurve:
        return (score, curve)
    else:
        return score
    

def getRampScore(pks,left_base,curve):
    # Fit a linear curve between the base and the peak
    # calculate the MSE between the fitted curve and actual curve
    # normalize it with the height of the peak
    # Subtract the result from 1, so that 1 is the optimal ramp cell
    x1 = int(left_base)
    y1 = curve[x1]
    x2 = int(pks)
    y2 = curve[x2]
    curveWidth = len(curve)

    x = np.array([x1,x2]).reshape(-1,1)
    y = np.array([y1,y2]).reshape(-1,1)
    
    reg = lm.LinearRegression().fit(x,y)
    predicted = reg.predict(np.arange(x1,x2).reshape(-1,1))
    mse = np.mean(abs(predicted - curve[x1:x2].reshape(-1,1)))
    return 1-mse/((y2-y1)*(x2-x1)/2)
    # print(mse)
    # return 1-mse



def calculateRampScore4Dataframe(df,ramp_range,col_suffix=''):
    scores = []
    slopes = []
    breakpts = []
    # Calculate the ramp score 
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        curve = row.meanTrainCurves[ramp_range]
        score,breakpt,_ = findBestRampScore(curve)
        scores.append(score)
        breakpts.append(breakpt)

        # Calculate the slope of the break point
        slope = (curve[breakpt[1]]-curve[breakpt[0]])/(breakpt[1]-breakpt[0])
        slopes.append(slope)

    df['ramp_score'+col_suffix] = scores
    df['ramp_slope'+col_suffix] = slopes
    df['ramp_breakpt'+col_suffix] = breakpts

    return df




def huberLoss(x,y,delta=1):
    # Calculate the Huber loss
    if isinstance(x,np.ndarray):
        error = x-y
        loss = np.zeros_like(error)
        idx1 = np.abs(x-y)<=delta
        loss[idx1] = 0.5*(x[idx1]-y[idx1])**2
        loss[~idx1] = np.abs(x[~idx1]-y[~idx1])
        return loss
    else:
        if np.abs(x-y) <= delta:
            return 0.5*(x-y)**2
        else:
            return np.abs(x-y)-0.5