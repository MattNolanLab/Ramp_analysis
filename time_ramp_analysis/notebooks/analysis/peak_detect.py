import warnings
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpre
from numpy.lib.shape_base import expand_dims
from palettable.colorbrewer.qualitative import Paired_8 as colors
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from utils import getWrappedSubplots

from .ramp_score import *
from tqdm.auto import tqdm


def preproAggDataFrame(dfs_agg,out_range,home_range):
    # pre-processing of aggregated dataframes
    dfs_agg_beaconed = dfs_agg[dfs_agg.trial_type=='beaconed'].copy()
    (dfs_agg_nonbeaconed,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'non-beaconed')
    (dfs_agg_probe,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'probe')
    (dfs_agg_nonbeaconed,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'non-beaconed')

    dfs_agg_all = pd.concat([dfs_agg_beaconed,dfs_agg_nonbeaconed,dfs_agg_probe])
    print(len(dfs_agg_beaconed))

    #add in variance explained
    varExplain_train=np.stack(dfs_agg_all.train_varExplain.values)
    varExplain_train = np.mean(varExplain_train,axis=1)
    dfs_agg_all['meanVar_train'] = varExplain_train

    # add in firing rate range
    curves = np.stack(dfs_agg_all.meanTrainCurves.values)
    fr_range_out = curves[:,out_range].max(axis=1)-curves[:,out_range].min(axis=1)
    fr_range_home = curves[:,home_range].max(axis=1)-curves[:,home_range].min(axis=1)
    fr_range = curves.max(axis=1)-curves.min(axis=1)

    dfs_agg_all['fr_range'] = fr_range
    dfs_agg_all['fr_range_home'] = fr_range_home
    dfs_agg_all['fr_range_out'] = fr_range_out

    dfs_agg_all.loc[:,['trial_type']] = [str(x) for x in dfs_agg_all.trial_type.values] #fix the data type so that it can be plotted easily

    return  dfs_agg_all

def plotPeakHistogram(df,x,ax=None, shift=0,title=None):
    # plot th histogram of peaks
    peak_detected = np.empty((len(df,)))
    for i in range(len(df)):
        peaks = df.iloc[i].peaks
        if len(peaks)>0:
            peak_detected[i] = peaks[0]
        else:
            peak_detected[i] = np.nan
    #convert peak to position
    pos = x[peak_detected.astype(int)]

    #calculate histogram
    # bins = np.linspace(min(x),max(x),nbins)
    bins = np.arange(min(x), max(x),5)
    [hist,bn] = np.histogram(pos,bins)

    #prepare plots
    if ax is None:
        fig,ax = plt.subplots(1,1)
    width = bn[1]-bn[0]
    center = (bn[:-1] + bn[1:]) / 2
    ax.bar(center,hist,align='center',width=width)
    addTrialStructure(ax,1.1*max(hist))
    ax.set_xlabel('Location (cm)')
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)

def plotMaximaHist(maxima,x,title=None,ax=None,xlim=None):
    # plot the histogram of maxima

    #convert peak to position
    pos = x[maxima.dropna().astype(int)]

    #calculate histogram
    # bins = np.linspace(min(x),max(x),nbins)
    bins = np.arange(min(x), max(x),5)
    [hist,bn] = np.histogram(pos,bins)

    #prepare plots
    if ax is None:
        fig,ax = plt.subplots(1,1)
    width = bn[1]-bn[0]
    center = (bn[:-1] + bn[1:]) / 2
    ax.bar(center,hist,align='center',width=width)
    addTrialStructure(ax,max(hist))
    ax.set_xlabel('Location (cm)')
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    
    if xlim is not None:
        ax.set_xlim(xlim)


def plotMaximaHist2(maxima,x,title=None,ax=None,xlim=None):
    # plot the histogram of maxima
    # using new data structure with Sarah's dataframe

    #convert peak to position
    pos = x[maxima.dropna().astype(int)]
    # pos = maxima

    #calculate histogram
    # bins = np.linspace(min(x),max(x),nbins)
    bins = np.arange(min(x), max(x),5)
    [hist,bn] = np.histogram(pos,bins)

    #prepare plots
    if ax is None:
        fig,ax = plt.subplots(1,1)
    width = bn[1]-bn[0]
    center = (bn[:-1] + bn[1:]) / 2
    # ax.bar(center,hist,align='center',width=width)
    addTrialStructure(ax,max(hist))
    sns.distplot(pos, bins=bins, ax=ax,hist_kws={'alpha':0.8})

    ax.set_xlabel('Location (cm)')
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    
    if xlim is not None:
        ax.set_xlim(xlim)

def makePeakDataframe(curves, idx, out_range, home_range):
    # Make the dataframe for peak analysis
    curves_out  = curves[:,out_range]
    curves_home = curves[:,home_range]

    df_out_corrected = doPeakDetection(curves_out)
    df_home_corrected = doPeakDetection(curves_home)
    df_out_corrected.index = idx

    return (df_out_corrected, df_home_corrected)

def doPeakDetection(curves,correctInverted=True):
    # Do invert correct and peak detection

    curves_raw = skpre.normalize(curves,norm='max')
    curves_smooth = signal.savgol_filter(curves_raw, 5,0,axis=1)

    isInverted = getPeakDirection(curves_smooth)

    if correctInverted:
        curves_smooth_corrected = correctCurve4Sign(curves_smooth,isInverted)
        curves_raw_corrected = correctCurve4Sign(curves_raw, isInverted)
    else:
        curves_smooth_corrected = curves_smooth
        curves_raw_corrected = curves_raw


    df_corrected = analyzePeaks(curves_smooth_corrected,curves_raw_corrected, isInverted)

    return df_corrected


def plotCurvesWithPeaks(df,x,ylim=None,plotRaw=True,
    xlim=False,isInverted=None, withTrialStruct=False):
    # plot tuning curves with peaks
    numRow = len(df)//6+1
    fig = plt.figure(figsize=(2.5*6,numRow*2.5))
    for i in range(len(df)):
        rawD = df.curve_raw.iloc[i]
        D = df.curve_smooth.iloc[i]
        peaks = df.peaks.iloc[i]
        peaks_props = df.peaks_props.iloc[i]
        rampScore = df.rampScore.iloc[i]

        if len(peaks)>0:
            ax = fig.add_subplot(numRow,6,i+1)
            if plotRaw:
                ax.plot(x,rawD,color=colors.mpl_colors[0]) 
            
            if withTrialStruct:
                # add shared regions
                addTrialStructure(ax,max(D)*1.1)

            ax.plot(x,D,color=colors.mpl_colors[1])
        
            ax.plot(x[peaks],D[peaks],'ro')

            left_bases = peaks_props['left_bases']
            right_bases = peaks_props['right_bases']
            ax.plot(x[left_bases],D[left_bases],'k^')
            ax.plot(x[right_bases],D[right_bases],'kv')

            title = ''
            widths = ' '.join([f'{w:.2f}' for w in peaks_props['widths']])
            prom = ' '.join([f'{p:.2f}' for p in peaks_props['prominences']])
            score = ' '.join([f'{s:.2f}' for s in rampScore])
            title = f'{i}\n'+f'widths: {widths}\n' + f'prom: {prom}\n' + f'Ramp score:{score}'

            if isInverted is not None:
                title += f'\nInverted: {isInverted[i]}'
            ax.set_title(title)

            ax.set_ylim([min(D),max(D)*1.1])

            ax.set_xlim([min(x), max(x)])
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Normalized firing rate')

            # print(ax.get_xticks())
    plt.tight_layout()

def peakAnalysis2(df, curveRange, col_suffix='', lm_check_col=None, range_thres=0, slope_thres=0):
    # perform peak analysis on a dataframe
    # it assume the df has lm_result
    maxima_locs = []
    maxima_types=[]
    smooth_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        curve = row.meanTrainCurves
        curve = curve[curveRange]
        curve = signal.savgol_filter(curve, 11,0) #smooth the curve first
        smooth_curve.append(curve)
        # print(curve.shape)

        maxima_type = np.nan
     

        #Perform additonal check of the signal range to make sure the maxima is meaningful
        if (curve.max()-curve.min()) < range_thres:
            maxima_loc = np.nan
            maxima_type = np.nan

        if lm_check_col is not None:
            lm_result = row[lm_check_col]

            if lm_result=='Positive':
                # Choose the max point
                maxima_loc = np.argmax(curve)
                maxima_type = 'max'

            elif lm_result =='Negative':
                maxima_loc = np.argmin(curve)
                maxima_type = 'min'
            else:
                maxima_loc = np.nan
                
            if ((row['ramp_slope'+col_suffix]>slope_thres and lm_result=='Negative') or
                (row['ramp_slope'+col_suffix]<-slope_thres and lm_result=='Positive')):
                #Conflicting results
                    maxima_loc = np.nan
                    maxima_type = np.nan

        else:
            # There is no lm check , use the slope of the break point as reference
            if row['ramp_slope'+col_suffix] > 0 :
                maxima_loc = np.argmax(curve)
                maxima_type = 'max-ramp'
            else:
                maxima_loc = np.argmin(curve)
                maxima_type = 'min-ramp'

        maxima_locs.append(maxima_loc)
        maxima_types.append(maxima_type)

    df['maxima'+col_suffix] = maxima_locs
    df['smooth_curve'+col_suffix] = smooth_curve
    df['maxima_type'+col_suffix] = maxima_types

    return df


def peakAnalysis3(df, curveRange, col_suffix='', lm_check_col=None, range_thres=0, slope_thres=0):
    # perform peak analysis on a dataframe
    # it assume the df has lm_result
    # it takes Sarah's average rate directly
    extrema_locs = []
    max_locs_cm = []
    min_locs_cm = []
    extrema_types=[]
    extrema_locs_cm =[]
    smooth_curve = []
    df = df.copy()

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        loc = np.array(df.iloc[i].pos)[curveRange]

        curve = np.array(row.fr_smooth)
        curve = curve[curveRange]
        # Impute NaN values with mean
        curve[np.isnan(curve)] = np.nanmean(curve)

        # print(i)
        if not np.any(np.isnan(curve)):
            curve = signal.savgol_filter(curve, 11,0) #smooth the curve first
   
        smooth_curve.append(curve)
        # print(curve.shape)

        extrema_type = np.nan

        #Perform additonal check of the signal range to make sure the maxima is meaningful
        if (curve.max()-curve.min()) < range_thres:
            extrema_loc = np.nan
            extrema_type = np.nan

        c_max = np.max(curve)
        c_min = np.min(curve)
        c_mean = np.mean(curve)
        max_idx = np.argmax(curve)
        min_idx = np.argmin(curve)
        

        if lm_check_col is not None:
            lm_result = row[lm_check_col]

            if lm_result=='Positive':
                # Choose the max point
                extrema_loc = max_idx
                extrema_type = 'max'

            elif lm_result =='Negative':
                extrema_loc = min_idx
                extrema_type = 'min'
            else:
                extrema_loc = np.nan
        else:
            # Determine the direction of the curve by the difference between the peak and the mean
            
            if(c_max-c_mean)>(c_mean-c_min):
                extrema_loc = max_idx
                extrema_type = 'max'
            else:
                extrema_loc = min_idx
                extrema_type = 'min'

        extrema_locs.append(extrema_loc)
        extrema_types.append(extrema_type)

        if not np.isnan(extrema_loc):
            # print(curveRange)
            # print(maxima_loc)
            extrema_locs_cm.append(loc[extrema_loc])
            max_locs_cm.append(loc[max_idx])
            min_locs_cm.append(loc[min_idx])
        else:
            extrema_locs_cm.append(np.nan)
            max_locs_cm.append(np.nan)
            min_locs_cm.append(np.nan)

        # if(i==1):
        #     raise

    df['extrema'+col_suffix] = extrema_locs
    df['extrema_cm'+col_suffix] = extrema_locs_cm
    df['smooth_curve'+col_suffix] = smooth_curve
    df['extrema_type'+col_suffix] = extrema_types
    df['min_cm'+col_suffix] = min_locs_cm
    df['max_cm'+col_suffix] = max_locs_cm

    return df

def plotCurvesWithRampscore2(df,x_tick, x_range,col_suffix='', ylim=None,plotRaw=True,
        xlim=False,withTrialStruct=False, sort_cols=None, sort_ascending=False,max_cell_plot=None,ncol=5):
        # for use withe new coeff-based ramp score

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=sort_ascending)

    # plot tuning curves with peaks, using the lm_result based dataframe
  

    if max_cell_plot:
        cell_range = range(min(max_cell_plot, len(df)))
    else:
        cell_range = range(len(df))

    fig,axes = getWrappedSubplots(ncol,len(cell_range),(5,5),dpi=100)
    axIdx = 0

    for i in cell_range:
        ax = axes[axIdx]
        row = df.iloc[i]
        curve = row.fr_smooth*10 #need to convert it to Hz

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     fr = np.nanmean(row['train_firing_rate'],axis=0)[x_range]

        if withTrialStruct:
            # add shared regions
            addTrialStructure(ax,max(curve)*1.1)

        # ax.plot(row.fr_smooth, marker='.', linestyle='None',color=colors.mpl_colors[0]) #raw firing rate

        if plotRaw:
            ax.plot(x_tick,curve,color=colors.mpl_colors[1]) 
        
        #point the break pt used for ramp score estimation
        breakpt = row.breakpoint
        # print(breakpt)
        # ax.axvspan(breakpt[0]-30,breakpt[1]-30,color='yellow')

        # slope = row['ramp_slope'+col_suffix]
        ramp_score = row['ramp_score']
        # ramp_fit = row['ramp_fit']
        title = f'{row.trial_type}\n'
        title += f'{i}'
        title += f'- Rs: {ramp_score:.2f}'
        title += f'- Fr: {row.fr_range:.2f}'
        title += f'\n {row.cell_id}'
        # title += f'\n- VAF: {row.meanVar:.2f}'


        ax.set_title(title)

        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([min(x_tick), max(x_tick)])
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Normalized firing rate')

        axIdx += 1

    # fig.subplots_adjust(wspace=1,hspace=1)
    fig.tight_layout()
    return fig

def plotCurvesWithRampscore(df,trial_types, x_tick, x_range,col_suffix='', ylim=None,plotRaw=True,
        xlim=False,withTrialStruct=False, sort_cols=None, sort_ascending=False,max_cell_plot=None):

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=sort_ascending)

    # plot tuning curves with peaks, using the lm_result based dataframe
    numRow = (len(df)//3*len(trial_types))//6+1 #df contain all trial types, so divide by 3
    fig = plt.figure(figsize=(2.5*6,numRow*2.5),dpi=100)
    fig_idx = 1

    if max_cell_plot:
        cell_range = range(min(max_cell_plot, len(df)//3))
    else:
        cell_range = range(len(df)//3)

    for i in cell_range:
        for trial_type in trial_types:
            
            row = df[df.trial_type==trial_type].iloc[i]
            curve = row.meanTrainCurves[x_range]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fr = np.nanmean(row['train_firing_rate'],axis=0)[x_range]

            ax = fig.add_subplot(numRow,6,fig_idx)

            if withTrialStruct:
                # add shared regions
                addTrialStructure(ax,max(curve)*1.1)

            ax.plot(x_tick, fr,marker='.', linestyle='None',color=colors.mpl_colors[0]) #raw firing rate

            if plotRaw:
                ax.plot(x_tick,curve,color=colors.mpl_colors[1]) 
            
            #point the break pt used for ramp score estimation
            breakpt = row['ramp_breakpt'+col_suffix]
            for b in breakpt:
                ax.plot(x_tick[b],curve[b],marker='^',color=colors.mpl_colors[3])

            slope = row['ramp_slope'+col_suffix]
            ramp_score = row['ramp_score'+col_suffix]
            ramp_fit = row['ramp_fit'+col_suffix]
            title = f'{trial_type}\n'
            title += f'{i}'
            title += f'- Rs: {ramp_score:.2f}'
            title += f'- Fr: {row.fr_range:.2f}'
            title += f'\n- VAF: {row.meanVar:.2f}'


            ax.set_title(title)

            if xlim:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([min(x_tick), max(x_tick)])
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Normalized firing rate')

            fig_idx +=1

    fig.tight_layout()
    return fig

def plotCurvesWithPeaks2(df,trial_types, x_tick, x_range,
        ramp_tick, ramp_range, col_suffix='', ylim=None,plotRaw=True,
        xlim=False,withTrialStruct=False):

    # plot tuning curves with peaks, using the lm_result based dataframe
    cell_ids = df.cell_id.unique()

    numRow = (len(cell_ids)*len(trial_types))//6+1 #df contain all trial types, so divide by 3
    fig = plt.figure(figsize=(2.5*6,numRow*2.5),dpi=100)
    fig_idx = 1


    for i,cell_id in enumerate(cell_ids):
    # for i,cell_id in enumerate(cell_ids[:10]):
        for trial_type in trial_types:
            
            row = df[(df.cell_id == cell_id) & (df.trial_type == trial_type) ]
            if len(row) == 1:
                row = row.iloc[0]
                curve = row.meanTrainCurves[x_range]
                curve_smooth = row['smooth_curve']

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    fr = np.nanmean(row['train_firing_rate'],axis=0)[x_range]

                ax = fig.add_subplot(numRow,6,fig_idx)

                if withTrialStruct:
                    # add shared regions
                    addTrialStructure(ax,max(curve)*1.1)

                ax.plot(x_tick, fr,marker='.', linestyle='None',color=colors.mpl_colors[2]) #raw firing rate

                if plotRaw:
                    ax.plot(x_tick,curve,color=colors.mpl_colors[0]) 
                
                ax.plot(x_tick, curve_smooth, color=colors.mpl_colors[1]) # smoothed curve

                peak_loc = row['maxima']
                if not np.isnan(peak_loc):
                    peak_loc = int(peak_loc)
                    ax.plot(x_tick[peak_loc], curve_smooth[peak_loc],'ro') #peak

                #point the break pt used for ramp score estimation
                breakpt = row['ramp_breakpt'+col_suffix]
                ramp_curve = row.meanTrainCurves[ramp_range]
                for b in breakpt:
                    ax.plot(ramp_tick[b],ramp_curve[b],marker='^',color=colors.mpl_colors[3])

                slope = row['ramp_slope'+col_suffix]
                title = f'{i}-{trial_type}-{row.maxima_type}\n'
                title += f' Fr: {row.fr_range:.1f}'
                title += f'-m:({slope:.1f})'
                title += f'-vaf:{row.meanVar:.1f}'


                ax.set_title(title)

                # ax.set_ylim([min(curve),max(curve)*1.1])

                ax.set_xlim([min(x_tick), max(x_tick)])
                ax.set_xlabel('Position (cm)')
                ax.set_ylabel('Normalized firing rate')

            fig_idx +=1

    fig.tight_layout()
    return fig

    
def plotCurvesWithPeaks3(df,trial_types, x_tick, x_range,
        ramp_tick, ramp_range, col_suffix='', ylim=None,plotRaw=True,
        xlim=False,withTrialStruct=False,maxplot=None):

    # plot tuning curves with peaks, using the lm_result based dataframe
    cell_ids = df.cell_id.unique()
    plotPerCell = len(trial_types)
    n = len(cell_ids)*plotPerCell
    if maxplot:
        n = min(n, maxplot)
        
    fig,axes = getWrappedSubplots(6,n,figsize=(3,3),dpi=100)

    # numRow = (len(cell_ids)*len(trial_types))//6+1 #df contain all trial types, so divide by 3
    # fig = plt.figure(figsize=(2.5*6,numRow*2.5),dpi=100)
    # fig_idx = 1

    axIds = 0


    for i,cell_id in enumerate(cell_ids[:n//plotPerCell]):
        for trial_type in trial_types:
            
            row = df[(df.cell_id == cell_id) & (df.trial_type == trial_type) ]
            if len(row) == 1:
                row = row.iloc[0]
                average_rate = np.array(row.Average_Firing_Rate)
                curve = average_rate[x_range]
                curve_smooth = row['smooth_curve']


                ax = axes[axIds]

                if withTrialStruct:
                    # add shared regions
                    addTrialStructure(ax,max(curve)*1.1)

                # ax.plot(x_tick, fr,marker='.', linestyle='None',color=colors.mpl_colors[2]) #raw firing rate

                if plotRaw:
                    ax.plot(x_tick,curve,color=colors.mpl_colors[0]) 
                
                ax.plot(x_tick, curve_smooth, color=colors.mpl_colors[1]) # smoothed curve

                peak_loc = row['maxima']
                if not np.isnan(peak_loc):
                    peak_loc = int(peak_loc)
                    ax.plot(x_tick[peak_loc], curve_smooth[peak_loc],'ro') #peak

                #point the break pt used for ramp score estimation
                # breakpt = row['ramp_breakpt'+col_suffix]
                # ramp_curve = row.meanTrainCurves[ramp_range]
                # for b in breakpt:
                #     ax.plot(ramp_tick[b],ramp_curve[b],marker='^',color=colors.mpl_colors[3])

                # slope = row['ramp_slope'+col_suffix]
                title = f'{i}-{trial_type}-{row.maxima_type}\n'
                # title += f' Fr: {row.fr_range:.1f}'
                # title += f'-m:({slope:.1f})'
                # title += f'-vaf:{row.meanVar:.1f}'


                ax.set_title(title)

                # ax.set_ylim([min(curve),max(curve)*1.1])

                ax.set_xlim([min(x_tick), max(x_tick)])
                ax.set_xlabel('Position (cm)')
                ax.set_ylabel('Normalized firing rate')

            axIds +=1

    # fig.subplots_adjust(wspace=2,hspace=2)
    return fig


def getPeakDirection(curves):
    # Determine peak direction by the mean and max/min
    # the large difference is the peak
    meanFr = curves.mean(axis=1)
    maxFr = curves.max(axis=1)
    minFr = curves.min(axis=1)
    return (maxFr-meanFr) < (meanFr-minFr)


def correctCurve4Sign(curves, isInverted):
    # Correct the curve direction given the provided sign
    # Substract the curve from 1 if isInverted==False
    c = curves.copy()
    c[isInverted,:] = 1 - curves[isInverted,:]

    return c

    
def analyzePeaks(curve_smooth,curves_raw, isInverted):
    #analyze the peaks in the input curves
    peaks = []
    peaks_props = []
    widths = []
    prom = []
    smooth_curves = []
    rampScores= []

    for i in range(curve_smooth.shape[0]):
        pks,pks_props = signal.find_peaks(curve_smooth[i,:],distance=5,prominence=0.1,width=5,rel_height=0.7)
        if len(pks) > 0:
            widths.append(pks_props['widths'].mean())
            prom.append(pks_props['prominences'].mean())
        else:
            widths.append(np.nan)
            prom.append(np.nan)
        peaks.append(pks)
        peaks_props.append(pks_props)

        #calculate ramp score
        rampScore =  []
        for j in range(len(pks)):
            s = getRampScore(pks[j], 
                pks_props['left_bases'][j], 
                curve_smooth[i,:])
            rampScore.append(s)
    
        rampScores.append(rampScore)

    df = pd.DataFrame({'widths':widths,
                    'prominences':prom,
                    'curve_smooth':[c for c in curve_smooth], 
                    'peaks_props':peaks_props,
                    'peaks':peaks,
                    'curve_raw': [c for c in curves_raw],
                    'rampScore':rampScores,
                    'isInverted':isInverted})

    df['peakIdx'] = df.widths*df.prominences


    return df


def plotCurvesClusters(df,k=3,isHalfTrack=True):
    peaks = np.concatenate(df.peaks.values)
    curve_smooth = np.stack(df.curve_smooth.values).transpose()
    (center,label,_) = sk.cluster.k_means(peaks.reshape(-1,1),k)
    fig,a = plt.subplots(1,k,figsize=(3*k,3),dpi=100)
    for i in range(len(center)):
        a[i].set_title(f'center: {center[i][0]:.2f}:({np.sum(label==i)})')
        a[i].plot(curve_smooth[:,label==i],color=colors.mpl_colors[i*2])
        a[i].plot(curve_smooth[:,label==i].mean(axis=1),color=colors.mpl_colors[i*2+1],lineWidth=2)







def filter_peaks(df):
    # return df[(df.peaks.apply(len)==1) & (df.widths>10) & (df.prominences > 0.2)]
    # return df[df.peaks.apply(len)==1]
    pkCriteria = df.peaks.apply(len)==1
    rampCriteria = np.array([(r[0] if len(r)==1 else 0) for r in df.rampScore.values])>0.95
    widthCriteria = df.widths> 10
    promCriteria = (df.prominences > 0.2)
    return df[rampCriteria & pkCriteria & widthCriteria]


def makeSankeyDict(df, source_col, target_col,color_col):
    # make a source flow dict for plotting the sankey diagram
    sankey= pd.DataFrame({'source': df[source_col], 'target': df[target_col].apply(str), 'type': df[color_col]})

    # get aggregate counts
    df_agg = sankey.groupby(['source','target','type']).size()
    df_agg = df_agg.reset_index()
    df_agg = df_agg.rename(columns={0:'value'})
    
    return df_agg.to_dict('records')

