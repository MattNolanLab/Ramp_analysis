
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Paired_8 as colors
from scipy import signal
from sklearn.decomposition.dict_learning import _update_dict
from sklearn.decomposition.dict_learning import *
import sklearn.metrics as skmetrics
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import sklearn as sk
import sklearn.linear_model as lm
from scipy.stats import pearsonr, wilcoxon
import matplotlib.patches as patches

def addTrialStructure(ax,h):
    rect = ax.axvspan(-30,0,alpha=0.2,zorder=0,color='#8C8C8C')
    rect = ax.axvspan(140,200,alpha=0.2,zorder=0,color='#8C8C8C')
    rect = ax.axvspan(60,80,alpha=0.2,zorder=0,color='#E4F4E5')

def addTrialStructureBox(ax,minH, maxH):
    blackbox1 = patches.Rectangle((-30,minH),30,maxH-minH,facecolor='#8C8C8C')
    blackbox2 = patches.Rectangle((140,minH),30,maxH-minH,facecolor='#8C8C8C')
    trial_box = patches.Rectangle((60,minH),20,maxH-minH,facecolor='#E4F4E5')
    ax.add_patch(trial_box)
    ax.add_patch(blackbox1)
    ax.add_patch(blackbox2)

def plotDictCurves(curves,D,xticks,animal_id,normCode,reconCurve,modelType,modelOrder,error):
    numRow = curves.shape[0]//5+1
    fig = plt.figure(figsize=(15,numRow*2.5))

    ax = fig.add_subplot(numRow,5,1) #plot the dictionary first
    ax.plot(xticks,D,color='red')
#     ax.set_ylim([0,0.5])

    for i in range(curves.shape[0]):
        ax = fig.add_subplot(numRow,5,i+2)
        ax.plot(xticks,reconCurve[i,:],color='lightgray')
        ax.plot(xticks,curves[i,:])
        h = np.max(curves[i,:])
        addTrialStructure(ax,h)
        ax.set_title(f'{animal_id[i]} \n Dominant atom:{normCode[i]:.2f} \n {modelType[i]}-{modelOrder[i]} \n {error[i]:.2f}')
        if modelType[i] == 'Position':
            for spine in ax.spines.values():
                spine.set_linewidth(2)
    return fig


def plotMonoMeasure(D):
    fig = plt.figure(figsize=(2.5*5,2*2.5))
    for i in range(D.shape[0]):
        ax = fig.add_subplot(2,5,i+1)
        D_est,error = monoFit(D[i,:],returnErr=True)
        ax.plot(np.arange(D.shape[1]),D[i,:],label='Atom')
        ax.set_title(f'Monotonicity: {error:.2f}')
        # ax.set_ylim([0,0.5])
        ax.plot(D_est,'r--',label='Isotonic estimation')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right')
    fig.tight_layout()
    return fig


def shortenCellId(cell_id):
    s = cell_id.split(':')
    s[1] = '_'.join(s[1].split('_')[:2])
    ss = ':'.join(s)
    return ss

def plotDictCurvesCompare(probCurves,beaconCurves,
    D,xticks,dictBound,cell_id, normCode,modelType,
    metrics,modelOrder=None,highlight='probe',
    err_sorted= True):
    # Compare probe and beaconed curves
    numRow = probCurves.shape[0]//5+1
    fig = plt.figure(figsize=(15,numRow*3))

    if err_sorted:
        # sort the
        sortedIdx = np.argsort(-metrics) #descedning
        probCurves = probCurves[sortedIdx,:]
        beaconCurves = beaconCurves[sortedIdx,:]
        cell_id = cell_id[sortedIdx]
        normCode = normCode[sortedIdx]
        modelType = modelType[sortedIdx]
        metrics = metrics[sortedIdx]

        if modelOrder:
            modelOrder = modelOrder[sortedIdx]


    if D is not None:
        ax = fig.add_subplot(numRow,5,1) #plot the dictionary first
        ax.plot(D,color='red')
        ax.set_ylim([0,0.5])

    for i in range(probCurves.shape[0]):
        #plot curve
        if D is not None:
            ax = fig.add_subplot(numRow,5,i+2)
        else:
            ax = fig.add_subplot(numRow,5,i+1)

        if highlight=='probe':
            ax.plot(xticks,beaconCurves[i,:],color=colors.mpl_colors[2],label='beaconed')
            ax.plot(xticks,probCurves[i,:],color=colors.mpl_colors[1],label='probe')
            ax.plot(xticks[dictBound],probCurves[i,dictBound],linewidth=3,color=colors.mpl_colors[1])
        else:
            ax.plot(xticks,probCurves[i,:],color=colors.mpl_colors[0],label='probe')
            ax.plot(xticks,beaconCurves[i,:],color=colors.mpl_colors[3],label='beaconed')
            ax.plot(xticks[dictBound],beaconCurves[i,dictBound],linewidth=3,color=colors.mpl_colors[3])


        #add trial structures
        h = np.max(np.concatenate([probCurves[i,:],beaconCurves[i,:]]))
        addTrialStructure(ax,h)

        id = shortenCellId(cell_id[i])
        title = f'{id} \n Dominant atom:{normCode[i]:.2f} \n {modelType[i]}'
        if modelOrder:
            title += f'-{modelOrder[i]} \n'
        else:
            title += '\n'

        title += f'vaf: {metrics[i]:.2f}'

        ax.set_title(title)
        if modelType[i] == 'Position':
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Firing rate (Hz) ')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right')
    return fig

def getModelType(df):
    modelOrder =[]
    modelType=[]
    for i in range(len(df)):
        if df.iloc[i].bestUniModel == 0 and df.iloc[i].unimodel_comparison_pvalue <0.05:
            modelType.append('Position')
        elif df.iloc[i].bestUniModel == 1 and df.iloc[i].unimodel_comparison_pvalue <0.05:
            modelType.append('Speed')
        else:
            modelType.append('Both')

        if df.iloc[i].bimodel_comparison_pvalue <0.05:
            modelOrder.append('bivariate')
        else:
            modelOrder.append('univariate')

    return np.array(modelOrder),np.array(modelType)



def getModelTypeSimple(df):
    # A simpler classification into positon, speed, and both
    modelType=[]
    for i in range(len(df)):
        if (df.iloc[i].bestUniModel == 0
            and df.iloc[i].unimodel_comparison_pvalue <=0.05
            and df.iloc[i].bimodel_comparison_pvalue >0.05):
            modelType.append('Position')
        elif (df.iloc[i].bestUniModel == 1
            and df.iloc[i].unimodel_comparison_pvalue <=0.05
            and df.iloc[i].bimodel_comparison_pvalue >0.05):
            modelType.append('Speed')
        elif df.iloc[i].bimodel_comparison_pvalue < 0.05:
            modelType.append('Position * Speed')
        else:
            modelType.append('Unclassified')

    return np.array(modelType)

def plotCurves(D,title=None,ylim=None,dpi=100):
    numRow = D.shape[0]//5+1
    fig = plt.figure(figsize=(2.5*5,numRow*2.5),dpi=dpi)
    for i in range(D.shape[0]):
        ax = fig.add_subplot(numRow,5,i+1)
        ax.plot(D[i,:])
        ax.set_title(i)
        if ylim:
            ax.set_ylim(ylim)
        if title:
            ax.set_title(title[i])

    plt.tight_layout()
    return fig





def plotReconCurveCompare(curve,reconCurve):
    #compare between the original curve and the reconstructed curves
    numRow = curve.shape[0]//5+1
    fig = plt.figure(figsize=(2.5*5,numRow*2.5))
    for i in range(curve.shape[0]):
        ax = fig.add_subplot(numRow,5,i+1)
        ax.plot(curve[i,:])
        ax.plot(reconCurve[i,:])
        # ax.set_ylim([0,0.5])

    plt.tight_layout()

def filterCurves(normCode,vaf):
    # return (normCode>0.8) & (vaf>-10)
    return (normCode>0.6)

def filterCurves2(normCode,selDict,vaf):
    return (np.argmax(normCode,axis=1) == selDict) & (vaf>-10)

def filterCurveMonotone(normCode,monoIndex,vaf,type='pos',thres=0.3):
    #filter curve based on monotonicity
    if type=='pos':
        #only consider atoms that are monotonic
        codeSum = np.sum(normCode[:,monoIndex>=thres],axis=1)
        return (codeSum>0.8) & (vaf>-10)
    else:
        codeSum = np.sum(normCode[:,monoIndex<=-thres],axis=1)
        return (codeSum>0.8) & (vaf>-10)

def plotDict(ax,D,bins):
    ax.plot(bins,D)

def plotDictModelDistribution(normCode,pos_bins,D,
    model_names, modelType,
    dominantVar_names, dominantVar,
    threshold=0.6):

    fig = plt.figure(figsize=(3*3,5*3))
    figIdx = 1
    # maxCode = np.argmax(normCode,axis=1)
    for selDict in range(normCode.shape[1]):
        # Plot dictionary atom
        idx = normCode[:,selDict] > threshold
        ax1 = fig.add_subplot(5,3,figIdx)
        plotDict(ax1,D[selDict],pos_bins)
        plt.ylim([0,0.5])
        ax1.set_ylabel('Normalized firing rate')
        ax1.set_xlabel('Position (cm)')
        figIdx +=1

        # Best model
        ax2 = fig.add_subplot(5,3,figIdx)
        counter = countItems(model_names, modelType[idx])
        plotDistribution(counter,ax2,orientation='h')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Best model')
        figIdx +=1

        ax3 = fig.add_subplot(5,3,figIdx)
        counter = countItems(dominantVar_names, dominantVar[idx])
        plotDistribution(counter,ax3,orientation='h')
        ax3.set_xlabel('Count')
        ax3.set_ylabel('Dominant variable')
        figIdx +=1



    plt.tight_layout()
    return fig

def countItems(item_name, item_list):
    # count number of items given in the item_name
    count = {}
    for n in item_name:
        count[n] = np.sum(item_list==n)
    return count

def plotDistribution(counter,ax=None,figsize=(3,2.5),dpi=100,orientation='v'):
    fig = None

    # plot distribution given a counter
    if ax is None:
        fig = plt.figure(figsize=figsize,dpi=dpi)
        ax = fig.add_subplot(1,1,1)

    if orientation == 'v':
        ax.bar(list(counter.keys()),counter.values(),width=0.5)
    else:
        ax.barh(list(counter.keys()),counter.values(),height=0.5)

    return fig,ax

def balanceTrials(df,trial_types=['beaconed','non-beaconed','probe']):
    # Balance the three different trial types, remove trial that can't find a match
    cell_ids = np.unique(df.cell_id)

    for c in cell_ids:
        toDelete = False
        for t in trial_types:
            if len(df[ (df.trial_type==t) & (df.cell_id==c)]) ==0:
                toDelete = True

        if toDelete:
            df = df.drop(df[df.cell_id==c].index)

    return df

def findMatchingNeuron(refDf,df,trial_type):
    #Find matching neuron based on cell id
    rows = []
    for i in range(len(df)):
        cell_id = df.iloc[i].cell_id
        #find the corresponding record in the big dataframe
        row = refDf[(refDf.cell_id==cell_id) & (refDf.trial_type==trial_type)]
        if len(row) == 1:
            rows.append(row)
        else:
            rows.append(None)
    idx = [r is not None for r in rows]

    if np.any(idx):
        matched = pd.concat(rows)
    else:
        matched = None

    return (matched,df[idx])


def matchAllTrials(dfs_agg):
    #match the record between all trial type, remove record that cannot be matched
    dfs_agg_beaconed = dfs_agg[dfs_agg.trial_type=='beaconed']
    (dfs_agg_nonbeaconed,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'non-beaconed')
    (dfs_agg_probe,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'probe')
    (dfs_agg_nonbeaconed,dfs_agg_beaconed) = findMatchingNeuron(dfs_agg,dfs_agg_beaconed,'non-beaconed')

    dfs_agg_all = pd.concat([dfs_agg_beaconed,dfs_agg_nonbeaconed,dfs_agg_probe])

    return dfs_agg_all

def parseModel(best_model):
    if isinstance(best_model,str):
        m = best_model.replace('pos_grid','P')
        m = m.replace('speed_grid','S')
        m = m.replace('accel_grid','A')
        m = '*'.join(m.split(' '))
        return f'({m})'
    else:
        return 'None'


def getMonotoneIndex(x,smooth=False):
    #distribution of delta should even
    # consider the number of peak
    # consider the range

    if smooth:
        x = signal.savgol_filter(x,5,1)


    #A simple metrics to measure the monotonicity of the curve
    M = np.zeros((len(x)-1))
    M[np.diff(x)>=0]=1
    M[np.diff(x)<0] = -1
    sumDelta = np.sum(M)

    # no. of peak
    M2 = np.diff(x)
    M2 = np.append(M2,M2[-1])

    sign = M2[:-1]*M2[1:]
    signChangeNo = np.sum(sign<0)
    return (sumDelta/len(M)**2+(1-signChangeNo/len(M)))

def expFit(x,returnCoeff=False):

    X = np.arange(len(x))
    Y = np.log(x+1e-10)
    coeff = np.polyfit(X,Y,1)
    y_est =  np.exp(coeff[1])*np.exp(coeff[0]*X)
    if returnCoeff:
        return (y_est,coeff)
    else:
        return y_est

def monoFit(x,returnErr=True):
    # montonicity as defined by the VAF of the isotonic regression
    ir = IsotonicRegression(increasing='auto')
    x_= ir.fit_transform(np.arange(len(x)),x)
    error = skmetrics.explained_variance_score(x,x_)
    if returnErr:
        return (x_,error)
    else:
        return x_

def linearFit(x,returnErr=True):
    t = np.arange(len(x)).reshape(-1,1)
    ir = LinearRegression().fit(t,x)
    x_ = ir.predict(t)
    error = skmetrics.explained_variance_score(x,x_)
    if returnErr:
        return (x_,error)
    else:
        return x_




def monotonicity(x):
    # measure the montonicity using difference of linear fit and isotonic fit

    #get isotonic fit
    x_iso,err_iso = monoFit(x)

    #get linaer fit
    x_linear,err_linear = linearFit(x)

    #Monotonicity score
    M = (1-np.mean((x_iso-x_linear)**2)/np.mean(x))*err_iso

    return M


# monkey patching the dict_learning function
def dict_learning2(X, n_components, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=None, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  return_n_iter=False, positive_dict=False,
                  positive_code=False, method_max_iter=1000):

    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.'
                         % method)

    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs, positive=positive_code)
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             random_state=random_state,
                                             positive=positive_dict)
        dictionary = dictionary.T

        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code)) + 20*np.sum(np.corrcoef(dictionary))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    else:
        return code, dictionary, errors

def filterByCompareWithShuffle(df,df_spikeShuffle):
    #filer curve by comparing with shuffle
    df_beaconed = df[df.trial_type=='beaconed']
    df_beaconed_spikeShuffle = df_spikeShuffle[df_spikeShuffle.trial_type=='beaconed']

    for index,row in df_beaconed.iterrows():
        #find matching cell
        matched=df_beaconed_spikeShuffle[df_beaconed_spikeShuffle.cell_id==row['cell_id']]
        vaf_shuffle = matched.iloc[0].test_log_llh
        vaf_success =  row.test_log_llh
        #do comparison
        result = wilcoxon(vaf_shuffle,vaf_success)
        df_beaconed.loc[index,'compare_spikeShuffle_pvalue'] =result.pvalue

    print(f'Original no. of cell: {len(df_beaconed)}')

    # filtering based on comparison with shuffled data
    df_beaconed = df_beaconed[df_beaconed.compare_spikeShuffle_pvalue<=0.01]
    print(f'After filtering : {len(df_beaconed)}')

    #find matching probe curves
    (df_probe,df_beaconed) = findMatchingNeuron(df,df_beaconed,'probe')
    (df_nonbeaconed,df_beaconed) = findMatchingNeuron(df,df_beaconed,'non-beaconed')


    return (df_beaconed,df_nonbeaconed,df_probe)
