#reload_ext autoreload
#autoreload 2
import os
os.environ['MPLCONFIGDIR'] = '../config'
import pickle
import pandas as pd
import sklearn as sk
from sklearn.decomposition import *
from sklearn.cluster import *
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from Python_PostSorting.Sparse_analysos import *
import sklearn.metrics as skmetrics
from scipy.stats import pearsonr, wilcoxon
from scipy import signal
#from palettable.colorbrewer.qualitative import Paired_8 as colors
np.random.seed(0) #for reproducibility
import matplotlib as mpl
from Python_PostSorting.Peak_Detect import *
import pickle
import pandas as pd
import seaborn as sns
sns.set_palette("Accent")


def calculate_peak_analysis(df_merged):

    out_range = np.arange(30,110) # from 30 to  100 cm, slightly beyond the reward zone x*2-30
    home_range = np.hstack([np.arange(110,199),np.arange(0,29)]) # 110 to 200, 0 to 30
    all_range = np.arange(200)



    dfs_agg_out_sel = peakAnalysis3(df_merged,out_range, lm_check_col='lm_result_outbound')
    dfs_agg_home_sel = peakAnalysis3(df_merged,home_range, lm_check_col='lm_result_homebound')
    dfs_agg_all_sel = peakAnalysis3(df_merged,all_range)

    #dfs_agg_out_sel


    # combine dataframe together
    dfs_agg_out_sel['peak_region'] = 'outbound'
    dfs_agg_home_sel['peak_region'] = 'homebound'
    dfs_agg_all_sel['peak_region'] = 'all'

    dfs_comb = pd.concat([dfs_agg_out_sel, dfs_agg_home_sel,dfs_agg_all_sel])

    # plt.figure(figsize=(7,8),dpi=300)
    # plt.rcParams.update({'font.size': 12})

    dfs2plot = dfs_comb[dfs_comb.peak_region != 'all']
    g = sns.FacetGrid(dfs2plot[dfs2plot.lm_result_outbound=='Positive'],
                      col='trial_type',hue='peak_region',height=5,aspect=1,legend_out=True)
    g.map(sns.distplot,'max_cm', hist=True,rug=True).add_legend().set_titles("Positive peak - {col_name}")

    g = sns.FacetGrid(dfs2plot[dfs2plot.lm_result_outbound=='Positive'],
                      col='trial_type',hue='peak_region',height=5,aspect=1,legend_out=True)
    g.map(sns.distplot,'min_cm', hist=True,rug=True).add_legend().set_titles("Positive trough - {col_name}")

    g = sns.FacetGrid(dfs2plot[dfs2plot.lm_result_outbound=='Negative'],
                      col='trial_type',hue='peak_region',height=5,aspect=1,legend_out=True)
    g.map(sns.distplot,'min_cm', hist=True,rug=True).add_legend().set_titles("Negative trough - {col_name}")

    g = sns.FacetGrid(dfs2plot[dfs2plot.lm_result_outbound=='Negative'],
                      col='trial_type',hue='peak_region',height=5,aspect=1,legend_out=True)
    g.map(sns.distplot,'max_cm', hist=True,rug=True).add_legend().set_titles("Negative peak - {col_name}")


    #save combined data
    dfs_comb.to_pickle('E:/in_vivo_vr/sarah_glm_new/peak_analysis.pkl',protocol=3)



    dfs = [dfs_agg_out_sel, dfs_agg_home_sel,dfs_agg_all_sel]
    dfs_type = ['Outbound','Homebound','all']
    dfs_range=[out_range,home_range,all_range]

    for df,df_type,df_range in zip(dfs,dfs_type,dfs_range):

        fig,ax = plt.subplots(1,3,figsize=(5*3,4),dpi=100)
        fig.suptitle(f'{df_type}')

        peak_num_beaconed = df[df.trial_type=='beaconed'].maxima.dropna().count()
        peak_num_nb = df[df.trial_type=='non-beaconed'].maxima.dropna().count()
        peak_num_p = df[df.trial_type=='probe'].maxima.dropna().count()

        plotMaximaHist2(df[df.trial_type=='beaconed'].maxima, df_range-30,f'Beaconed ({peak_num_beaconed})', ax=ax[0], xlim=[-30,170]);
        plotMaximaHist2(df[df.trial_type=='non-beaconed'].maxima, df_range-30,f'Non-beaconed ({peak_num_nb})',ax=ax[1], xlim=[-30,170]);
        plotMaximaHist2(df[df.trial_type=='probe'].maxima, df_range-30,f'Probe ({peak_num_p})',ax=ax[2], xlim=[-30,170]);

    #     fig.savefig('figures/peak_hist_out.pdf')



    dfs = [dfs_agg_out_sel, dfs_agg_home_sel]
    dfs_type = ['Outbound','Homebound']
    dfs_range=[out_range,home_range]
    xlims = [[0,80], [80,170]]

    for maxima_type in ['max','min']:
        for df,df_type,df_range,xlim in zip(dfs,dfs_type,dfs_range,xlims):

            df = df[df.maxima_type==maxima_type] #get a subset of the maxima type
            fig,ax = plt.subplots(1,3,figsize=(5*3,4),dpi=100)
            max_type_str = 'Pos' if maxima_type=='max' else 'Neg'
            fig.suptitle(f'{df_type} - {max_type_str}')

            #Calculate sample number in each trial type
            peak_num_beaconed = df[df.trial_type=='beaconed'].maxima.dropna().count()
            peak_num_nb = df[df.trial_type=='non-beaconed'].maxima.dropna().count()
            peak_num_p = df[df.trial_type=='probe'].maxima.dropna().count()

            plotMaximaHist2(df[df.trial_type=='beaconed'].maxima, df_range-30,f'Beaconed ({peak_num_beaconed})', ax=ax[0], xlim=xlim);
            plotMaximaHist2(df[df.trial_type=='non-beaconed'].maxima, df_range-30,f'Non-beaconed ({peak_num_nb})',ax=ax[1], xlim=xlim);
            plotMaximaHist2(df[df.trial_type=='probe'].maxima, df_range-30,f'Probe ({peak_num_p})',ax=ax[2], xlim=xlim);

    #         fig.savefig(f'figures/peak_hist_{df_type}_{max_type_str}.pdf')


    ramp_out_range = np.arange(15,45) # from 0 to  60 cm, slightly beyond the reward zone x*2-30
    ramp_out_tick = ramp_out_range*2-30
    ramp_home_range = np.arange(55,85) # 80 to 140
    ramp_home_tick = ramp_home_range*2-30



def getCellId(row):
    return f'{row.session_id}:{row.cluster_id}'

    dfs_agg_out_sel['cell_id'] = dfs_agg_out_sel.apply(getCellId,axis=1)


    fig = plotCurvesWithPeaks3(dfs_agg_out_sel,['beaconed','non-beaconed','probe'],
                               out_range*2-30, out_range,
                               ramp_out_tick, ramp_out_range,
                               col_suffix='_out',withTrialStruct=True,maxplot=30)

    fig.subplots_adjust(wspace=0.5,hspace=0.8)
    # fig.savefig('figures/lm_peak_out.pdf')
