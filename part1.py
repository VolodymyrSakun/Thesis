
# COX + AFT + Trees
# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#fitting-the-regression

# metrics
# https://cran.r-project.org/web/packages/SurvMetrics/vignettes/SurvMetrics-vignette.html

# survival regression 



import os
import pandas as pd
import numpy as np
from datetime import timedelta
import lib2
# from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import itertools
# from pandas import ExcelWriter
from lifelines.utils import concordance_index
from sklearn.inspection import permutation_importance
from scipy.interpolate import interp1d
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.datasets import get_x_y
import random        
# from datetime import datetime
# from sksurv.metrics import integrated_brier_score
# from sksurv.metrics import brier_score
# from sksurv.metrics import cumulative_dynamic_auc
# from FrameList import FrameList
# import lifelines
# import libChurn2
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.ensemble import ExtraSurvivalTrees
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines import GeneralizedGammaFitter
from lifelines import WeibullFitter
from lifelines import LogNormalFitter
from lifelines import LogLogisticFitter
from lifelines import ExponentialFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter

from scipy.integrate import simps

from matplotlib import pyplot as plt
# import lib3
# from State import TS_STEP
# from States import loadCDNOW
# import State
# from States import States
from datetime import date
from sklearn.metrics import mean_absolute_error
# from Distributions import Statistics
# from sksurv.datasets import get_x_y
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.ensemble import ExtraSurvivalTrees
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
import libSurvival
from lifelines import CoxTimeVaryingFitter
# from math import exp
# from lifelines import NelsonAalenFitter, PiecewiseExponentialFitter, SplineFitter
# from scipy.integrate import simps
# from numpy import trapz
from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from copy import deepcopy
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP
import lib3
# from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
# from sksurv.datasets import load_whas500
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.linear_model import CoxnetSurvivalAnalysis
# from Encoder import DummyEncoderRobust
# from sklearn.linear_model import LogisticRegression
# import lib4
# from math import log
from sklearn.metrics import median_absolute_error
from random import sample

    

def plotBrierScore(modelsUnivariate, fitResultsDict, suffix):
    
#   Plot Brier scores
    bsList = []
    ibsList = []
    models = []
    for sModel in modelsUnivariate:
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        if fitResults['Metrics'].IBS is None:
            # nothing to plot
            return
        models.append(sModel)
        bsList.append(fitResults['Metrics'].BrierScore)
        ibsList.append(fitResults['Metrics'].IBS) 
        
    if len(modelsUnivariate) <= 4:
        M = 2
        N = 2
    else:
        M = 2
        N = 3
#   Plot brier scores for models

    fig, ax = plt.subplots(M, N, figsize=(19,15))
    i = 0
    j = 0
    for k in range(0, len(models), 1):
        model = models[k]
        bsSeries = bsList[k]
        ibs = ibsList[k]
    # for model, bsSeries in bsDict.items():
        # ibs = ibsDict[model]
        ax[i][j].plot(bsSeries)
        ax[i][j].title.set_text('{}. IBS: {}'.format(model, round(ibs, 6)))
        ax[i][j].set_xlabel('Time lived in weeks')
        ax[i][j].set_ylabel('Brier Score')
        j += 1
        if j > N-1:
            i += 1
            if i > M-1:
                break
            j = 0
    title = 'Brier score {}'.format(suffix)
    fig.suptitle(title, fontsize=16)
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    return

def plot_cumulative_dynamic_auc(times, aucTS_Dict, aucDict):
#   Plot plot_cumulative_dynamic_auc for models
    fig, ax = plt.subplots(2, 2, figsize=(19,15))
    i = 0
    j = 0
    for model, aucTS in aucTS_Dict.items():
        aucMean = aucDict[model]
        ax[i][j].plot(times, aucTS)
        ax[i][j].title.set_text(model)
        ax[i][j].set_xlabel('Time lived in weeks')
        ax[i][j].set_ylabel('Time-dependent AUC')
        ax[i][j].axhline(aucMean, linestyle="--")
        j += 1
        if j > 1:
            i += 1
            j = 0
    title = 'Time-dependent AUC'
    fig.suptitle(title, fontsize=16)
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    return

def fixFloat(d):
    """
    in dictionary change numpy float to float

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    dNew : TYPE
        DESCRIPTION.

    """
    dNew = {}
    for key, value in d.items():
        if isinstance(value, np.float64) or isinstance(value, np.float32):
            newValue = float(value)
        else:
            newValue = value                
        dNew[key] = newValue                
    return dNew
    

def plotSurvivalCurvesUnivariate(modelsUnivariate, fitResultsDict, km):
#   Plot Survival functions
    fig, ax = plt.subplots(1, 1, figsize=(19, 10))
    ax = km.plot_survival_function()
    # S_KM_Series.plot(ax=ax)
    for sModel in modelsUnivariate:
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        # ax = fitResults['fitterObserved'].plot_survival_function(ax=ax)
        fitResults['fitter'].survival_function_.plot(ax=ax)        
    
    ax.axhline(y=0.5, color='k', linestyle='-', label='S(t)=0.5')
    ax.legend()
    ax.set_xlabel('Survival time, weeks')
    ax.set_ylabel('Probability of being alive')
    # ax.title.set_text('Survival function')
    title = 'Survival functions obtained by univariate estimators'
    fig.suptitle(title, fontsize=14)    
    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()         
    return 
    
def boxPlotError(models, fitResultsDict, suffix):
    errorAbsExpectedList = []
    errorAbsMedianList = []
    maxExpected = -np.inf
    minExpected = np.inf
    sizeExpectedList = []
    maxMedian = -np.inf
    minMedian = np.inf
    sizeMedianList = []
    for sModel in models:            
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        testCensoredDict = fitResults.get('testCensored')
        if testCensoredDict is None:
            continue
        errorAbsExpected = [i for i in testCensoredDict['errorAbsExpected'] if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
        if len(errorAbsExpected) > 0:
            maxExpected = max(max(errorAbsExpected), maxExpected)
            minExpected = min(min(errorAbsExpected), minExpected)
        sizeExpectedList.append(len(errorAbsExpected))
        errorAbsExpectedList.append(errorAbsExpected)
        errorAbsMedian = [i for i in testCensoredDict['errorAbsMedian'] if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
        if len(errorAbsMedian) > 0:
            maxMedian = max(max(errorAbsMedian), maxMedian)        
            minMedian = min(min(errorAbsMedian), minMedian)
        sizeMedianList.append(len(errorAbsMedian))
        errorAbsMedianList.append(errorAbsMedian)

    if len(errorAbsMedianList) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,15))

    _ = ax1.boxplot(errorAbsExpectedList, vert = 1)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel('Absolute Error Expected')
    ax1.title.set_text('Box plot Absolute Error Expected')
    for i in range(0, len(sizeExpectedList), 1):
        ax1.text(i+1, -int((maxExpected - minExpected) / 30), '{}'.format(sizeExpectedList[i]), ha='center', fontsize=10)        
    
    _ = ax2.boxplot(errorAbsMedianList, vert = 1)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylabel('Absolute Error Median')
    ax2.title.set_text('Box plot Absolute Error Median')
    for i in range(0, len(sizeMedianList), 1):
        ax2.text(i+1, -int((maxMedian - minMedian) / 30), '{}'.format(sizeMedianList[i]), ha='center', fontsize=10)

    title = 'Box plot Absolute Error {}'.format(suffix)
    # fig.suptitle(title, fontsize=14)  
            
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return

def boxPlotSurvivalTime(models, fitResultsDict, suffix):
    tExpectedList = []
    tMedianList = []    
    maxExpected = -np.inf
    minExpected = np.inf
    sizeExpected = []
    maxMedian = -np.inf
    minMedian = np.inf
    sizeMedian = []
    for sModel in models:            
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        tExpected = fitResults.get('tExpected')        
        if tExpected is not None:
            tExpected = [i for i in tExpected if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
            maxExpected = max(max(tExpected), maxExpected)
            minExpected = min(min(tExpected), minExpected)            
            sizeExpected.append(len(tExpected))            
            tExpectedList.append(tExpected)
        tMedian = fitResults.get('tMedian')
        if tMedian is not None:
            tMedian = [i for i in tMedian if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
            maxMedian = max(max(tMedian), maxMedian)
            minMedian = min(min(tMedian), minMedian)            
            sizeMedian.append(len(tMedian))             
            tMedianList.append(tMedian)
            
    if len(tMedianList) == 0:
        return
        
    if len(tExpectedList) == 0:        
        fig, ax = plt.subplots(1, 1, figsize=(19,15))                
        _ = ax.boxplot(tMedianList, vert = 1)
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylabel('Median survival time')
        ax.title.set_text('Box plot median survival time')  
        for i in range(0, len(tMedianList), 1):
            ax.text(i+1, -int((maxMedian - minMedian) / 30), '{}'.format(sizeMedian[i]), ha='center', fontsize=10)

        title = 'Box plot survival time {}'.format(suffix)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,15))
    
        _ = ax1.boxplot(tExpectedList, vert = 1)
        ax1.set_xticklabels(models, rotation=45)
        ax1.set_ylabel('Expected survival time')
        ax1.title.set_text('Box plot expected survival time')
        for i in range(0, len(tExpectedList), 1):
            ax1.text(i+1, -int((maxExpected - minExpected) / 30), '{}'.format(sizeExpected[i]), ha='center', fontsize=10)

        _ = ax2.boxplot(tMedianList, vert = 1)
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylabel('Median survival time')
        ax2.title.set_text('Box plot median survival time')
        for i in range(0, len(tMedianList), 1):
            ax2.text(i+1, -int((maxMedian - minMedian) / 30), '{}'.format(sizeMedian[i]), ha='center', fontsize=10)
        
        title = 'Box plot survival time {}'.format(suffix)
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return

def boxPlot_IAE_ISE(models, fitResultsDict, suffix):
    iaeList = []
    iseList = []
    for sModel in models:            
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        iaeList.append(fitResults['Metrics'].IAE)        
        iseList.append(fitResults['Metrics'].ISE)
        
    if len(iaeList) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,15))

    _ = ax1.boxplot(iaeList, vert = 1)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel('Integrated Absolute Error')
    ax1.title.set_text('Box plot Integrated Absolute Error')
       
    _ = ax2.boxplot(iseList, vert = 1)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylabel('Integrated Square Error')
    ax2.title.set_text('Box plot Integrated Square Error')
    
    title = 'Box plot IAE ISE {}'.format(suffix)
    # fig.suptitle(title, fontsize=14)  
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return
    

def plotSurvivalSample(sModel, sampleId, S_Large):
    
    # get 
    tMax = 0
    for iD in sampleId:    
        S_LargeSeries = S_Large[iD]
        tMean = round(simps(np.array(S_LargeSeries.values).reshape(-1), dx=1))
        tMax = max(tMax, tMean)
    tMax += 10

    S_Medium = S_Large.iloc[0 : tMax].copy(deep=True)
#   Plot Survival functions
    fig, ax = plt.subplots(1, 1, figsize=(19, 10))
    for i, iD in enumerate(sampleId):
        S_MediumSeries = S_Medium[iD]
        S_LargeSeries = S_Large[iD]
    
        tMedian = libSurvival.getSurvivalTime(S_LargeSeries, tSurvived=None, method='median')
        tMean = libSurvival.getSurvivalTime(S_LargeSeries, tSurvived=None, method='expected')

        ax.plot(S_MediumSeries, color=colors[i], linestyle='-', label=iD)
        if tMean < np.inf and not lib2.isNaN(tMean):
            ax.axvline(tMean, color=colors[i], linestyle=':', label='Mean lifetime {}'.format(tMean)) 
        if tMedian < np.inf and not lib2.isNaN(tMedian):            
            ax.axvline(tMedian, color=colors[i], linestyle='-.', label='Median lifetime {}'.format(tMedian))
    
    ax.axhline(y=0.5, color='k', linestyle='-', label='S(t)=0.5')
    ax.legend()
    ax.set_xlabel('Survival time, weeks')
    ax.set_ylabel('Probability of being alive')
    # ax.title.set_text('Survival function')
    title = 'Survival functions of {} random customers obtained by {}'.format(len(sampleId), sModel)
    fig.suptitle(title, fontsize=14)    
    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()         
    return



###############################################################################    
if __name__ == "__main__":

    sData = 'Simulated1' # CDNOW casa_2 plex_45 Retail Simulated1 SimulatedShort001

    if sData.find('SimulatedShort') != -1: # SimulatedShort001
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2000, 7, 1), 'dEnd': date(2004, 7, 1)}
    elif sData == 'casa_2':
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2016, 1, 1), 'dEnd': date(2019, 1, 1)}
    elif sData == 'CDNOW':
        args = {'dStart': None, 'holdOut': 90,
            'dZeroState': None, 'dEnd': None}
    elif sData == 'Retail':
        args = {'dStart': None, 'holdOut': 90,
            'dZeroState': None, 'dEnd': None}
    elif sData == 'Simulated1':
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2000, 7, 1), 'dEnd': date(2003, 7, 1)}     
        
    bGRU_only = False
    bUseGRU_predictions = False
    bUseLongTrends = True
    bUseShortTrends = True
    bUseRecency = True
    bUseLoyalty = True
    
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'o', 'k']

    columnsScores = ['AIC', 'BIC', 'CI', 'CI IPCW', 'IBS', 'AUC', 'IAE Median', 
                     'ISE Median', 'tExpected', 'sizeExpected', 'tMedian', 'sizeMedian']
    
    columnsScoresCensored = ['MAE Expected', 'sizeExpected', 'MedianAE', 'sizeMedian']


    # random_state = 101
    random.seed(101)
    # lag = 0
    # pMax = 0.9999 # cut outliers in predicted time
    #!!! dev
    # fractionTest = 0.8 # dev
    fractionTest = 0.2 # work

    mergeTransactionsPeriod = 1
    
    dEndTrain = args.get('dEnd') # None for real data
    dZeroState = args.get('dZeroState') # None for real data
    holdOut = args.get('holdOut') # None for real data

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    # dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)
    
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)
    lib2.checkDir(resultsSubDir)
        
    inputFilePath = os.path.join(dataSubDir, '{}.txt'.format(sData))
    transactionsDf = loadCDNOW(inputFilePath, 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    
    if dEndTrain is not None:
        transactionsCutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=None, dEnd=dEndTrain,\
            bIncludeStart=True, bIncludeEnd=False, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        transactionsHoldOutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=dEndTrain, dEnd=None,\
            bIncludeStart=False, bIncludeEnd=True, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        clientsAliveHoldOut = sorted(list(transactionsHoldOutDf['iId'].unique()))
    else:
        clientsAliveHoldOut = None
        
    descriptors = ['recency', 'frequency']
    
    # all features
    # features = ['pPoisson', 
    #     'pChurn','trend_pChurn', 'trend_short_pChurn',
    #     'pSurvival', 'trend_pSurvival', 'trend_short_pSurvival',
    #     'C_Orig', 'trend_C_Orig', 'trend_short_C_Orig', 
    #     'moneySum', 'moneyMedian', 'moneyDailyStep',
    #     'r10_R', 'r10_F', 'r10_L', 'r10_M',
    #     'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
    #     'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
    #     'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
    #     'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
    #     'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']

#   no recency, loyalty
    # features = [
    #     'C_Orig', 'trend_C_Orig', 'trend_short_C_Orig', 
    #     'moneySum', 'moneyMedian', 'moneyDailyStep',
    #     'r10_F', 'r10_M',
    #     'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC',
    #     'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC',
    #     'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC']

    features = ['C', 'trend_C', 'trend_short_C', # value and trend
        # 'trend_ratePoisson', 'trend_short_ratePoisson', # only trends
        # 'trend_pPoisson', 'trend_short_pPoisson', # only trends       
        # 'r10_R', # never include it
        'moneySum', 
        # 'moneyMedian', 
        'moneyDailyStep',        
        'r10_F', 
        # 'r10_M',
        'r10_RF', 
        'r10_FM', 
        'r10_RFM',
        'trend_r10_RF', 'trend_short_r10_RF', 
        'trend_r10_FM', 'trend_short_r10_FM', 
        'trend_r10_RFM', 'trend_short_r10_RFM',
        'trend_r10_RFML', 'trend_short_r10_RFML']

        
        
          
    
        # features = [
        # 'C_Orig', 'trend_C_Orig', 'trend_short_C_Orig', 
        # 'moneySum', 'moneyMedian', 'moneyDailyStep',
        # 'r10_F', 'r10_M',
        # 'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC',
        # 'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC',
        # 'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC']




#     if not bUseRecency:
#         # no recency
#         features = [x for x in features if x not in ['r10_R', 'r10_RF', 'r10_RFM', 'r10_RFMC', 'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']]

#     if not bUseLoyalty:
#         features = [x for x in features if x not in ['r10_L', 'r10_LFM', 'r10_LFMC', 'trend_r10_LFMC', 'trend_short_r10_LFMC']]
     
#     if not bUseGRU_predictions:
# #   Use probability of churn obtained from ANN as input feature for survival        
#         features = [x for x in features if x not in ['pChurn','trend_pChurn', 'trend_short_pChurn']]
    
#     if not bUseLongTrends:
#         filtered = []
#         for feature in features:
#             if feature.find('trend') != -1 and feature.find('trend_short') == -1:
#                 continue
#             filtered.append(feature)
#         features = filtered

#     if not bUseShortTrends:
#         filtered = []
#         for feature in features:
#             if feature.find('trend_short') != -1:
#                 continue
#             filtered.append(feature)
#         features = filtered

#     if bGRU_only:
#         features = ['pChurn','trend_pChurn', 'trend_short_pChurn']

    featuresKeepOrig = ['pPoisson', 'pChurn', 'C', 'C_Orig', 'pSurvival']
    featuresScale = [x for x in features if x not in featuresKeepOrig]
        
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))
    stateLast = states.loadLastState()
    
    # true durations for simulated data
    if sData in ['CDNOW', 'Retail']:
        mapClientToDurationTS = None      
    elif sData.find('Simulated') != -1:
        mapClientToDuration, mapClientToDurationTS = states.getTrueLifetime(transactionsDf)
        mapClientToChurnDate = states.getChurnDate(transactionsDf)
    else:
        raise RuntimeError()
        
    nStates = len(states.dStateList)
    
    print('Prepare inputs for survival')
    survivalDf = states.makeSurvivalData(descriptors + features)        

    if clientsAliveHoldOut is not None:
    # clients that died in training intercval according to q95 assumption, but truely alive
    # made purchases in hold-out period
        #   for simulation; check assumprion of death modeling
        clientsDead = list(survivalDf[survivalDf['E'] == 1]['user'])
        clientsAssumptionError = set(clientsDead).intersection(set(clientsAliveHoldOut))
        print('{} customers or {} fraction from population of {} clients violate churn assumption'.\
              format(len(clientsAssumptionError), \
                     len(clientsAssumptionError) / transactionsDf['iId'].nunique(), \
                         transactionsDf['iId'].nunique()))

    
    # for censored last state is modified from dLastPurchase to dLastState
    # duration remains the same: from dBirthState to dLastState
    survivalCensoredDf = survivalDf[survivalDf['E'] == 0].copy(deep=True)    
    for i, row in tqdm(survivalCensoredDf.iterrows()):
        if row['dLastState'] == stateLast.dState:
            # already last state, do not modify features
            continue
        for feature in features + descriptors:
            value = stateLast.Clients[row['user']].descriptors[feature]            
            survivalCensoredDf[feature].loc[i] = value        
       
    tmp = survivalDf[survivalDf['E'] == 0]
    idCensored = list(tmp.index)
    del(tmp)
    idTestCensored = sample(idCensored, int(len(idCensored) * fractionTest))    
    idTrainCensored = list(set(idCensored).difference(set(idTestCensored)))
    idDead = list(set(survivalDf.index).difference(set(idCensored)))        
    idTestDead = sample(idDead, int(len(idDead) * fractionTest))
    idTrainDead = list(set(idDead).difference(set(idTestDead)))
    
###############################################################################
    # Scale features
    survivalScaledDf = libSurvival.scaleSurvivalData(survivalDf, featuresScale)

#!!! check to avoid matrix singularity
    c = survivalScaledDf[features].corr()
    # c2 = c[[i for i in c.columns if i.find('trend') != -1]]

    xTrainDf = survivalScaledDf.loc[idTrainCensored + idTrainDead].copy(deep=True)
    xTrainDf.sort_index(inplace=True)
    
    xTestDf = survivalScaledDf.loc[idTestCensored + idTestDead].copy(deep=True)
#!!! dev    
    # xTestDf = xTestDf.sample(1000)    
    xTestDf.sort_index(inplace=True)
    
    # xTestDeadDf = xTestDf[xTestDf['E'] == 1].copy(deep=True)
    # xTestDeadDf.sort_index(inplace=True)
    
    xTestCensoredDf = xTestDf[xTestDf['E'] == 0].copy(deep=True)
    xTestCensoredDf.sort_index(inplace=True)    
    
###############################################################################
    timeLineExtended = list(range(0, nStates*50, 1))
    timeLineObserved = np.arange(0, nStates)
    metricsDict = {}
    modelsDict = {}
    resultsDict = {} # model -> data frame censoredDf
    medianDict = {}
    columnsPredict = []
    metricsDict = {}
###############################################################################

#   Baseline    
    T = xTrainDf['duration']
    E = xTrainDf['E']

#   Kaplan mayer non-parametric estimator
#   Kaplan Mayer estimator; will serve as 'true' survival function to calculate IAE  
#   neet km instance to plot original survival function
    km = KaplanMeierFitter()
    km.fit(T, event_observed=E, label='Kaplan-Meier')
    # Smooth extended to timeLineObserved survival function
    S_KM_Series = libSurvival.fitKM(T, E, timeLine=timeLineObserved)
    
#   Fit univariate models
    fittersU = {'Exponential': ExponentialFitter(), 'Weibull': WeibullFitter(),
        'LogNormal': LogNormalFitter(), 'LogLogistic': LogLogisticFitter(),
        'GeneralizedGamma': GeneralizedGammaFitter(), 'NelsonAalen': NelsonAalenFitter()}
    
    fitResultsDict = {}
    modelsUnivariate = ['Exponential', 'Weibull', 'LogNormal', 'LogLogistic', 'GeneralizedGamma']
    summaryList = []
    scoresList = []
    scoresCensoredList = []
    for sModel in modelsUnivariate:
        print(sModel)
        # sModel = modelsUnivariate[0]
        fitter = fittersU.get(sModel)
        if fitter is None:
            raise ValueError('fiter must be obe of {}'.format([x for x in fittersU.keys()]))    
        
        fitter.fit(T, E, timeline=timeLineObserved, label='{}'.format(sModel))
               
        S_Series = fitter.survival_function_at_times(timeLineObserved)
        H_Series = fitter.cumulative_hazard_at_times(timeLineObserved)
        S_LargeSeries = fitter.survival_function_at_times(timeLineExtended)
        # make index int and replace values below 1e-7 by 0
        S_LargeSeries = libSurvival.interpolateSurvivalHazard(S_LargeSeries, returnType='pandas', timeLine=None, survivalZero=1e-7)
        
        tMedian = fitter.median_survival_time_
        tExpected = simps(np.array(S_LargeSeries.values).reshape(-1), dx=1)    
    
        # Common metrics
        sE = pd.DataFrame(columns=xTestDf.index, index=S_Series.index)    
        for column in sE.columns:
            sE[column] = S_Series
        
        hE = pd.DataFrame(columns=xTestDf.index, index=H_Series.index)
        for column in hE.columns:
            hE[column] = H_Series    
                        
        metrics = libSurvival.Metrics(T, E, xTestDf['duration'], xTestDf['E'], sE, hE, S_KM_Series=S_KM_Series)
        
        summaryDf = fitter.summary
        summaryDf['CoefName'] = summaryDf.index
        summaryDf['Model'] = sModel
        summaryDf.reset_index(drop=True, inplace=True)

        scoresDf = pd.DataFrame(index=[sModel], columns=columnsScores)
        scoresDf['AIC'] = round(fitter.AIC_, 4)
        scoresDf['BIC'] = round(fitter.BIC_, 4)
        scoresDf['CI'] = round(metrics.CI, 4)
        scoresDf['CI IPCW'] = round(metrics.CI_IPCW['CI'], 4)
        scoresDf['IBS'] = round(metrics.IBS, 6)
        scoresDf['AUC'] = round(metrics.aucMean, 4)                
        scoresDf['IAE Median'] = round(np.median(metrics.IAE), 6)        
        scoresDf['ISE Median'] = round(np.median(metrics.ISE), 6)        
        scoresDf['tExpected'] = round(tExpected, 2)
        scoresDf['sizeExpected'] = round(len(xTestDf), 2)
        scoresDf['tMedian'] = round(tMedian, 2)
        scoresDf['sizeMedian'] = round(len(xTestDf), 2)
                                
        fitResults = {'fitter': fitter, 'Metrics': metrics, 'tExpected': tExpected, 'tMedian': tMedian} 
        
        if mapClientToDurationTS is not None:
#   Simulated data                                   
            tRemainingExpectedTestList = libSurvival.getSurvivalTime(S_LargeSeries, 
                tSurvived=xTestCensoredDf['duration'].values, method='expected')
            tRemainingMedianTestList = libSurvival.getSurvivalTime(S_LargeSeries, 
                tSurvived=xTestCensoredDf['duration'].values, method='median')
                 
            xTestCensoredPredictDf = xTestCensoredDf[['user', 'dBirthState', 'dLastState', 'duration']].copy(deep=True)
            xTestCensoredPredictDf['tTrue'] = xTestCensoredPredictDf['user'].map(mapClientToDurationTS)
            xTestCensoredPredictDf['tRemainingTrue'] = xTestCensoredPredictDf['tTrue'] - xTestCensoredPredictDf['duration']
            xTestCensoredPredictDf['tRemainingExpected'] = tRemainingExpectedTestList
            xTestCensoredPredictDf['tRemainingMedian'] = tRemainingMedianTestList
                    
            xTestCensoredPredictDf['errorAbsExpected'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingExpected']).abs()
            xTestCensoredPredictDf['errorAbsMedian'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingMedian']).abs()            
            
            x1 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingExpected'] != np.inf]
            x1 = x1[~x1['tRemainingExpected'].isna()]        
            MAE = round(x1['errorAbsExpected'].mean(), 6)            
            
            x2 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingMedian'] != np.inf]
            x2 = x2[~x2['tRemainingMedian'].isna()]                                
            MedianAE = round(x2['errorAbsMedian'].median(), 6)

            fitResults['testCensored'] = {'errorAbsExpected': list(xTestCensoredPredictDf['errorAbsExpected']),
                                          'errorAbsMedian': list(xTestCensoredPredictDf['errorAbsMedian']),   
                                          'MAE Expected': MAE,
                                          'Expected Size': len(x1),
                                          'MedianAE Median': MedianAE,
                                          'Median Size': len(x2)
                                          }    
                        
            scoresCensoredDf = pd.DataFrame(index=[sModel], columns=columnsScoresCensored)            
            scoresCensoredDf['MAE Expected'] = MAE
            scoresCensoredDf['sizeExpected'] = len(x1)            
            scoresCensoredDf['MedianAE'] = MedianAE
            scoresCensoredDf['sizeMedian'] = len(x2)
            scoresCensoredList.append(scoresCensoredDf)            
            
        summaryList.append(summaryDf)
        scoresList.append(scoresDf)
        
        fitResultsDict[sModel] = fitResults
    
    summaryDf = pd.concat(summaryList, axis=0)
    summaryDf.set_index(['Model', 'CoefName'], inplace=True)
    summaryDf.to_excel(os.path.join(resultsSubDir, 'Coefficients univariate.xlsx'))

    # scoresDf = pd.concat(scoresList, axis=0)    
    # scoresDf.to_excel(os.path.join(resultsSubDir, 'Scores univariate.xlsx'))
    
    # if len(scoresCensoredList) > 0:
    #     scoresCensoredDf = pd.concat(scoresCensoredList, axis=0)    
    #     scoresCensoredDf.to_excel(os.path.join(resultsSubDir, 'Scores Censored univariate.xlsx'))        
    
    plotSurvivalCurvesUnivariate(modelsUnivariate, fitResultsDict, km)    
    plotBrierScore(modelsUnivariate, fitResultsDict, 'Univariate')
                        
###############################################################################
#   predicted survival time of COX is restricted by study interval    
#   AFT
    
    print('AFT + COX')
    smoothing_penalizer = 0
    ancillary = False
    fit_left_censoring = False
    modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter']#, 'GeneralizedGammaRegressionFitter', 'AalenAdditiveFitter']
    # scoresList = []
    # scoresCensoredList = []
    for sModel in tqdm(modelsAFT):
        # sModel = modelsAFT[0]
        # sModel = 'GeneralizedGammaRegressionFitter'
        penalizer = 0.1
        try:
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', timeline=timeLineObserved, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)
        except:
            print('Try higher penalizer')
            penalizer = 1
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', timeline=timeLineObserved, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)

        S = modelAFT.survivalFunction(xTestDf, times=None) # for metrics
        H = modelAFT.cumulativeHazard(xTestDf, times=None) # for metrics

#   S_Large is for estimating expected / median survival times
        if sModel == 'CoxPHFitter':
            S_Large = S
        else:
            S_Large = modelAFT.survivalFunction(xTestDf, times=timeLineExtended)
            
        sampleId = sample(idTestCensored, 5)        
        plotSurvivalSample(sModel, sampleId, S_Large)

        metrics = libSurvival.Metrics(xTrainDf['duration'], 
            xTrainDf['E'], xTestDf['duration'], xTestDf['E'], S, H, S_KM_Series)

        tExpected = libSurvival.getSurvivalTime(S_Large, None, method='expected')
        tExpectedFound = [i for i in tExpected if i not in [np.inf, np.nan]]
        tMedian = libSurvival.getSurvivalTime(S_Large, None, method='median')    
        tMedianFound = [i for i in tMedian if i not in [np.inf, np.nan]]

        scoresDf = pd.DataFrame(index=[sModel], columns=columnsScores)
        scoresDf['AIC'] = round(modelAFT.AIC, 4)
        scoresDf['BIC'] = None
        scoresDf['CI'] = round(metrics.CI, 4)
        scoresDf['CI IPCW'] = round(metrics.CI_IPCW['CI'], 4)
        scoresDf['IBS'] = round(metrics.IBS, 6)
        scoresDf['AUC'] = round(metrics.aucMean, 4)                
        scoresDf['IAE Median'] = round(np.median(metrics.IAE), 6)        
        scoresDf['ISE Median'] = round(np.median(metrics.ISE), 6)        
        scoresDf['tExpected'] = round(np.mean(tExpectedFound), 2)
        scoresDf['sizeExpected'] = round(len(tExpectedFound), 2)
        scoresDf['tMedian'] = round(np.median(tMedianFound), 2)
        scoresDf['sizeMedian'] = round(len(tMedianFound), 2)

        fitResults = {'fitter': modelAFT, 'Metrics': metrics, 'tExpected': tExpected, 'tMedian': tMedian} 
        
        if mapClientToDurationTS is not None:
#   Simulated data            
            if sModel == 'CoxPHFitter':
            # COx cannot extrapolate beyond study interval
                S_LargeTestCensored = modelAFT.survivalFunction(xTestCensoredDf, times=None)
            else:
                S_LargeTestCensored = modelAFT.survivalFunction(xTestCensoredDf, times=timeLineExtended)    
            
            # extract manually from survival curve            
            tRemainingExpectedTestList = libSurvival.getSurvivalTime(S_LargeTestCensored, 
                xTestCensoredDf['duration'].values, method='expected')
            
            tRemainingMedianTestList = libSurvival.getSurvivalTime(S_LargeTestCensored, 
                xTestCensoredDf['duration'].values, method='median')              

            xTestCensoredPredictDf = xTestCensoredDf[['user', 'dBirthState', 'dLastState', 'duration']].copy(deep=True)
            xTestCensoredPredictDf['tTrue'] = xTestCensoredPredictDf['user'].map(mapClientToDurationTS)
            xTestCensoredPredictDf['tRemainingTrue'] = xTestCensoredPredictDf['tTrue'] - xTestCensoredPredictDf['duration']
            
            xTestCensoredPredictDf['tRemainingExpected'] = tRemainingExpectedTestList
            xTestCensoredPredictDf['tRemainingMedian'] = tRemainingMedianTestList

            xTestCensoredPredictDf['errorAbsExpected'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingExpected']).abs()
            xTestCensoredPredictDf['errorAbsMedian'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingMedian']).abs()            
                    
            x1 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingExpected'] != np.inf]
            x1 = x1[~x1['tRemainingExpected'].isna()]   
            if len(x1) > 0:
                MAE = round(x1['errorAbsExpected'].mean(), 6)            
            else:
                MAE = np.nan
                
            x2 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingMedian'] != np.inf]
            x2 = x2[~x2['tRemainingMedian'].isna()]  
            if len(x2) > 0:                              
                MedianAE = round(x2['errorAbsMedian'].median(), 6)
            else:
                MedianAE = np.nan
                
            fitResults['testCensored'] = {'errorAbsExpected': list(xTestCensoredPredictDf['errorAbsExpected']),
                                          'errorAbsMedian': list(xTestCensoredPredictDf['errorAbsMedian']),   
                                          'MAE Expected': MAE,
                                          'Expected Size': len(x1),
                                          'MedianAE Median': MedianAE,
                                          'Median Size': len(x2)
                                          }    
                        
            scoresCensoredDf = pd.DataFrame(index=[sModel], columns=columnsScoresCensored)            
            scoresCensoredDf['MAE Expected'] = MAE
            scoresCensoredDf['sizeExpected'] = len(x1)            
            scoresCensoredDf['MedianAE'] = MedianAE
            scoresCensoredDf['sizeMedian'] = len(x2)
            scoresCensoredList.append(scoresCensoredDf)             
            
        fitResultsDict[sModel] = fitResults
        scoresList.append(scoresDf)

    # box plot
    boxPlot_IAE_ISE(modelsAFT, fitResultsDict, 'AFT COX')
    boxPlotSurvivalTime(modelsAFT, fitResultsDict, 'AFT COX')

#   Brier scores plot
    plotBrierScore(modelsAFT, fitResultsDict, 'AFT_COX')

    # scoresDf = pd.concat(scoresList, axis=0)    
    # scoresDf.to_excel(os.path.join(resultsSubDir, 'Scores AFT COX.xlsx'))

    # if len(scoresCensoredList) > 0:
    #     scoresCensoredDf = pd.concat(scoresCensoredList, axis=0)    
    #     scoresCensoredDf.to_excel(os.path.join(resultsSubDir, 'Scores Censored AFT COX.xlsx'))        

# plots errors remaining life on simulated data; on real data does nothing
    boxPlotError(modelsAFT, fitResultsDict, 'AFT COX')

            
###############################################################################
# Tree
#   survival curve restricted by duration of study interval
#   beyond it, estimation of time is impossible
#   does not have expected value of time, since it is impossible to extrapolate 
#   survival function and integrate to infinity, only median
    print('Trees')
    
    # argsDict = {'RandomSurvivalForest': 
    #                 {'n_estimators': 100,
    #                 'max_depth': 5,
    #                 'n_jobs': 6,
    #                 'verbose': True,
    #                 'max_features': 'sqrt'},
    #             'ExtraSurvivalTrees': 
    #                 {'n_estimators': 1000,
    #                 'max_depth': None,
    #                 'n_jobs': 6,
    #                 'verbose': True,
    #                 'max_features': None},
    #             'GradientBoostingSurvivalAnalysis': 
    #                 {'n_estimators': 1000,
    #                 'max_depth': 5,
    #                 'n_jobs': 6,
    #                 'verbose': True,
    #                 'max_features': 'sqrt',
    #                 'min_samples_split': 2,
    #                 'min_samples_leaf': 1,
    #                 'dropout_rate': 0.2,
    #                 'subsample': 0.7,
    #                 'monitor': (25, 50)},
    #             'ComponentwiseGradientBoostingSurvivalAnalysis': 
    #                 {'n_estimators': 100,
    #                 'max_depth': 5,
    #                 'n_jobs': 6,
    #                 'verbose': True,
    #                 'max_features': 'sqrt'},     
    #             'CoxnetSurvivalAnalysis': {'l1_ratio': 0.2, 'verbose': True},
    #             'CoxPHSurvivalAnalysis': {'alpha': 0.1, 'verbose': True}
    #             }
        
    argsDict = {'RandomSurvivalForest': 
                    {'n_estimators': 100,
                    'max_depth': 5,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': 'sqrt'},
                'ExtraSurvivalTrees': 
                    {'n_estimators': 1000,
                    'max_depth': None,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': None},
                'GradientBoostingSurvivalAnalysis': 
                    {'n_estimators': 300,
                    'max_depth': 5,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': 'sqrt',
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'dropout_rate': 0.0,
                    'subsample': 0.5,
                    'monitor': None},
                'ComponentwiseGradientBoostingSurvivalAnalysis': 
                    {'n_estimators': 200,
                    'max_depth': 5,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': 'sqrt'},     
                'CoxnetSurvivalAnalysis': {'l1_ratio': 0.2, 'verbose': True},
                'CoxPHSurvivalAnalysis': {'alpha': 0.1, 'verbose': True}
                }
        
    modelsTree = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
        'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis',
        'CoxPHSurvivalAnalysis', 'CoxnetSurvivalAnalysis']
    
    # modelsTree = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
    #     'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    
#!!! dropout makes worse   GradientBoostingSurvivalAnalysis          
    # scoresList = []
    # scoresCensoredList = []
    for sModel in modelsTree:
        print(sModel)
        # sModel = modelsTree[1]
        
        argsTree = argsDict.get(sModel)
        modelTree = libSurvival.Tree(sModel, **argsTree)
        modelTree.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', sample_weight=None)

        # print("Fitted base learners:", modelTree.fitter.n_estimators)
        if sModel == 'GradientBoostingSurvivalAnalysis':
            featureImportance = modelTree.featureImportance
            nFeatures = 20
            libSurvival.plotFeatureImportance(featureImportance, nFeatures, os.path.join(resultsSubDir, 'GB_feature_importance.png'))
        
        S = modelTree.survivalFunction(xTestDf[features])
        H = modelTree.cumulativeHazard(xTestDf[features])
        metrics = libSurvival.Metrics(xTrainDf['duration'], 
            xTrainDf['E'], xTestDf['duration'], xTestDf['E'], S, H, S_KM_Series.loc[list(S.index)].copy(deep=True))
                
        tExpected = libSurvival.getSurvivalTime(S, None, method='expected')
        tExpectedFound = [i for i in tExpected if i not in [np.inf, np.nan]]
        tMedian = libSurvival.getSurvivalTime(S, None, method='median')            
        tMedianFound = [i for i in tMedian if i not in [np.inf, np.nan]]
        
        scoresDf = pd.DataFrame(index=[sModel], columns=columnsScores)
        scoresDf['AIC'] = None
        scoresDf['BIC'] = None
        scoresDf['CI'] = round(metrics.CI, 4)
        scoresDf['CI IPCW'] = round(metrics.CI_IPCW['CI'], 4)
        scoresDf['IBS'] = round(metrics.IBS, 6)
        scoresDf['AUC'] = round(metrics.aucMean, 4)                
        scoresDf['IAE Median'] = round(np.median(metrics.IAE), 6)        
        scoresDf['ISE Median'] = round(np.median(metrics.ISE), 6)        
        scoresDf['tExpected'] = round(np.mean(tExpectedFound), 2)
        scoresDf['sizeExpected'] = round(len(tExpectedFound), 2)
        scoresDf['tMedian'] = round(np.median(tMedianFound), 2)
        scoresDf['sizeMedian'] = round(len(tMedianFound), 2)
                
        fitResults = {'fitter': modelTree, 'Metrics': metrics, 'tExpected': tExpected, 'tMedian': tMedian} 
        
        if mapClientToDurationTS is not None:
#   Simulated data            
            
            S = modelTree.survivalFunction(xTestCensoredDf[features])

            # extract manually from survival curve            
            tRemainingExpectedTestList = libSurvival.getSurvivalTime(S,
                xTestCensoredDf['duration'].values, method='expected')
            
            tRemainingMedianTestList = libSurvival.getSurvivalTime(S,
                xTestCensoredDf['duration'].values, method='median')              

            xTestCensoredPredictDf = xTestCensoredDf[['user', 'dBirthState', 'dLastState', 'duration']].copy(deep=True)
            xTestCensoredPredictDf['tTrue'] = xTestCensoredPredictDf['user'].map(mapClientToDurationTS)
            xTestCensoredPredictDf['tRemainingTrue'] = xTestCensoredPredictDf['tTrue'] - xTestCensoredPredictDf['duration']
            
            xTestCensoredPredictDf['tRemainingExpected'] = tRemainingExpectedTestList
            xTestCensoredPredictDf['tRemainingMedian'] = tRemainingMedianTestList

            xTestCensoredPredictDf['errorAbsExpected'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingExpected']).abs()
            xTestCensoredPredictDf['errorAbsMedian'] = xTestCensoredPredictDf['tRemainingTrue'].sub(xTestCensoredPredictDf['tRemainingMedian']).abs()            
                    
            x1 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingExpected'] != np.inf]
            x1 = x1[~x1['tRemainingExpected'].isna()]  
            if len(x1) > 0:
                MAE = round(x1['errorAbsExpected'].mean(), 6)            
            else:
                MAE = np.nan
            
            x2 = xTestCensoredPredictDf[xTestCensoredPredictDf['tRemainingMedian'] != np.inf]
            x2 = x2[~x2['tRemainingMedian'].isna()] 
            if len(x2) > 0:                               
                MedianAE = round(x2['errorAbsMedian'].median(), 6)
            else:
                MedianAE = np.nan
                
            fitResults['testCensored'] = {'errorAbsExpected': list(xTestCensoredPredictDf['errorAbsExpected']),
                                          'errorAbsMedian': list(xTestCensoredPredictDf['errorAbsMedian']),   
                                          'MAE Expected': MAE,
                                          'Expected Size': len(x1),
                                          'MedianAE Median': MedianAE,
                                          'Median Size': len(x2)
                                          }
                                    
            scoresCensoredDf = pd.DataFrame(index=[sModel], columns=columnsScoresCensored)            
            scoresCensoredDf['MAE Expected'] = MAE
            scoresCensoredDf['sizeExpected'] = len(x1)            
            scoresCensoredDf['MedianAE'] = MedianAE
            scoresCensoredDf['sizeMedian'] = len(x2)
            scoresCensoredList.append(scoresCensoredDf)      
            
        scoresList.append(scoresDf)
        fitResultsDict[sModel] = fitResults                

    # box plot
    boxPlot_IAE_ISE(modelsTree, fitResultsDict, 'Trees')
    boxPlotSurvivalTime(modelsTree, fitResultsDict, 'Trees')

#   Brier scores plot
    plotBrierScore(modelsTree, fitResultsDict, 'Trees')

    scoresDf = pd.concat(scoresList, axis=0)    
    scoresDf.to_excel(os.path.join(resultsSubDir, 'Scores.xlsx'))

    if len(scoresCensoredList) > 0:
        scoresCensoredDf = pd.concat(scoresCensoredList, axis=0)    
        scoresCensoredDf.to_excel(os.path.join(resultsSubDir, 'Scores Censored.xlsx'))        

# plots errors remaining life on simulated data; on real data does nothing
    boxPlotError(modelsTree, fitResultsDict, 'Trees')

    lib2.saveObject(os.path.join(resultsSubDir, 'fitResultsDict.dat'), fitResultsDict, protocol=4)
    
###############################################################################
#   Predict future profit for population
#   Simulated data

    if mapClientToDurationTS is not None:
        lastStateDf = states.getLastUserState()
        # dLastState = states.dStateExistList[-1] or 
        dLastState = stateLast.dState
    
        idCensored = sorted(idTestCensored + idTrainCensored)
        idCensored2 = [key.split('|')[0].strip() for key in idCensored]
    
        xCensoredDf = pd.concat([xTrainDf, xTestDf], axis=0)
        xCensoredDf = xCensoredDf.loc[idCensored]
    
        models = ['Exponential', 'Weibull', 'LogNormal', 'LogLogistic', 'GeneralizedGamma', 
                  'WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter',
                  'GradientBoostingSurvivalAnalysis', 'CoxPHSurvivalAnalysis', 'CoxnetSurvivalAnalysis']
    
        moneyDailyStepDict = stateLast.getDescriptor('moneyDailyStep', activity=None, error='raise')
        moneyDailyDict = stateLast.getDescriptor('moneyDaily', activity=None, error='raise')
    
    #   check with simulation
        # only censored clients
        transactionsHoldOutCensoredDf = transactionsHoldOutDf[transactionsHoldOutDf['iId'].isin(idCensored2)].copy(deep=True)
        print('{} remaining censored'.format(transactionsHoldOutCensoredDf['iId'].nunique()))
        trueProfit = transactionsHoldOutCensoredDf['Sale'].sum()
        print('True spending:', trueProfit)
        
        column1 = ['Expected', 'Median']
        column2 = ['Smooth', 'Step']
        resultsList = []
        for sModel in models:
            print(sModel)
            # sModel = models[0]  
            # model = fitResultsDict['CoxnetSurvivalAnalysis']['fitter']
            model = fitResultsDict[sModel]['fitter']
        
            if sModel in ['Exponential', 'Weibull', 'LogNormal', 'LogLogistic', 'GeneralizedGamma']:
                S_CensoredLarge = model.survival_function_at_times(timeLineExtended)
            elif sModel in ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT']:
                S_CensoredLarge = model.survivalFunction(xCensoredDf, times=timeLineExtended)
            else:
                S_CensoredLarge = model.survivalFunction(xCensoredDf[features])

            tRemainingExpectedCensoredList = libSurvival.getSurvivalTime(S_CensoredLarge, 
                xCensoredDf['duration'].values, method='median') 
            tRemainingMedianCensoredList = libSurvival.getSurvivalTime(S_CensoredLarge, 
                xCensoredDf['duration'].values, method='expected')
        
            for i, tRemaining in enumerate([tRemainingExpectedCensoredList, tRemainingMedianCensoredList]):
                xCensoredDf['tRemaining'] = tRemaining
                xCensoredDf['tRemaining'] = xCensoredDf['tRemaining'].replace(np.inf, np.nan)    
                maxRemaining = xCensoredDf['tRemaining'].max()
                xCensoredDf['tRemaining'] = xCensoredDf['tRemaining'].fillna(maxRemaining)
                
                xCensoredDf['dChurn'] = xCensoredDf['dLastState'] + \
                    pd.to_timedelta(xCensoredDf['tRemaining'].mul(TS_STEP).astype(int), unit='D')    
                dLastChurn = xCensoredDf['dChurn'].max()
                mapClientToChurnDate = xCensoredDf[['dChurn']].squeeze().to_dict()
                mapClientToChurnDate = {key.split('|')[0].strip(): value for key, value in mapClientToChurnDate.items()}
                        
                dDate = dLastState
                datesList = []
                while dDate < dLastChurn:
                    dDate += timedelta(days=1)
                    datesList.append(dDate)
                
                for j, moneyDict in enumerate([moneyDailyDict, moneyDailyStepDict]):
                    profitList = []
                    for dDate in tqdm(datesList):
                        dailyProfit = 0
                        for client, dChurn in mapClientToChurnDate.items():
                            if dChurn > dDate:
                                dailyProfit += moneyDict.get(client, 0)
                        profitList.append(dailyProfit)
                                        
                    result1Df = pd.DataFrame(index=[sModel], columns=['Remaining', 'Descriptor', 'Predicted', 'True_Predicted'])
                    result1Df['Remaining'] = column1[i]
                    result1Df['Descriptor'] = column2[j]                
                    result1Df['Predicted'] = sum(profitList)
                    result1Df['True_Predicted'] = trueProfit - result1Df['Predicted']
                    resultsList.append(result1Df)
                    
        resultsDf = pd.concat(resultsList, axis=0)
    
        resultsDf.to_excel(os.path.join(resultsSubDir, 'Profit simulated.xlsx'))

    
    
    
    # plt.plot(datesList, profitList)



    
    # mapClientToDuration
        
        
        
        
        
        
        
        