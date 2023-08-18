

# pycox




import os
import pandas as pd
import numpy as np
# from datetime import timedelta
import lib2
# from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# import itertools
# from pandas import ExcelWriter
# from lifelines.utils import concordance_index
# from sklearn.inspection import permutation_importance
# from scipy.interpolate import interp1d
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
# from sksurv.datasets import get_x_y
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
# from lifelines import GeneralizedGammaFitter
# from lifelines import WeibullFitter
# from lifelines import LogNormalFitter
# from lifelines import LogLogisticFitter
# from lifelines import ExponentialFitter
from lifelines import KaplanMeierFitter
# from lifelines import NelsonAalenFitter

from scipy.integrate import simps

from matplotlib import pyplot as plt
# import lib3
# from State import TS_STEP
# from States import loadCDNOW
# import State
# from States import States
from datetime import date
# from sklearn.metrics import mean_absolute_error

import libSurvival
from lifelines import CoxTimeVaryingFitter
from FrameArray import FrameArray

from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from copy import deepcopy
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP
import lib3

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw

from sklearn.metrics import median_absolute_error
from random import sample

    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PMF
from pycox.models import LogisticHazard
from pycox.models import DeepHitSingle
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models import CoxCC



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

#   Baseline    
    T = xTrainDf['duration']
    E = xTrainDf['E']
    timeLineObserved = np.arange(0, nStates)

#   Kaplan mayer non-parametric estimator
#   Kaplan Mayer estimator; will serve as 'true' survival function to calculate IAE  
#   neet km instance to plot original survival function
    km = KaplanMeierFitter()
    km.fit(T, event_observed=E, label='Kaplan-Meier')
    # Smooth extended to timeLineObserved survival function
    S_KM_Series = libSurvival.fitKM(T, E, timeLine=timeLineObserved)
    
###############################################################################

    get_target = lambda df: (df['duration'].values, df['E'].values.astype('float32'))

    xTrainDf[features]
    
    xValidDf = xTrainDf.sample(frac=0.2)
    xTrainDf = xTrainDf[~xTrainDf.index.isin(xValidDf.index)]
    
    x_train = xTrainDf[features].values.astype('float32')
    x_val = xValidDf[features].values.astype('float32')
    x_test = xTestDf[features].values.astype('float32')
    x_censored = xTestCensoredDf[features].values.astype('float32')
    
    # labtrans = CoxTime.label_transform()


    y_train = get_target(xTrainDf)
    y_val = get_target(xValidDf)
    
    # CoxTime
    # y_train = labtrans.fit_transform(*get_target(xTrainDf))
    # y_val = labtrans.transform(*get_target(xValidDf))

    # durations_test, events_test = get_target(xTestDf)
    val = x_val, y_val
    
    # CoxTime, CoxCC
    # val = tt.tuplefy(x_val, y_val)

    in_features = x_train.shape[1]
    # num_nodes = [128, 128]
    num_nodes = [256, 256, 256, 256]    
    out_features = 1
    batch_norm = True
    dropout = 0.4
    output_bias = False
    
    
    # # CoxCC
    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
    #                               dropout, output_bias=output_bias)  
    # model = CoxCC(net, tt.optim.Adam)
    
    # CoxTime    
    # net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)    
    # model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    
    
    
    
    
    # CoxPH
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                    dropout, output_bias=output_bias)        
    model = CoxPH(net, tt.optim.Adam)
    
    
    
    
    
    
#   discrete
    
    # # PMF
    # out_features = len(timeLineObserved)
    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

    # # model = PMF(net, tt.optim.Adam, duration_index=timeLineObserved)
    
    # # LogisticHazard
    # model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=timeLineObserved.astype('float32'))
    
    # # DeepHitSingle
    # model = DeepHitSingle(net, tt.optim.Adam(0.01), duration_index=timeLineObserved.astype('float32'))
        
    
    batch_size = 256
    # lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
    # _ = lrfinder.plot()
    
    # lrfinder.get_best_lr()
    
    model.optimizer.set_lr(0.01)
    
    epochs = 1000
    patience = 50
    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    verbose = True
    
    # CoxTime  , CoxCC
    # log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
    #                 val_data=val.repeat(10).cat())

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val, val_batch_size=batch_size)
    
    _ = log.plot()


    
    # CoxPH, CoxTime
    _ = model.compute_baseline_hazards()
    
    S_Deep = model.predict_surv_df(x_test)
    
    # CoxPH
    # H_Deep = model.predict_cumulative_hazards(x_test)
    
    H_Deep = libSurvival.makeHazardFromSurvival(S_Deep, emptyCell=np.nan, returnType='pandas')

    H = libSurvival.interpolateSurvivalHazard(H_Deep, returnType='pandas', timeLine=None, survivalZero=1e-99)
    
    if H.index[0] != 0:
    # insert zeros
        H0 = H.iloc[[0]].copy(deep=True)
        H0.index = [0]
        H0[:] = 0
        H = pd.concat([H0, H], axis=0)
    
    S = libSurvival.interpolateSurvivalHazard(S_Deep, returnType='pandas', timeLine=None, survivalZero=1e-7)
    if S.index[0] != 0:
        # insert ones
        S0 = S.iloc[[0]].copy(deep=True)
        S0.index = [0]
        S0[:] = 1
        S = pd.concat([S0, S], axis=0)
    
    S_KM_Series = S_KM_Series.loc[S.index]    
    
    metrics = libSurvival.Metrics(xTrainDf['duration'], 
        xTrainDf['E'], xTestDf['duration'], xTestDf['E'], S, H, S_KM_Series)
    print(metrics)

    if mapClientToDurationTS is not None:

        S_Deep = model.predict_surv_df(x_censored)
        S = libSurvival.interpolateSurvivalHazard(S_Deep, returnType='pandas', timeLine=None, survivalZero=1e-7)
        if S.index[0] != 0:            
            # insert ones
            S0 = S.iloc[[0]].copy(deep=True)
            S0.index = [0]
            S0[:] = 1
            S = pd.concat([S0, S], axis=0)          
        
        tRemainingExpectedTestList = libSurvival.getSurvivalTime(S, 
            xTestCensoredDf['duration'].values, method='expected', doWhatYouCan=False)
        
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
            
        print(MedianAE)


















