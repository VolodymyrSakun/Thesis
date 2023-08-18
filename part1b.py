
# for 20 simulations
# remove forests, keep only GradientBoost
# bring regresions in one piecee

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
from tqdm import tqdm
import random        
from lifelines import GeneralizedGammaFitter
from lifelines import WeibullFitter
from lifelines import LogNormalFitter
from lifelines import LogLogisticFitter
from lifelines import ExponentialFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from Distributions import Statistics

from scipy.integrate import simps

from matplotlib import pyplot as plt
from datetime import date
import libSurvival
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP
import lib3
from random import sample
import seaborn
    
def plot_partial_effects_on_outcome(featuresPlotDict, featuresScalerDict, fitter, suffix, resultsSubDir):                
    M = 2
    N = 2
    fig, ax = plt.subplots(M, N, figsize=(19,15))
                    
    i = 0
    j = 0
    for feature, valuesList in featuresPlotDict.items():
        values = featuresScalerDict[feature].transform(np.array(featuresPlotDict[feature]).reshape(-1, 1))
        fitter.plot_partial_effects_on_outcome(feature, 
            values=values, plot_baseline=True, y='survival_function', ax=ax[i][j])# cmap='coolwarm'
        current_handles, current_labels = ax[i][j].get_legend_handles_labels()
        
        new_labels = []
        k = 0
        for label in current_labels:
            if label.find(feature) != -1:
                new_label = '{}'.format(valuesList[k])
                new_labels.append(new_label)
                k += 1
            else:
                new_labels.append(label)                                                    
            
        ax[i][j].legend(current_handles, new_labels)
        title = '{}'.format(feature)
        ax[i][j].title.set_text(title)
        ax[i][j].set_xlabel('Time lived in weeks')
        ax[i][j].set_ylabel('Survival probability')    
        j += 1
        if j > N-1:
            i += 1
            if i > M-1:
                break
            j = 0

    suptitle = 'Influence of certain features on survival curve'
    fig.suptitle(suptitle, fontsize=14)    

    fileName = '{}. {}.png'.format(suptitle, suffix)
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    return

def plotBrierScore(modelsUnivariate, fitResultsDict, suffix, resultsSubDir):
    
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
    elif len(modelsUnivariate) <= 6:
        M = 2
        N = 3
    else:
        M = 3
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

def plot_cumulative_dynamic_auc(times, aucTS_Dict, aucDict, resultsSubDir):
#!!! not used    
#!!!   Plot plot_cumulative_dynamic_auc for models
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
    

def plotSurvivalCurvesUnivariate(modelsUnivariate, fitResultsDict, km, resultsSubDir):
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
    
def boxPlotError(models, fitResultsDict, suffix, resultsSubDir):
    """
    Simulated data only

    Parameters
    ----------
    models : TYPE
        DESCRIPTION.
    fitResultsDict : TYPE
        DESCRIPTION.
    suffix : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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

            # stats = Statistics(periods, models=['Lognormal', 'Exponential'])

def boxPlotSurvivalTime(models, fitResultsDict, suffix, resultsSubDir, p=None):
    tExpectedList = []
    tMedianList = []    
    maxExpected = -np.inf
    minExpected = np.inf
    sizeExpected = []
    maxMedian = -np.inf
    minMedian = np.inf
    sizeMedian = []
    for sModel in models: 
        # sModel = models[1]
        fitResults = fitResultsDict.get(sModel)
        if fitResults is None:
            continue
        tExpected = fitResults.get('tExpected')        
        if tExpected is not None:
            tExpected = [i for i in tExpected if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
            
            if p is not None:
                # cut extreme outliers
                stats = Statistics(tExpected, testStats='ks', criterion='pvalue')
                restrict = stats.bestModel.ppf(p)
                tExpected = [i for i in tExpected if i < restrict]    
                
            maxExpected = max(max(tExpected), maxExpected)
            minExpected = min(min(tExpected), minExpected)            
            sizeExpected.append(len(tExpected))            
            tExpectedList.append(tExpected)
        tMedian = fitResults.get('tMedian')
        if tMedian is not None:
            tMedian = [i for i in tMedian if i not in [np.inf, -np.inf] and not lib2.isNaN(i)]
            
            if p is not None:
                # cut extreme outliers
                stats = Statistics(tMedian, testStats='ks', criterion='pvalue')
                restrict = stats.bestModel.ppf(p)
                tMedian = [i for i in tMedian if i < restrict]   
                
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

def boxPlot_IAE_ISE(models, fitResultsDict, suffix, resultsSubDir):
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
    

def plotSurvivalSample(sModel, sampleId, S_Large, resultsSubDir, baseline=None):
    
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

    if baseline is not None:
        ax.plot(baseline, color=colors[-1], linestyle='-', linewidth=3, label='Baseline')
    for i, iD in enumerate(sampleId):
        S_MediumSeries = S_Medium[iD]
        S_LargeSeries = S_Large[iD]
    
        tMean = libSurvival.getSurvivalTime(S_LargeSeries, tSurvived=None, method='expected')
        tMedian = libSurvival.getSurvivalTime(S_LargeSeries, tSurvived=None, method='median')

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

def plotCorrelation(c, resultsSubDir, cmap='BuPu'):
    fig, ax = plt.subplots(1, 1, figsize=(19,15))
    # seaborn.heatmap(c, cmap="plasma", annot=True, ax=ax)
    seaborn.heatmap(c, cmap="BuPu", annot=True, ax=ax)
    
    fig.tight_layout()    
    title = 'Features correlation'
    # fig.suptitle(title, fontsize=14)    
    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)


def main(sData, dataDir, resultsDir):    

    if sData.find('SimulatedShort') != -1: # SimulatedShort001
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2000, 7, 1), 'dEnd': date(2003, 7, 1)}
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

    columnsScores = ['AIC', 'BIC', 'CI', 'CI IPCW', 'IBS', 'AUC', 'IAE Median', 
                     'ISE Median', 'tExpected', 'sizeExpected', 'tMedian', 'sizeMedian']
    
    columnsScoresCensored = ['MAE Expected', 'sizeExpected', 'MedianAE', 'sizeMedian']

    fractionTest = 0.2 # work

    mergeTransactionsPeriod = 1
    
    dEndTrain = args.get('dEnd') # None for real data
    # dZeroState = args.get('dZeroState') # None for real data
    # holdOut = args.get('holdOut') # None for real data

    dataSubDir = os.path.join(dataDir, sData)    
    lib2.checkDir(dataSubDir)
    resultsSubDir = os.path.join(resultsDir, sData)
    lib2.checkDir(resultsSubDir)
        
    inputFilePath = os.path.join(dataSubDir, '{}.txt'.format(sData))
    transactionsDf = loadCDNOW(inputFilePath, 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    print('Start:', transactionsDf['ds'].min())
    print('End:', transactionsDf['ds'].max())
    
    if dEndTrain is not None:
        transactionsCutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=None, dEnd=dEndTrain,\
            bIncludeStart=True, bIncludeEnd=False, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        transactionsHoldOutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=dEndTrain, dEnd=None,\
            bIncludeStart=False, bIncludeEnd=True, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        clientsAliveHoldOut = sorted(list(transactionsHoldOutDf['iId'].unique()))
        # client -> mean inter-purchase interval training period
        mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsCutDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            minFrequency=2)    
        mapClientToMeanPeriod = {x : round(np.mean(y), 2) for x, y in mapClientToPeriods.items()}         
    else:
        clientsAliveHoldOut = None
        
    descriptors = ['recency', 'frequency']
    
    features = ['C', 'trend_C', 'trend_short_C', # value and trend
        # 'pPoisson', 
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

    # featuresKeepOrig = ['pPoisson', 'pChurn', 'C', 'C_Orig', 'pSurvival']
    featuresKeepOrig = [] # scale everythins
    featuresScale = [x for x in features if x not in featuresKeepOrig]
        
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))
    print('dZeroState:', states.dZeroState)
    print('Number of states: ', len(states.dStateList))
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
        sAssumption = '{} customers or {} fraction from population of {} clients violate churn assumption'.\
              format(len(clientsAssumptionError), \
                     len(clientsAssumptionError) / transactionsDf['iId'].nunique(), \
                         transactionsDf['iId'].nunique())
        print(sAssumption)
        f = open(os.path.join(resultsSubDir, 'assumption.txt'), 'w')
        f.write(sAssumption)
        f.close()
    
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
    
    print('Number of TrainCensored', len(idTrainCensored))
    print('Number of TrainDead', len(idTrainDead))
    print('Number of TestCensored', len(idTestCensored))
    print('Number of TestDead', len(idTestDead))
    
###############################################################################
    # Scale features
    survivalScaledDf, featuresScalerDict = libSurvival.scaleSurvivalData(survivalDf, featuresScale, returnScaler=True)
    
    c = survivalScaledDf[features].corr()
    plotCorrelation(c, resultsSubDir, cmap='BuPu')

    xTrainDf = survivalScaledDf.loc[idTrainCensored + idTrainDead].copy(deep=True)
    xTrainDf.sort_index(inplace=True)
    
    xTestDf = survivalScaledDf.loc[idTestCensored + idTestDead].copy(deep=True)  
    xTestDf.sort_index(inplace=True)
    
    # xTestDeadDf = xTestDf[xTestDf['E'] == 1].copy(deep=True)
    # xTestDeadDf.sort_index(inplace=True)
    
    xTestCensoredDf = xTestDf[xTestDf['E'] == 0].copy(deep=True)
    xTestCensoredDf.sort_index(inplace=True)
    
    print('Train size:', len(xTrainDf))
    print('Test size:', len(xTestDf))
    
###############################################################################
    timeLineExtended = list(range(0, nStates*100, 1))
    timeLineObserved = np.arange(0, nStates)
    # metricsDict = {}
    # modelsDict = {}
    # resultsDict = {} # model -> data frame censoredDf
    # medianDict = {}
    # columnsPredict = []
    # metricsDict = {}
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
        S_LargeSeries = libSurvival.interpolateSurvivalHazard(S_LargeSeries, 
            dataType='survival', returnType='pandas', timeLine=None, survivalZero=1e-7)
        
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

    plotSurvivalCurvesUnivariate(modelsUnivariate, fitResultsDict, km, resultsSubDir)    
    plotBrierScore(modelsUnivariate, fitResultsDict, 'Univariate', resultsSubDir)
                        
###############################################################################
#   predicted survival time of COX is restricted by study interval    
#   AFT

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
                    {'n_estimators': 300, # try 150 / 7
                    'max_depth': 7,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': 'sqrt',
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'dropout_rate': 0.0,
                    'subsample': 0.9,
                    'monitor': None},
                'ComponentwiseGradientBoostingSurvivalAnalysis': 
                    {'n_estimators': 200,
                    'max_depth': 5,
                    'n_jobs': 6,
                    'verbose': True,
                    'max_features': 'sqrt'},     
                'CoxnetSurvivalAnalysis': {'l1_ratio': 0.2, 'verbose': True},
                'CoxPHSurvivalAnalysis': {'alpha': 0.1, 'verbose': True},
                'WeibullAFT': {'penalizer': 0.01},
                'LogNormalAFT': {'penalizer': 0.01},
                'LogLogisticAFT': {'penalizer': 0.01},
                'CoxPHFitter': {'penalizer': 0.001},
                'AalenAdditiveFitter': {'penalizer': 0.01},
                'CRCSplineFitter': {}
                }
        
    print('Regression')
    # smoothing_penalizer = 0
    ancillary = False
    fit_left_censoring = False
    # modelsRegression = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter',
    #     'AalenAdditiveFitter',
    #     'GradientBoostingSurvivalAnalysis', 'CoxPHSurvivalAnalysis', 'CoxnetSurvivalAnalysis']
    modelsRegression = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 
        'CoxPHFitter', 'CoxPHSurvivalAnalysis', 'GradientBoostingSurvivalAnalysis']    
#   'GeneralizedGammaRegressionFitter'
    # CRCSplineFitter # bugs
    featuresPlotDict = {'r10_FM': [10, 7, 5, 3, 1],
        'C': [0, 0.3, 0.5, 0.7, 1],
        'r10_RF': [10, 7, 5, 3, 1],
        'r10_RFM': [10, 7, 5, 3, 1]}
                
    for sModel in tqdm(modelsRegression):
        # sModel = modelsRegression[3]
        # sModel = 'WeibullAFT'
        
        if sModel in ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter', 'AalenAdditiveFitter']:
            # lifelines
            try:
                # penalizer = 0.01
                model = libSurvival.AFT(sModel, penalizer=argsDict[sModel].get('penalizer', 0.01))
                model.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', timeline=timeLineObserved, ancillary=ancillary,\
                    fit_left_censoring=fit_left_censoring)
                model.plotCoefficients(resultsSubDir)
            except:
                try:
                    print('Try higher penalizer')
                    # penalizer = 0.1
                    model = libSurvival.AFT(sModel, penalizer=argsDict[sModel].get('penalizer', 0.1))     
                    model.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', timeline=timeLineObserved, ancillary=ancillary,\
                        fit_left_censoring=fit_left_censoring)
                    model.plotCoefficients(resultsSubDir)
                except:
                    print('Try even higher penalizer')
                    # penalizer = 1
                    model = libSurvival.AFT(sModel, penalizer=argsDict[sModel].get('penalizer', 1))       
                    model.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', timeline=timeLineObserved, ancillary=ancillary,\
                        fit_left_censoring=fit_left_censoring)
                    model.plotCoefficients(resultsSubDir)
                    
            if sModel in ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter']:
                model.fitter.summary.to_excel(os.path.join(resultsSubDir, 'Summary {}.xlsx'.format(sModel)))
                plot_partial_effects_on_outcome(featuresPlotDict, featuresScalerDict, 
                                                model.fitter, sModel, resultsSubDir)
                
#   get survival and cumulative hazard functions
            S = model.survivalFunction(xTestDf, times=None) # for metrics
            H = model.cumulativeHazard(xTestDf, times=None) # for metrics

        else:
            # scikit-survival
            args = argsDict.get(sModel)
            model = libSurvival.Tree(sModel, **args)
            model.fit(xTrainDf[features + ['duration', 'E']], 'duration', 'E', sample_weight=None)
    
            # print("Fitted base learners:", modelTree.fitter.n_estimators)
            if sModel == 'GradientBoostingSurvivalAnalysis':
                featureImportance = model.featureImportance
                nFeatures = 20
                libSurvival.plotFeatureImportance(featureImportance, nFeatures, os.path.join(resultsSubDir, 'GB_feature_importance.png'))
            
            S = model.survivalFunction(xTestDf[features])
            H = model.cumulativeHazard(xTestDf[features])
                                         
#   S_Large is for estimating expected / median survival times
        if sModel.find('AFT') != -1:
            S_Large = model.survivalFunction(xTestDf, times=timeLineExtended)
        else:
            S_Large = S
            
        sampleId = sample(idTestCensored, 5)        
        if sModel == 'CoxPHFitter':
            plotSurvivalSample(sModel, sampleId, S_Large, resultsSubDir, baseline=model.fitter.baseline_survival_.squeeze())
        else:
            plotSurvivalSample(sModel, sampleId, S_Large, resultsSubDir)

        metrics = libSurvival.Metrics(xTrainDf['duration'], 
            xTrainDf['E'], xTestDf['duration'], xTestDf['E'], S, H, S_KM_Series)

        tExpected = libSurvival.getSurvivalTime(S_Large, None, method='expected')
        tExpectedFound = [i for i in tExpected if i not in [np.inf, np.nan]]
        tMedian = libSurvival.getSurvivalTime(S_Large, None, method='median')    
        tMedianFound = [i for i in tMedian if i not in [np.inf, np.nan]]

        scoresDf = pd.DataFrame(index=[sModel], columns=columnsScores)
        if sModel.find('AFT') != -1:
            scoresDf['AIC'] = round(model.AIC, 4)
        else:
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

        fitResults = {'fitter': model, 'Metrics': metrics, 'tExpected': tExpected, 'tMedian': tMedian} 
        
        if mapClientToDurationTS is not None:
#   Simulated data
            if sModel.find('AFT') != -1:
                S_LargeTestCensored = model.survivalFunction(xTestCensoredDf, times=timeLineExtended)    
            elif sModel == 'CoxPHFitter':
                S_LargeTestCensored = model.survivalFunction(xTestCensoredDf, times=None)
            else:    
                S_LargeTestCensored = model.survivalFunction(xTestCensoredDf[features])
            
            # extract manually from survival curve            
            tRemainingExpectedTestList = libSurvival.getSurvivalTime(S_LargeTestCensored, 
                xTestCensoredDf['duration'].values, method='expected', doWhatYouCan=True)
            
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
    boxPlot_IAE_ISE(modelsRegression, fitResultsDict, 'Regression', resultsSubDir)
    boxPlotSurvivalTime(modelsRegression, fitResultsDict, 'Regression', resultsSubDir, p=0.99)

#   Brier scores plot
    plotBrierScore(modelsRegression, fitResultsDict, 'Regression', resultsSubDir)

# plots errors remaining life on simulated data; on real data does nothing
    boxPlotError(modelsRegression, fitResultsDict, 'Regression', resultsSubDir)

#   SAVE scores
    scoresDf = pd.concat(scoresList, axis=0)    
    # rank models according to CI IPCW
    scoresDf.sort_values('CI IPCW', ascending=False, inplace=True)
    scoresDf['rankCI'] = range(1, len(scoresDf)+1, 1)
    scoresDf['rankCI'] = np.where(scoresDf['rankCI'] > 7, 8, scoresDf['rankCI'])    
    mapModelToRankCI = scoresDf['rankCI'].squeeze().to_dict()    
    scoresDf.to_excel(os.path.join(resultsSubDir, 'Scores.xlsx'))
    
#   save censored scored for simulated data
    if len(scoresCensoredList) > 0:
        scoresCensoredDf = pd.concat(scoresCensoredList, axis=0)  
        # rank models according to MedianAE
        scoresCensoredDf.sort_values('MedianAE', ascending=True, inplace=True)
        scoresCensoredDf['rankMedianAE'] = range(1, len(scoresCensoredDf)+1, 1)        
        mapModelToRankMedianAE = scoresCensoredDf['rankMedianAE'].squeeze().to_dict()            
        scoresCensoredDf.to_excel(os.path.join(resultsSubDir, 'Scores Censored.xlsx'))

    lib2.saveObject(os.path.join(resultsSubDir, 'fitResultsDict.dat'), fitResultsDict, protocol=4)
    
#   Plot two COX on one chart and replace from one model
    mapFeatureToCoef = {}
    for feature, coef in zip(fitResultsDict['CoxPHSurvivalAnalysis']['fitter'].fitter.feature_names_in_, \
                             fitResultsDict['CoxPHSurvivalAnalysis']['fitter'].fitter.coef_):
        mapFeatureToCoef[feature] = coef
                
    coxParamsSeries = fitResultsDict['CoxPHFitter']['fitter'].fitter.params_.copy()
    coxParamsSeries.sort_values(inplace=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(19,15))
    fitResultsDict['CoxPHFitter']['fitter'].fitter.plot(columns=None, parameter=None, ax=ax)
    for y, row in enumerate(coxParamsSeries.iteritems()):
        feature = row[0]
        x = mapFeatureToCoef.get(feature)    
        ax.scatter(x, y, color='r')        
    
    title = 'Coefficients CoxPHFitter'
    fig.suptitle(title, fontsize=16)
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    
###############################################################################
#   Predict future profit for population

    # lastStateDf = states.getLastUserState()
    # dLastState = states.dStateExistList[-1] or 
    dLastState = stateLast.dState

    idCensored = sorted(idTestCensored + idTrainCensored)
    idCensored2 = [key.split('|')[0].strip() for key in idCensored]

    xCensoredDf = pd.concat([xTrainDf, xTestDf], axis=0)
    xCensoredDf = xCensoredDf.loc[idCensored]
    xCensoredDf['meanInterPurchase'] = xCensoredDf['user'].map(mapClientToMeanPeriod)

    models = modelsUnivariate + modelsRegression
        
    moneyDailyStepDict = stateLast.getDescriptor('moneyDailyStep', activity=None, error='raise')
    moneyDailyDict = stateLast.getDescriptor('moneyDaily', activity=None, error='raise')

    if mapClientToDurationTS is not None:
    #   check with simulation
        # only censored clients
        transactionsHoldOutCensoredDf = transactionsHoldOutDf[transactionsHoldOutDf['iId'].isin(idCensored2)].copy(deep=True)
        print('{} remaining censored'.format(transactionsHoldOutCensoredDf['iId'].nunique()))
        trueProfit = transactionsHoldOutCensoredDf['Sale'].sum()
        print('True spending:', trueProfit)
        
    column1 = ['Expected', 'Median']
    column2 = ['Smooth', 'Step']
    resultsList = []
    profitDict = {}
    for sModel in models:
        # sModel = 'LogNormal'
        profitDict[sModel] = {}

        print(sModel)
        # sModel = models[0]  
        model = fitResultsDict[sModel]['fitter']
    
        if sModel in ['Exponential', 'Weibull', 'LogNormal', 'LogLogistic', 'GeneralizedGamma']:
            S_CensoredLarge = model.survival_function_at_times(timeLineExtended)
        elif sModel in ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT']:
            S_CensoredLarge = model.survivalFunction(xCensoredDf, times=timeLineExtended)
        else:
            S_CensoredLarge = model.survivalFunction(xCensoredDf[features])

        tRemainingExpectedCensoredList = libSurvival.getSurvivalTime(S_CensoredLarge, 
            xCensoredDf['duration'].values, method='expected', doWhatYouCan=True)
        tRemainingMedianCensoredList = libSurvival.getSurvivalTime(S_CensoredLarge, 
            xCensoredDf['duration'].values, method='median')    

        for i, tRemaining in enumerate([tRemainingExpectedCensoredList, tRemainingMedianCensoredList]):
            profitDict[sModel][column1[i]] = {}

            # make a constrain for remaining time to maxRemainingYears years
            xCensoredDf['tRemaining'] = [min(x, int(maxRemainingYears * 52)) for x in tRemaining]
            xCensoredDf['tRemaining'] = xCensoredDf['tRemaining'].replace(np.inf, np.nan)    
            maxRemaining = xCensoredDf['tRemaining'].max()
            if lib2.isNaN(maxRemaining):
                maxRemaining = 0
            xCensoredDf['tRemaining'] = xCensoredDf['tRemaining'].fillna(maxRemaining)
            
            xCensoredDf['dEndBuy'] = xCensoredDf['dLastState'] + \
                pd.to_timedelta(xCensoredDf['tRemaining'].mul(TS_STEP).astype(int) \
                    - xCensoredDf['meanInterPurchase'].astype(int), unit='D')   
                      
            dLastBuy = xCensoredDf['dEndBuy'].max()
            mapClientToChurnDate = xCensoredDf[['dEndBuy']].squeeze().to_dict()
            mapClientToChurnDate = {key.split('|')[0].strip(): value for key, value in mapClientToChurnDate.items()}
                    
            dDate = dLastState
            datesList = []
            while dDate < dLastBuy:
                dDate += timedelta(days=1)
                datesList.append(dDate)
            
            for j, moneyDict in enumerate([moneyDailyDict, moneyDailyStepDict]):
                profitList = []
                for dDate in tqdm(datesList):
                    dailyProfit = 0
                    for client, dEndBuy in mapClientToChurnDate.items():
                        if dEndBuy > dDate:
                            dailyProfit += moneyDict.get(client, 0)
                    profitList.append(dailyProfit)
                                    
                result1Df = pd.DataFrame(index=[sModel], columns=['Remaining', 'Descriptor', 'Predicted'])
                result1Df['Remaining'] = column1[i]
                result1Df['Descriptor'] = column2[j]                
                result1Df['Predicted'] = sum(profitList)
                if mapClientToDurationTS is not None:
#   Simulated data
                    result1Df['True'] = trueProfit
                    result1Df['Error'] = trueProfit - result1Df['Predicted']
                    result1Df['AbsError'] = result1Df['Error'].abs()
                                        
                resultsList.append(result1Df)
                
                profitDict[sModel][column1[i]][column2[j]] = (datesList, profitList)
                           
    profitDf = pd.concat(resultsList, axis=0)
    if mapClientToDurationTS is not None:
        profitDf.sort_values('AbsError', inplace=True)
    else:
        profitDf.sort_values('Predicted', inplace=True)

    lib2.saveObject(os.path.join(resultsSubDir, 'profitDict.dat'), profitDict, protocol=4)

    profitDf['rankCI'] = profitDf.index.map(mapModelToRankCI)
    if mapClientToDurationTS is not None:
        profitDf['rankMedianAE'] = profitDf.index.map(mapModelToRankMedianAE)
        
    profitDf.to_excel(os.path.join(resultsSubDir, 'Profit.xlsx'))


###############################################################################    
if __name__ == "__main__":

    random.seed(101)
    maxRemainingYears = 10
    
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'o', 'k']

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data', 'Simulated20')
    resultsDir = os.path.join(workDir, 'results', 'Simulated20')
    # 'SimulatedShort001' to 'SimulatedShort020'
    dirsList = [x for x in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, x))]
    
    for sData in dirsList:        
        main(sData, dataDir, resultsDir)
    
    
    
    
    
    
    
    
    
    
    
    
    