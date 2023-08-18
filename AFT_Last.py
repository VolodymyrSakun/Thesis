# COX + AFT + Trees
# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#fitting-the-regression

# metrics
# https://cran.r-project.org/web/packages/SurvMetrics/vignettes/SurvMetrics-vignette.html

# survival refgression 
# plots


import os
import pandas as pd
import numpy as np
# from datetime import timedelta
import lib2
# from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
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
# from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
# import lib3
# from State import TS_STEP
# from States import loadCDNOW
# import State
# from States import States
# from datetime import date
from sklearn.metrics import mean_absolute_error
# from Distributions import Statistics
# from sksurv.datasets import get_x_y
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.ensemble import ExtraSurvivalTrees
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
import libSurvival
# from lifelines import CoxTimeVaryingFitter
# from FrameArray import FrameArray
# from math import exp
# from lifelines import KaplanMeierFitter
# from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, NelsonAalenFitter, PiecewiseExponentialFitter, GeneralizedGammaFitter, SplineFitter
from lifelines import ExponentialFitter
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
# from FrameArray import FrameArray
# from sksurv.metrics import concordance_index_censored
# from sksurv.metrics import concordance_index_ipcw
# from sksurv.datasets import load_whas500
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.linear_model import CoxnetSurvivalAnalysis
# from Encoder import DummyEncoderRobust
# from sklearn.linear_model import LogisticRegression
# import lib4
# from math import log
from sklearn.metrics import median_absolute_error

# def makeHugeDTPO(data, columnsX, columnsTS):
#     dataCopy = data.copy(deep=True)
#     newColumns = []
#     for column in tqdm(columnsX):
#         tsDf = dataCopy[columnsTS].copy(deep=True)
#         tsDf = tsDf.multiply(dataCopy[column], axis='index')
#         columns = ['{}_{}'.format(column, x) for x in tsDf.columns]
#         newColumns.extend(columns)
#         tsDf.columns = columns
#         dataCopy = pd.concat([dataCopy, tsDf], axis=1)
#     dataCopy.drop(columns=columnsX, inplace=True)
#     return dataCopy, newColumns

def plotBrierScore(bsDict, ibsDict):
#   Plot brier scores for models
    fig, ax = plt.subplots(2, 2, figsize=(19,15))
    i = 0
    j = 0
    for model, bsSeries in bsDict.items():
        ibs = ibsDict[model]
        ax[i][j].plot(bsSeries)
        ax[i][j].title.set_text('{}. IBS: {}'.format(model, round(ibs, 6)))
        ax[i][j].set_xlabel('Time lived in weeks')
        ax[i][j].set_ylabel('Brier Score')
        j += 1
        if j > 1:
            i += 1
            j = 0
    title = 'Brier score'
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
    
###############################################################################    
if __name__ == "__main__":

    # CDNOW
    args = {'dStart': None, 'holdOut': None,
        'dZeroState': None, 'dEnd': None}
    
    # SimulatedShort001
    # args = {'dStart': None, 'holdOut': None,
    #     'dZeroState': date(2000, 7, 1), 'dEnd': date(2004, 7, 1)}

    # casa_2
    # args = {'dStart': None, 'holdOut': None,
    #     'dZeroState': date(2016, 1, 1), 'dEnd': date(2019, 1, 1)}
        
    bGRU_only = False
    bUseGRU_predictions = False
    bUseLongTrends = True
    bUseShortTrends = True
    bUseRecency = True
    bUseLoyalty = False
    

    random_state = 101
    # lag = 0
    pMax = 0.9999
    
    sData = 'CDNOW' # CDNOW casa_2 plex_45 Retail Simulated5 SimulatedShort001
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
        mergeTransactionsPeriod=TS_STEP, minFrequency=MIN_FREQUENCY)
    
    if dEndTrain is not None:
        transactionsCutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=None, dEnd=dEndTrain,\
            bIncludeStart=True, bIncludeEnd=False, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        transactionsHoldOutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=dEndTrain, dEnd=None,\
            bIncludeStart=False, bIncludeEnd=True, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
        clientsAliveHoldOut = sorted(list(transactionsHoldOutDf['iId'].unique()))
    else:
        clientsAliveHoldOut = None
        
    descriptors = ['recency', 'frequency']
    
    #!!! do not use pPoisson and loyalty features
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
    features = [
        'C_Orig', 'trend_C_Orig', 'trend_short_C_Orig', 
        'moneySum', 'moneyMedian', 'moneyDailyStep',
        'r10_F', 'r10_M',
        'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC',
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC',
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC']



#!!! feature selection    
    if not bUseRecency:
        # no recency
        features = [x for x in features if x not in ['r10_R', 'r10_RF', 'r10_RFM', 'r10_RFMC', 'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']]

    if not bUseLoyalty:
        features = [x for x in features if x not in ['r10_L', 'r10_LFM', 'r10_LFMC', 'trend_r10_LFMC', 'trend_short_r10_LFMC']]
     
    if not bUseGRU_predictions:
#   Use probability of churn obtained from ANN as input feature for survival        
        features = [x for x in features if x not in ['pChurn','trend_pChurn', 'trend_short_pChurn']]
    
    if not bUseLongTrends:
        filtered = []
        for feature in features:
            if feature.find('trend') != -1 and feature.find('trend_short') == -1:
                continue
            filtered.append(feature)
        features = filtered

    if not bUseShortTrends:
        filtered = []
        for feature in features:
            if feature.find('trend_short') != -1:
                continue
            filtered.append(feature)
        features = filtered

    if bGRU_only:
        features = ['pChurn','trend_pChurn', 'trend_short_pChurn']

    featuresKeepOrig = ['pPoisson', 'pChurn', 'C_Orig', 'pSurvival']
    featuresScale = [x for x in features if x not in featuresKeepOrig]
        
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))
    stateLast = states.loadLastState()

    if sData == 'casa_2':

        from States import States
        statesFull = States(dataSubDir, transactionsDf, tsStep=TS_STEP, dZeroState=dZeroState, holdOut=holdOut)
        # statesFull.mapClientToStatus
        # statesFull.mapClientTo_qDeath
        statesCut = States(dataSubDir, transactionsCutDf, tsStep=TS_STEP, dZeroState=dZeroState, holdOut=holdOut)
    
        # were active in train period but dead in holdout period
        clientsDeadHoldOut = []
        for client in statesCut.mapClientToStatus.keys():
            statusTrain = statesCut.mapClientToStatus[client]
            if statusTrain != 0:
                # not censored
                continue
            statusFull = statesFull.mapClientToStatus[client]
            if statusFull == 1:
                # dies after dEndTrain
                clientsDeadHoldOut.append(client)
                
        mapClientToChurnDate = statesFull.getChurnDate(transactionsDf)
        # stateLast = states.loadLastState()
    
        mapClientToBirthState = {}
        for client, clientObj in stateLast.Clients.items():
            dBirthState = clientObj.descriptors.get('dBirthState')
            mapClientToBirthState[client] = dBirthState
            
        clientsDeadHoldOut = sorted(list(set(clientsDeadHoldOut).intersection(set(stateLast.Clients.keys()))))
        
        mapDeadToDuration = {}
        mapDeadToDurationTS = {}
        for client in clientsDeadHoldOut:
            duration = (mapClientToChurnDate[client] - mapClientToBirthState[client]).days
            durationTS = int(duration / TS_STEP)
            mapDeadToDuration[client] = duration
            mapDeadToDurationTS[client] = durationTS
    
    
    # true durations for simulated data
    if sData in ['CDNOW', 'Retail']:
        mapClientToDurationTS = None
    elif sData == 'casa_2':
        mapClientToDurationTS = mapDeadToDurationTS        
    elif sData.find('Simulated') != -1:
        mapClientToDuration, mapClientToDurationTS = states.getTrueLifetime(transactionsDf)
        mapClientToChurnDate = states.getChurnDate(transactionsDf)
    else:
        raise RuntimeError()
        
    nStates = len(states.dStateList)
    
    print('Prepare inputs for survival')
    survivalDf = states.makeSurvivalData(descriptors + features, 'D_TS')
    
    clientsDead = list(survivalDf[survivalDf['E'] == 1]['user'])
    
    # clients that died in training intercval according to q95 assumption, but truely alive
    # made purchases in hold-out period
    
    if clientsAliveHoldOut is not None:
        clientsAssumptionError = set(clientsDead).intersection(set(clientsAliveHoldOut))
    
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

    if sData == 'casa_2':
        tmp = survivalDf[survivalDf['user'].isin(clientsDeadHoldOut)]
    else:        
        tmp = survivalDf[survivalDf['E'] == 0]
    idCensored = sorted(list(tmp.index))
    idFit = sorted(list(survivalDf.index))
    del(tmp)


###############################################################################
    # Scale features
    xScaled, xPredict2 = libSurvival.scaleSurvivalData(survivalDf, featuresScale, survivalCensoredDf)

#!!! check to avoid matrix singularity
    c = xScaled.corr()

    # Censored data
    xPredict = xScaled.loc[idCensored].copy(deep=True)
    
#!!! select censored last state  
    x_Predict = xPredict2 # xPredict, xPredict2

    users = [x.split(' | ')[0] for x in x_Predict.index]
    # Template data frame
    censoredCopyDf = x_Predict[['duration']].copy(deep=True)
    censoredCopyDf['user'] = users
    censoredCopyDf = censoredCopyDf.join(survivalCensoredDf[['dBirthState', 'dLastState'] + descriptors], how='left')
    censoredCopyDf = censoredCopyDf[['user', 'dBirthState', 'dLastState', 'duration'] + descriptors]

    if mapClientToDurationTS is not None:
        censoredCopyDf['lifetimeTrue'] = censoredCopyDf['user'].map(mapClientToDurationTS)
        censoredCopyDf['churnDate'] = censoredCopyDf['user'].map(mapClientToChurnDate)
        censoredCopyDf['remainingTrue'] = censoredCopyDf['lifetimeTrue'] - censoredCopyDf['duration']
        censoredCopyDf = censoredCopyDf[['user', 'dBirthState', 'dLastState', 'churnDate', 'duration', 'lifetimeTrue', 'remainingTrue'] + descriptors]
        
###############################################################################
    timeLine = list(range(0, nStates*10, 1))
    timeLineObserved = np.arange(0, nStates)
    metricsDict = {}
    modelsDict = {}
    resultsDict = {} # model -> data frame censoredDf
    medianDict = {}
    columnsPredict = []
###############################################################################
copy file    
review this crap
#   Baseline    
    sModel = 'Exponential'
    T = xScaled['duration']
    E = xScaled['E']
    
    fitterLarge = ExponentialFitter().fit(T, E, timeline=timeLine)  
    fitterObserved = ExponentialFitter().fit(T, E, timeline=timeLineObserved)  
        
    S_Observed = fitterObserved.survival_function_
    H_Observed = fitterObserved.cumulative_hazard_
    S_LargeSeries = fitterLarge.survival_function_
    
    # sExponential = exf.survival_function_
    S_LargeSeries.index = [int(x) for x in S_LargeSeries.index]
    S_LargeSeries = S_LargeSeries['Exponential_estimate'] # Series

    S_Large = pd.DataFrame(columns=x_Predict.index, index=S_LargeSeries.index)
    for column in tqdm(S_Large.columns):
        S_Large[column] = S_LargeSeries
    
    # extract manually from survival curve  
    remainingExpectedList, remainingMedianList = libSurvival.getRemainingSurvivalTime( \
        censoredCopyDf['duration'].values, S_Large, p=0.95)            
        
    censoredDf = censoredCopyDf.copy(deep=True)    
    censoredDf['remainingExpected'] = remainingExpectedList
    censoredDf['remainingMedian'] = remainingMedianList

    # Common metrics
    sE = pd.DataFrame(columns=xScaled.index, index=S_Observed.index)    
    for column in sE.columns:
        sE[column] = S_Observed
    
    hE = pd.DataFrame(columns=xScaled.index, index=H_Observed.index)
    for column in hE.columns:
        hE[column] = H_Observed    

    metrics = libSurvival.Metrics(xScaled['duration'], 
        xScaled['E'], xScaled['duration'], xScaled['E'], sE, hE)       
    metrics.getCI(xScaled['duration'], [fitterObserved.median_survival_time_ for x in range(len(xScaled))], xScaled['E'])
    
    # MAE if simulated
    if 'churnDate' in censoredDf:
       
        metrics.MAE_Expected = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
        metrics.MAE_Median = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

        metrics.MedianAE_Expected = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
        metrics.MedianAE_Median = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))                
        
        metrics.R2_Expected = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
        metrics.R2_Median = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

        metrics.MAPE_Expected = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
        metrics.MAPE_Median = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

        metrics.RemainingExpectedMedian = int(censoredDf['remainingExpected'].median())
        metrics.RemainingMedianMedian = int(censoredDf['remainingMedian'].median())
        
    metricsDict[sModel] = metrics
    resultsDict[sModel] = censoredDf
    
    
###############################################################################
    
#   AFT
    print('AFT + COX')
    smoothing_penalizer = 0
    ancillary = True
    fit_left_censoring = False
        
    modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter']#, 'GeneralizedGammaRegressionFitter', 'AalenAdditiveFitter']

    for sModel in tqdm(modelsAFT):
        # sModel = modelsAFT[0]
        penalizer = 0.1
        try:
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xScaled[features + ['duration', 'E']], 'duration', 'E', timeline=timeLine, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)
        except:
            print('Try higher penalizer')
            penalizer = 1
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xScaled[features + ['duration', 'E']], 'duration', 'E', timeline=timeLine, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)

        S_Large = modelAFT.predictSurvival(x_Predict, times=timeLine)
        
        # extract manually from survival curve  
        remainingExpectedList, remainingMedianList = libSurvival.getRemainingSurvivalTime( \
            censoredCopyDf['duration'].values, S_Large, p=pMax)                       
            
        remainingConditional = modelAFT.predictSurvivalTime(x_Predict[features], what='median', 
            conditional_after=x_Predict['duration'], fixNan=None, p=pMax, replaceInf=True)
            
        remainingConditionalList = list(remainingConditional.values)
                
        censoredDf = censoredCopyDf.copy(deep=True)
        censoredDf['remainingExpected'] = remainingExpectedList
        censoredDf['remainingMedian'] = remainingMedianList
        censoredDf['remainingConditional'] = remainingConditionalList
        
        # Common metrics
        timeMedianTrain = modelAFT.predictSurvivalTime(xScaled[features], what='median', 
            fixNan=None, p=pMax, replaceInf=True)
            
        metrics = libSurvival.Metrics(xScaled['duration'], 
            xScaled['E'], xScaled['duration'], xScaled['E'], 
            modelAFT.predictSurvival(xScaled, times=timeLineObserved), 
            modelAFT.predictHazard(xScaled, times=timeLineObserved))
        metrics.getCI(xScaled['duration'], timeMedianTrain, xScaled['E'])
        
        # MAE if simulated
        if 'remainingTrue' in censoredDf:            
            metrics.MAE_Expected = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MAE_Median = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))
            metrics.MAE_Conditional = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingConditional']))
            
            metrics.MedianAE_Expected = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MedianAE_Median = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))                
            metrics.MedianAE_Conditional = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingConditional']))
            
            metrics.R2_Expected = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.R2_Median = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingMedian']))
            metrics.R2_Conditional = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingConditional']))

            metrics.MAPE_Expected = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MAPE_Median = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))
            metrics.MAPE_Conditional = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingConditional']))

            metrics.RemainingExpectedMedian = int(censoredDf['remainingExpected'].median())
            metrics.RemainingMedianMedian = int(censoredDf['remainingMedian'].median())

            censoredDf['error'] = (censoredDf['remainingTrue'] - censoredDf['remainingMedian']).abs()
            
        metricsDict[sModel] = deepcopy(metrics)
        resultsDict[sModel] = deepcopy(censoredDf)
        
        
    # c = resultsDict['WeibullAFT']
    # c['error'] = abs(c['remainingTrue'] - c['remainingMedian'])
    # c['pSurvival'] = x_Predict['pSurvival']
        
    
    # remainingTrue = censoredDf['remainingTrue'].values        
    # Plot  scatters
    # idx = 0
    # N = 4
    # fig3, ax = plt.subplots(N, N, figsize=(19,15))
    # for i in range(0, N, 1):
    #     for j in range(0, N, 1):
    #         idx = i * N + j
    #         if idx < len(data2):
    #             ax[i][j].scatter(data3[idx], data2[idx], s=2)
    #             ax[i][j].title.set_text('{}'.format(models[idx].replace('dM_', '')))
    #             ax[i][j].set_ylabel('Predicted duration')            
    #             ax[i][j].set_xlabel('True duration')

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        
    # fileName = 'Scatter True vs. predicted duration.png'
    # plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    # plt.close()
        

    # plt.scatter(censoredDf['recency'], censoredDf['error'], s=2)
        
    # from Rank import Rank
    # rank10 = Rank(method='Quantiles', nBins=10, dropZero=False, \
    #     returnZeroClass=True, includeMargin='right', zeroCenter=False)    
        
    # rank10.fit(censoredDf['recency'])
    # r10_R = rank10.getClusters(censoredDf['recency'], reverse=False)
        
    # censoredDf['r10_R'] = r10_R
    
    # g = censoredDf.groupby('r10_R').agg({'error': 'median'})
    
    
    # censoredDf['error'].mean()
    
###############################################################################
# Tree
    print('Trees')
    argsTree = {'n_estimators': 100,
                'max_depth': 5,
                'n_jobs': 6,
                'verbose': True,
                'max_features': 'sqrt'}

    # modelsTree = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
    #     'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis',
    #     'CoxPHSurvivalAnalysis', 'CoxnetSurvivalAnalysis']
    
    modelsTree = ['GradientBoostingSurvivalAnalysis']
    
    for sModel in modelsTree:
        # sModel = modelsTree[2]
        # sModel = 'CoxPHSurvivalAnalysis'
        
        modelTree = libSurvival.Tree(sModel, **argsTree)
        modelTree.fit(xScaled[features + ['duration', 'E']], 'duration', 'E', sample_weight=None)

        if sModel == 'GradientBoostingSurvivalAnalysis':
            featureImportance = modelTree.featureImportance
            nFeatures = 20
            libSurvival.plotFeatureImportance(featureImportance, nFeatures, os.path.join(resultsSubDir, 'GB_feature_importance.png'))

        S_Large = modelTree.predictSurvival(x_Predict[features], returnType='pandas', 
            interpolate=True, timeLine=x_Predict['duration'].max())
                       
        # extract manually from survival curve  
        remainingExpectedList, remainingMedianList = libSurvival.getRemainingSurvivalTime( \
            censoredCopyDf['duration'].values, S_Large, p=pMax)
                    
        censoredDf = censoredCopyDf.copy(deep=True)
        censoredDf['remainingExpected'] = remainingExpectedList
        censoredDf['remainingMedian'] = remainingMedianList      
        
        # Common metrics        
        s = modelTree.predictSurvival(xScaled[features], returnType='pandas')
        h = modelTree.predictHazard(xScaled[features], returnType='pandas')

        timeMedianTrain = modelTree.predictSurvivalTime(xScaled[features], p=0.5, replaceInf=True)
        
        metrics = libSurvival.Metrics(xScaled['duration'], 
            xScaled['E'], xScaled['duration'], xScaled['E'], 
            modelTree.predictSurvival(xScaled[features], returnType='pandas'),
            modelTree.predictHazard(xScaled[features], returnType='pandas'))
        metrics.getCI(xScaled['duration'], timeMedianTrain, xScaled['E'])
        
        # MAE if simulated
        if 'remainingTrue' in censoredDf:
            metrics.MAE_Expected = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MAE_Median = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))
            
            metrics.MedianAE_Expected = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MedianAE_Median = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))                
            
            metrics.R2_Expected = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.R2_Median = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

            metrics.MAPE_Expected = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
            metrics.MAPE_Median = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

            metrics.RemainingExpectedMedian = int(censoredDf['remainingExpected'].median())
            metrics.RemainingMedianMedian = int(censoredDf['remainingMedian'].median())
            
            censoredDf['error'] = (censoredDf['remainingTrue'] - censoredDf['remainingMedian']).abs()

        metricsDict[sModel] = deepcopy(metrics)
        resultsDict[sModel] = deepcopy(censoredDf)
        
        # a = modelTree.fitter.coef_
        # modelTree.fitter.feature_names_in_
###############################################################################
#   COX TV

#     mapClientToFrame, mapIdToStatus, mapClientToFirstState = states.loadFeatures(features, nTimeSteps=0)
#     ppfDf = states.buildPersonPeriodFrame(features, mapClientToFrame)
#     xPPF = libSurvival.scaleSurvivalData(ppfDf, featuresScale)
#     xPPF_All = xPPF.drop_duplicates(subset=['user'], keep='last')
#     xPPF_Censored = xPPF[xPPF.index.isin(idCensored)].copy(deep=True)
#     idCensoredTV = list(xPPF_Censored.index)
#     xPPF_Censored.sort_index(inplace=True)
#     clientsCensored = sorted(list(xPPF_Censored['user'].unique()))
#     durations = censoredCopyDf[censoredCopyDf['user'].isin(clientsCensored)]['duration']
#     durations.sort_index(inplace=True)

#     i = xPPF[xPPF.index.isin(idFit)].copy(deep=True)
#     idFit = sorted(list(i.index))
    
# #   COX time-varying regression fit
#     ctv = CoxTimeVaryingFitter(penalizer=0.00001)
#     ctv.fit(xPPF, id_col='user', event_col='E', 
#         start_col='start', stop_col='stop', show_progress=True, fit_options={'step_size': 0.05})       
#     coxSummary = ctv.summary

# #   Plot hazard Rate    
#     plt.figure(1, figsize=(19,10)) 
#     ax = ctv.plot()
#     title = 'Hazard rate Cox time-varying'
#     plt.title(title)
#     fileName = '{}{}'.format(title, '.png')
#     plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
#     plt.close() 
    
#     H_All, S_All = libSurvival.get_H_and_S(ctv, xPPF_All, timeLine=None, returnType='Frame')
#     H_Censored, S_Censored = libSurvival.get_H_and_S(ctv, xPPF_Censored, timeLine=xPPF_Censored['stop'].max(), returnType='pandas')

#     # extract manually from survival curve  
#     remainingExpectedList, remainingMedianList = libSurvival.getRemainingSurvivalTime( \
#         durations.values, S_Censored, p=0.95) 
            
#     censoredDf = censoredCopyDf.copy(deep=True)
#     censoredDf = censoredDf[censoredDf.index.isin(idCensoredTV)].copy(deep=True)
#     censoredDf.sort_index(inplace=True)
#     censoredDf['remainingExpected'] = remainingExpectedList
#     censoredDf['remainingMedian'] = remainingMedianList 
        
#     timeMedianTrain = libSurvival.getMedianSurvivalTime(S_All, p=0.5, replaceInf=True)
        
#     tmp = xPPF_All[['user', 'E']].copy(deep=True)
#     tmp = tmp.merge(survivalDf[['user', 'duration']], how='left', on='user')    
#     metrics = libSurvival.Metrics(tmp['duration'], tmp['E'], tmp['duration'], tmp['E'], 
#         S_All.to_pandas(), H_All.to_pandas())
#     metrics.getCI(tmp['duration'], timeMedianTrain, tmp['E'])

#     # MAE if simulated
#     if 'remainingTrue' in censoredDf:            
#         metrics.MAE_Expected = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
#         metrics.MAE_Median = float(mean_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))
        
#         metrics.MedianAE_Expected = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
#         metrics.MedianAE_Median = float(median_absolute_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))                
        
#         metrics.R2_Expected = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
#         metrics.R2_Median = float(r2_score(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

#         metrics.MAPE_Expected = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingExpected']))
#         metrics.MAPE_Median = float(mean_absolute_percentage_error(censoredDf['remainingTrue'], censoredDf['remainingMedian']))

#         metrics.RemainingExpectedMedian = int(censoredDf['remainingExpected'].median())
#         metrics.RemainingMedianMedian = int(censoredDf['remainingMedian'].median())

#         censoredDf['error'] = (censoredDf['remainingTrue'] - censoredDf['remainingMedian']).abs()

#     metricsDict['COX_TV'] = deepcopy(metrics)
#     resultsDict['COX_TV'] = deepcopy(censoredDf)
        
    if 'remainingTrue' in censoredDf:

        RemainingTrueMedian = int(censoredDf['remainingTrue'].median())
        RemainingTrueMean = int(censoredDf['remainingTrue'].mean())    
        print(RemainingTrueMedian)
        print(RemainingTrueMean)
    
    models = sorted(list(metricsDict.keys()))
    metricsList = ['CI', 'IBS', 'aucMean', 'MAE_Conditional', 'MAE_Expected', 'MAE_Median', \
        'MedianAE_Conditional', 'MedianAE_Expected', 'MedianAE_Median',
        'MAPE_Conditional', 'MAPE_Expected', 'MAPE_Median', 'R2_Conditional', \
        'R2_Expected', 'R2_Median', 'RemainingExpectedMedian', 'RemainingMedianMedian']
    resultsDf = pd.DataFrame(index=models, columns= metricsList)
        
    
    for sModel, metrics in metricsDict.items():
        # break
        resultsDf['CI'].loc[sModel] = metrics.CI
        resultsDf['IBS'].loc[sModel] = metrics.IBS
        resultsDf['aucMean'].loc[sModel] = metrics.aucMean
        resultsDf['MAE_Conditional'].loc[sModel] = metrics.MAE_Conditional
        resultsDf['MAE_Expected'].loc[sModel] = metrics.MAE_Expected
        resultsDf['MAE_Median'].loc[sModel] = metrics.MAE_Median        
        resultsDf['MedianAE_Conditional'].loc[sModel] = metrics.MedianAE_Conditional
        resultsDf['MedianAE_Expected'].loc[sModel] = metrics.MedianAE_Expected
        resultsDf['MedianAE_Median'].loc[sModel] = metrics.MedianAE_Median
        resultsDf['MAPE_Conditional'].loc[sModel] = metrics.MAPE_Conditional
        resultsDf['MAPE_Expected'].loc[sModel] = metrics.MAPE_Expected
        resultsDf['MAPE_Median'].loc[sModel] = metrics.MAPE_Median
        resultsDf['R2_Conditional'].loc[sModel] = metrics.R2_Conditional
        resultsDf['R2_Expected'].loc[sModel] = metrics.R2_Expected
        resultsDf['R2_Median'].loc[sModel] = metrics.R2_Median    
        resultsDf['RemainingExpectedMedian'].loc[sModel] = metrics.RemainingExpectedMedian
        resultsDf['RemainingMedianMedian'].loc[sModel] = metrics.RemainingMedianMedian
    
    resultsDf = resultsDf.transpose()
    
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'Metrics.xlsx')) as writer:
        resultsDf.to_excel(writer)    
    
#   Save predictions to excel
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'Censored.xlsx')) as writer:
        for sModel, censoredDf in resultsDict.items():
            sheet_name = '{}'.format(sModel)[0:30]
            censoredDf.to_excel(writer, sheet_name=sheet_name)

        

# Box plots of errors

    if 'remainingTrue' in censoredDf:
    
        data = []
        data2 = []
        data3 = []
        models = list(resultsDict.keys())
        for sModel in models:
            censoredDf = resultsDict[sModel]
            if 'remainingConditional' in censoredDf.columns:
                columnRemaining = 'remainingConditional'
            else:
                columnRemaining = 'remainingMedian'            
            error = (censoredDf[columnRemaining] - censoredDf['remainingTrue']).abs()
            data.append(error.values)    
            data2.append(censoredDf[columnRemaining].values) 
            data3.append(censoredDf['remainingTrue'].values) 
    
        title = 'Duration error box plots'
        fig3, ax = plt.subplots(1, 1, figsize=(19,15))
        bp = ax.boxplot(data, vert = 1)
        ax.set_xticklabels(models, rotation=45)
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        ax.set_ylabel('Error')
        ax.title.set_text(title)
        fileName = '{}.png'.format(title)
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close()      
    
        
        # remainingTrue = censoredDf['remainingTrue'].values        
        # Plot  scatters
        idx = 0
        N = 3
        fig3, ax = plt.subplots(N, N, figsize=(19,15))
        for i in range(0, N, 1):
            for j in range(0, N, 1):
                idx = i * N + j
                if idx < len(data2):
                    ax[i][j].scatter(data3[idx], data2[idx], s=2)
                    ax[i][j].title.set_text('{}'.format(models[idx].replace('dM_', '')))
                    ax[i][j].set_ylabel('Predicted duration')            
                    ax[i][j].set_xlabel('True duration')
    
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
            
        fileName = 'Scatter True vs. predicted duration.png'
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close()
        
    
    yamlDict = {'Features': features}
    for key, value in metricsDict.items():
        yamlDict[key] = value.to_dict()

    if 'remainingTrue' in censoredDf:        
        yamlDict['RemainingTrueMean'] = RemainingTrueMean
        yamlDict['RemainingTrueMedian'] = RemainingTrueMedian
        yamlDict['nViolatingClients'] = len(clientsAssumptionError)    
        yamlDict['ViolatingAssumptionFraction'] = len(clientsAssumptionError) / len(survivalDf)
        
    lib2.saveYaml(os.path.join(resultsSubDir, 'Metrics.yaml'), yamlDict, sort_keys=False)

       
    train / test split
    Two additional evaluation metrics for survival models used are IAE (Integrated Absolute Error) and ISE (Integrated Square Error). ([6]; [16];).
    the non-parametric KM estimation method is used to obtain the approximate expression of S(t)
    MAE on test dead
    
"""    
###############################################################################


    lastStateDf = states.getLastUserState()
    mapClientToStatus = lastStateDf[['user', 'status']].set_index('user')['status'].to_dict()
    mapClientToLastState = lastStateDf[['user', 'dLastState']].set_index('user')['dLastState'].to_dict()
    
#                    user dBirthState  dLastState  status
# iD                                                     
# 0001 | 2000-08-26  0001  2000-07-01  2000-08-26       1
# 0002 | 2002-12-14  0002  2002-11-09  2002-12-14       1
# 0003 | 2002-08-17  0003  2002-04-06  2002-08-17       1
# 0004 | 2001-10-27  0004  2001-01-20  2001-10-27       1
# 0005 | 2001-11-24  0005  2001-08-25  2001-11-24       1
    
    mapClientToFrame, mapIdToStatus, mapClientToBirthState = states.loadFeatures(features, nTimeSteps=0)

    dataTrainList = []
    durationTrainList = []
    eventTrainList = []
    userTrainList = []    
    dataCensoredList = []
    durationCensoredList = []
    userCensoredList = []
    lastStateList = []
    for client, frame in tqdm(mapClientToFrame.items()):
        # if client == '0002':
        #     break
        # if client == '0006':
        #     break
    
        status = mapClientToStatus.get(client)      
        if status is None:
            # Not born
            continue
        
        dBirthState = mapClientToBirthState[client]
        dLastState = mapClientToLastState[client]  
        if dBirthState == dLastState:
            # third purchase is the last one
            continue
        
        if status == 0:
            frameCensored = frame.sliceFrame(rows=[frame.rowNames[-1]], columns=None)
            durationMax = int((dLastState - dBirthState).days / TS_STEP) # weeks; for censored as well            
            dataCensoredList.append(frameCensored.to_pandas())
            durationCensoredList.append(durationMax)
            userCensoredList.append(client) 
            lastStateList.append(dLastState)            
            dStateList = [x for x in frame.rowNames if x > dBirthState and x <= dLastState]
            durations = [x for x in range(1, len(dStateList)+1)]
            events = [0 for x in range(0, len(durations))]
            users = [client for x in range(0, len(durations))]
            frameTrain = frame.sliceFrame(rows=dStateList, columns=None)
        elif status == 1:
            try to restrict to death observed at q95
            durationMax = int((dLastState - dBirthState).days / TS_STEP) # weeks
            dStateList = [x for x in frame.rowNames if x > dBirthState]
            events = []
            durations = []
            for dState in dStateList:
                if dState < dLastState:
                    events.append(0)
                else:
                    events.append(1)  
                duration = int((dState - dBirthState).days / TS_STEP) # weeks
                durations.append(min(duration, durationMax))                          
            frameTrain = frame.sliceFrame(rows=dStateList, columns=None)    
            users = [client for x in range(0, len(durations))]
        else:
            continue
        
        dataTrainList.append(frameTrain.to_pandas())
        durationTrainList.extend(durations)
        eventTrainList.extend(events)
        userTrainList.extend(users)
        
        

    dataTrainDf = pd.concat(dataTrainList, axis=0)
    dataTrainDf['dState'] = dataTrainDf.index
    dataTrainDf['user'] = userTrainList
    dataTrainDf['duration'] = durationTrainList
    dataTrainDf['E'] = eventTrainList    
    dataTrainDf['iD'] = dataTrainDf['user'] + ' | ' + dataTrainDf['dState'].astype(str)
    dataTrainDf.set_index('iD', drop=True, inplace=True)
    dataTrainDf.sort_index(inplace=True)
    dataTrainDf = dataTrainDf[['user', 'dState', 'duration', 'E'] + features]
    
    # a = dataTrainDf.iloc[0:1000]
    
    dataCensoredDf = pd.concat(dataCensoredList, axis=0)
    dataCensoredDf['dLastState'] = lastStateList
    dataCensoredDf['user'] = userCensoredList
    dataCensoredDf['duration'] = durationCensoredList    
    dataCensoredDf['iD'] = dataCensoredDf['user'] + ' | ' + dataCensoredDf['dLastState'].astype(str)
    dataCensoredDf.set_index('iD', drop=True, inplace=True)
    dataCensoredDf.sort_index(inplace=True)
    dataCensoredDf = dataCensoredDf[['user', 'dLastState', 'duration'] + features]
        
    
    
    xTrainScaledExt, xPredictScaledExt = libSurvival.scaleSurvivalData(dataTrainDf, featuresScale, dataCensoredDf)

    dataCensoredCopyDf = xPredictScaledExt[['user', 'dLastState', 'duration']].copy(deep=True)

    if mapClientToDurationTS is not None:
        dataCensoredCopyDf['lifetimeTrue'] = dataCensoredCopyDf['user'].map(mapClientToDurationTS)
        dataCensoredCopyDf['churnDate'] = dataCensoredCopyDf['user'].map(mapClientToChurnDate)
        dataCensoredCopyDf['remainingTrue'] = dataCensoredCopyDf['lifetimeTrue'] - dataCensoredCopyDf['duration']
        dataCensoredCopyDf = dataCensoredCopyDf[['user', 'dLastState', 'churnDate', 'duration', 'lifetimeTrue', 'remainingTrue']]
    
    print('AFT + COX')
    smoothing_penalizer = 0
    ancillary = True
    fit_left_censoring = False
        
    modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter'] #, 'AalenAdditiveFitter']

    for sModel in tqdm(modelsAFT):
        # sModel = modelsAFT[3]
        # sModel = 'GeneralizedGammaRegressionFitter'
        penalizer = 0.1
        try:
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xTrainScaledExt[features + ['duration', 'E']], 'duration', 'E', timeline=timeLine, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)
        except:
            print('Try higher penalizer')
            penalizer = 1
            modelAFT = libSurvival.AFT(sModel, penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)        
            modelAFT.fit(xTrainScaledExt[features + ['duration', 'E']], 'duration', 'E', timeline=timeLine, ancillary=ancillary,\
                fit_left_censoring=fit_left_censoring)
            modelAFT.plotCoefficients(resultsSubDir)

        S_Large = modelAFT.predictSurvival(xPredictScaledExt[features], times=timeLine)
    
        # extract manually from survival curve  
        remainingExpectedList, remainingMedianList = libSurvival.getRemainingSurvivalTime( \
            xPredictScaledExt['duration'].values, S_Large, p=0.95)                       
            
        remainingConditional = modelAFT.predictSurvivalTime(xPredictScaledExt[features], what='median', 
            conditional_after=xPredictScaledExt['duration'], fixNan=None, p=0.95, replaceInf=True)
            
        remainingConditionalList = list(remainingConditional.values)
                
        dataCensoredCopyDf['remainingExpected'] = remainingExpectedList
        dataCensoredCopyDf['remainingMedian'] = remainingMedianList
        dataCensoredCopyDf['remainingConditional'] = remainingConditionalList
        
        # Common metrics
        # timeMedianTrain = modelAFT.predictSurvivalTime(xTrainScaledExt[features], what='median', 
        #     fixNan=None, p=0.95, replaceInf=True)
            
        # metrics = libSurvival.Metrics(xTrainScaledExt['duration'], 
        #     xTrainScaledExt['E'], xTrainScaledExt['duration'], xTrainScaledExt['E'], 
        #     modelAFT.predictSurvival(xTrainScaledExt, times=timeLineObserved), 
        #     modelAFT.predictHazard(xTrainScaledExt, times=timeLineObserved))
        # metrics.getCI(xTrainScaledExt['duration'], timeMedianTrain, xTrainScaledExt['E'])
        
        # MAE if simulated
        if 'remainingTrue' in dataCensoredCopyDf:            
            metrics.MAE_Expected = float(mean_absolute_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingExpected']))
            metrics.MAE_Median = float(mean_absolute_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingMedian']))
            metrics.MAE_Conditional = float(mean_absolute_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingConditional']))
            
            metrics.R2_Expected = float(r2_score(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingExpected']))
            metrics.R2_Median = float(r2_score(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingMedian']))
            metrics.R2_Conditional = float(r2_score(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingConditional']))

            metrics.MAPE_Expected = float(mean_absolute_percentage_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingExpected']))
            metrics.MAPE_Median = float(mean_absolute_percentage_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingMedian']))
            metrics.MAPE_Conditional = float(mean_absolute_percentage_error(dataCensoredCopyDf['remainingTrue'], dataCensoredCopyDf['remainingConditional']))

            metrics.RemainingExpectedMedian = int(dataCensoredCopyDf['remainingExpected'].median())
            metrics.RemainingMedianMedian = int(dataCensoredCopyDf['remainingMedian'].median())

            print(sModel)
            print(metrics)


















# #   Save coefficients to excel
#     with pd.ExcelWriter(os.path.join(resultsSubDir, 'Coefficients.xlsx')) as writer:
#         for aftModelName, coef in coefDict.items():
#             coef.to_excel(writer, sheet_name='{}'.format(aftModelName))

    # from States import loadCDNOW

    # transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(sData)), 
        # mergeTransactionsPeriod=7, minFrequency=3)
        
    # client = '2104'
    # t = transactionsDf[transactionsDf['iId'] == client]
    
    
    # censored2Df = censoredDf[censoredDf['durationTrue'] > censoredDf['duration']].copy(deep=True)
        
    # censoredError2Df = censored2Df[['user', 'duration', 'durationTrue']].copy()
    # # columnsError = []
    # for column in columnsPredict:
    #     columnError = '{}'.format(column.replace('dM_', ''))
    #     censoredError2Df[columnError] = (censored2Df[column] - censored2Df['durationTrue']).abs()
    #     # columnsError.append(columnError)    
    
    
    # censoredError2Df.mean()
    
#         dataOut = []
#         for column in columnsError:
#             df2 = censoredDf[column]
#             df2.dropna(how='any', inplace=True)                    
#             dataOut.append(df2.values)
#             plt.boxplot(dataOut, vert = 1)
        
        
#         # plt.boxplot(censoredDf[columnError], vert = 1)




        
        
            
        
        


        
###############################################################################
        
        
        
            
#     a = libSurvival.getMedianSurvivalTime(S, p=0.5, replaceInf=True)
    
#     censored2Df = pd.DataFrame(columns=['user', 'dM_COX_TV'])
#     censored2Df['user'] = clientsPredict
#     censored2Df['dM_COX_TV'] = a
    
#     censored2Df = censored2Df.merge(censoredDf, how='inner', on='user')


#     mae = mean_absolute_error(censored2Df['durationTrue'].values, censored2Df['dM_COX_TV'].values)
#     maeDict['COX_TV'] = mae    




###############################################################################


# trees

    # xTrain, yTrain = get_x_y(xScaled, attr_labels=['E', 'duration'], pos_label=1, survival=True)
    # xPredictTree, _ = get_x_y(xPredict, attr_labels=['E', 'duration'], pos_label=1, survival=True)

    # max_features='auto'
    # n_estimators = 500
    # max_depth = 5
    # min_samples_split = 2
    # min_samples_leaf = 1
    # n_jobs = 6
    # maeList = []
    # binScoreList = []
    # models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
    #     'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    
    # for modelName in tqdm(models):
    #     # modelName = models[1]
    #     model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
    #         max_depth=max_depth, min_samples_split=min_samples_split, 
    #         min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
    #         n_jobs=n_jobs, random_state=random_state) 
        
    #     model.fit(xTrain, yTrain)
    #     timeMedian, _ = predictSurvivalTime(model, xPredictTree, list(xPredict['duration'].values), probability=0.5)
    #     censoredDf['dM_{}'.format(modelName)] = timeMedian
        
    #     mae = mean_absolute_error(censoredDf['durationTrue'].values, censoredDf['dM_{}'.format(modelName)].values)
    #     maeDict[modelName] = mae





    # yHatFiles = lib3.findFiles(dataSubDir, 'yHat.dat', exact=False)
    # if len(yHatFiles) == 0:
    #     raise RuntimeError('GRU first')
        
    # # clients and states to fit / predict
    # # last clients state only if censored or death state if dead        
    # yHat = lib2.loadObject(yHatFiles[0])
    # yHat['iD'] = yHat['user'] + ' | ' + yHat['date'].astype(str)
    # yHat.set_index('iD', drop=True, inplace=True)
    # clientsAllList = list(yHat['user'].unique())
    # dStateList = states.getExistingStates()
    # nStates = len(dStateList)
    # dStateReverseList = sorted(dStateList, reverse=True)
    # stateLast = states.loadState(dStateList[-1], error='raise')


    # columnsSpecial = ['client', 'date', 'duration', 'E', 'yHat']
    


    

            
#     yPredictDf = getLastState(stateLast)

#     columnsFrame = columnsSpecial + features
        
#     # make frameList and fill with nan
#     frame = FrameList(rowNames=list(yPredictDf.index), columnNames=columnsFrame, df=None, dtype=None)
#     for column in frame.columnNames:
#         frame.fillColumn(column, np.nan)


# #   Fill FrameArray with data from states
# #   loop on states
#     for i in tqdm(range(1, len(dStateList), 1)):
#         dState = dStateList[i]
#         yPredictDf1 = yPredictDf[yPredictDf['date'] == dState]
#         if len(yPredictDf1) == 0:
#             continue    
#         # load state
#         state = states.loadState(dState, error='raise')
#         # loop on users in state
#         for i, row in yPredictDf1.iterrows():
#             # break
#             client = row['user']
#             clientObj = state.Clients.get(client)
#             if client is None:
#                 raise ValueError('State: {} has no client: {}'.format(dState, client)) 
#             iD = '{} | {}'.format(client, dState)
#             status = row['status']
#             if status == 0:
#                 p = 0
#             elif status in [1, 2]:
#                 p = 1 # probability of being dead
#             else:
#                 # get probability for alive from predictions
#             # data from yHat to frame
#                 ySeries = yHat.loc[iD]
#                 p = ySeries['yHat']
                
#             frame.put(iD, 'yHat', p)
#             frame.put(iD, 'client', client)
#             frame.put(iD, 'date', row['date'])

#             # loop on features for user in state
#             for column in features:               
#                 value = clientObj.descriptors.get(column)
#                 if value is None:
#                     raise ValueError('Client: {} has no feature: {}'.format(client, column))            
#                 frame.put(iD, column, value)
                    
# #   prepare data for AFT
# #   frame to pandas
#     df = frame.to_pandas()
#     df['E'] = df['status'].map(states.mapStatusToResponse)
#     df['duration'] = df['D_TS']
#     # make duratin 1 for clients with duration zero
#     df['duration'] = np.where(df['duration'] == 0, 1, df['duration']) 
#     df = df[df['duration'] > 0]
#     df = df[columnsFrame] # arange columns
#     del(df['D_TS'])
#     df.sort_index(inplace=True)
#     features.remove('D_TS')
#     features.remove('status')
    

        
    # #   Scale features    
    # scaler = StandardScaler()    
    # xScaled = scaler.fit_transform(df[features])
    # xScaled = pd.DataFrame(xScaled, index=df.index, columns=features)
    # xScaled['duration'] = df['duration']
    # xScaled['E'] = df['E']    
    
# fit survival; one observation - one client
#     fitResultsDict = {} # AIC, CI for each model 
#     coefDict = {}
#     ibsDict = {}    
#     aucTS_Dict = {}
#     aucDict = {}
#     bsDict = {}
    
# #   AFT
#     smoothing_penalizer = 0
#     penalizer = 0.1
#     ancillary = True
#     fit_left_censoring = True
#     timeLine = list(range(0, nStates*5, 1))
#     timeLineObserved = np.arange(0, nStates)
    
#     predictionsDf = df[columnsSpecial + ['status']].copy(deep=True)    
#     modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter']#, 'GeneralizedGammaRegressionFitter', 'AalenAdditiveFitter']
#     sY_List = [] # names for predicting columns
#     for aftModelName in tqdm(modelsAFT):
#     # aftModelName = modelsAFT[1]
    
#         fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
#             penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
            
#         fitterAFT = fitAFT(fitterAFT, xScaled, duration_col='duration', event_col='E', 
#             ancillary=ancillary, fit_left_censoring=fit_left_censoring, timeline=timeLine)
        
#         fitResults = predictAFT(fitterAFT, xScaled)
#         fitResults = fixFloat(fitResults)
#         fitResults['yExpected'].name = 'yExpected {}'.format(aftModelName)
#         fitResults['yMedian'].name = 'yMedian {}'.format(aftModelName)
#         predictionsDf = predictionsDf.join(fitResults['yExpected'], how='left')
#         predictionsDf = predictionsDf.join(fitResults['yMedian'], how='left')
#         sY_List.append('yExpected {}'.format(aftModelName))
#         sY_List.append('yMedian {}'.format(aftModelName))
#         del(fitResults['yExpected'])
#         del(fitResults['yMedian'])
#         coefDict[aftModelName] = fitterAFT.summary

#         fitResultsDict[aftModelName] = fitResults                        

# #   evaluate survival curve on observed timeline for all states
#         s = fitterAFT.predict_survival_function(xScaled, times=timeLineObserved)
#         h = fitterAFT.predict_cumulative_hazard(xScaled, times=timeLineObserved)                    
                       
#         survival_train = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
#         survival_test = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
#         times = sorted(list(xScaled['duration'].unique()))
#         _ = times.pop(-1)
#         estimate = s[s.index.isin(times)]
#         estimate = estimate.values.T
#         estimateH = h[h.index.isin(times)]
#         estimateH = estimateH.values.T  
        
# #   Integrated Brier score   
#         ibs = integrated_brier_score(survival_train=survival_train, survival_test=survival_test,
#                                estimate=estimate, times=times)        
#         ibsDict[aftModelName] = ibs
        
# # dynamic AUC for right censored data:
#         aucTS, aucMean = cumulative_dynamic_auc(survival_train=survival_train, 
#             survival_test=survival_test, estimate=estimateH, times=times)
#         aucTS_Dict[aftModelName] = aucTS
#         aucDict[aftModelName] = aucMean

# #   Brier score for plot
#         bs = brier_score(survival_train=survival_train, survival_test=survival_test,
#                                estimate=estimate, times=times) 
#         bsDict[aftModelName] = pd.Series(data=bs[1], index=bs[0])
                                        
#     # plot coefficiants
#         fig, ax = plt.subplots(1, 1, figsize=(19,15))
#         fitterAFT.plot(columns=None, parameter=None, ax=ax)
#         title = 'Coefficients {}'.format(aftModelName)
#         fig.suptitle(title, fontsize=16)
#         fileName = '{}{}'.format(title, '.png')
#         plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
#         plt.close()             
    
# ###############################################################################
# # trees

#     event_col = 'E'
#     duration_col = 'duration'
#     xTrain, yTrain = get_x_y(xScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
#     xPredictTree, _ = get_x_y(xPredict, attr_labels=[event_col, duration_col], pos_label=1, survival=True)

#     # maxLen = max(mapClientToTS.values()) + 1
#     max_features='auto'
#     n_estimators = 500
#     max_depth = 5
#     min_samples_split = 2
#     min_samples_leaf = 1
#     n_jobs = 6
#     maeList = []
#     binScoreList = []
#     models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
#         'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    
#     for modelName in tqdm(models):
#         # modelName = models[0]
#         model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
#             max_depth=max_depth, min_samples_split=min_samples_split, 
#             min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
#             n_jobs=n_jobs, random_state=random_state) 
        
#         model.fit(xTrain, yTrain)
#         timeMedian, _ = predictSurvivalTime(model, xPredictTree, list(xPredict[duration_col].values), probability=0.5)
#         censoredDf['dM_{}'.format(modelName)] = timeMedian
        
        
        
        
# ###############################################################################

# # Metrics

# #   CI for prediction obtained by folds
#     ciDict = {}
#     for column in sY_List: 
#         ci = concordance_index(df['duration'].values, predictionsDf[column].values, df['E']) # 1 good
#         ciDict[column] = ci

# # plot 4 diagrams on one pic
#     plotBrierScore(bsDict, ibsDict)
#     plot_cumulative_dynamic_auc(times, aucTS_Dict, aucDict)

# #   Save coefficients to excel
#     with pd.ExcelWriter(os.path.join(resultsSubDir, 'Coefficients.xlsx')) as writer:
#         for aftModelName, coef in coefDict.items():
#             coef.to_excel(writer, sheet_name='{}'.format(aftModelName))
                
        
#     with pd.ExcelWriter(os.path.join(resultsSubDir, 'LifeTime.xlsx')) as writer:
#         predictionsDf.to_excel(writer)
        
        
#     yamlDict = {'Features': features, 'Concordance index': fixFloat(ciDict), 'Integrated Brier score': fixFloat(ibsDict),
#         'Dynamic AUC': fixFloat(aucDict), 'fitResultsFold': fixFloat(fitResultsDict)}

#     lib2.saveYaml(os.path.join(resultsSubDir, 'Metrics.yaml'), yamlDict, sort_keys=False)



#     if dEndTrain is not None:
#         transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(sData)), 
#             mergeTransactionsPeriod=TS_STEP, minFrequency=MIN_FREQUENCY)
        
#         mapClientToBirth = {}
#         mapClientToBirthState = {}
#         for client, clientObj in stateLast.Clients.items():
#             status = clientObj.descriptors['status']
#             if status in [0, 4]:
#                 # censored            
#                 mapClientToBirth[client] = clientObj.descriptors['dThirdEvent']
#                 mapClientToBirthState[client] = clientObj.descriptors['dBirthState']
    
#         mapClientToFirstBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
#             leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
#             eventType='first')
     
#         mapClientToLastBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
#             leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
#             eventType='last')
        
#         mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
#             leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
#             minFrequency=2)
    
#         mapClientToMeanPeriodTS = {x : np.mean(y) / TS_STEP for x, y in mapClientToPeriods.items()}
        
#         # from third buy to last buy + mean interevent time
#         mapClientToDurationTS = {}
#         for client, dBirth in mapClientToBirth.items():
#             mapClientToDurationTS[client] = float((mapClientToLastBuy[client] - mapClientToBirth[client]).days / TS_STEP + mapClientToMeanPeriodTS[client])
    
#         # only just died
#         deadDf = predictionsDf[predictionsDf['status'] == 2].copy(deep=True)
#         deadDf['dBirth'] = deadDf['client'].map(mapClientToFirstBuy)
#         deadDf['meanPeriod'] = deadDf['client'].map(mapClientToMeanPeriodTS)
#         deadDf['lifetime'] = (deadDf['date'] - deadDf['dBirth']).dt.days
#         deadDf['lifetimeTS'] = deadDf['lifetime'].add(deadDf['meanPeriod']).div(TS_STEP)
#         meanDead = deadDf['lifetimeTS'].mean()
        
#         # only censored
#         censoredDf = predictionsDf[predictionsDf['status'].isin([0, 4])].copy(deep=True)
#         censoredDf['D_TS_True'] = censoredDf['client'].map(mapClientToDurationTS)
#         censoredDf['Remaining_TS_True'] = censoredDf['D_TS_True'] - censoredDf['duration']
                
#         # MAE for censored clients only
#         for column in sY_List:
#             # fit predictions to distribution

#             stats1 = Statistics(censoredDf[column], testStats='ks', criterion='pvalue')
#             # restrict outliers by 95 quantile, few crazy numbers crush all metrics
#             maxLifetime = stats1.bestModel.ppf(0.95)
#             censoredDf[column] = np.where(censoredDf[column] > maxLifetime, maxLifetime, censoredDf[column])
#             # MAE for duration
#             mae = mean_absolute_error(censoredDf['D_TS_True'].values, censoredDf[column].values)
#             yamlDict['MAE {}'.format(column)] = float(mae)
                
#         with pd.ExcelWriter(os.path.join(resultsSubDir, 'LifeTimeCensored.xlsx')) as writer:
#             censoredDf.to_excel(writer)    
        
#         columnsRemaining = []
#         columnsError = []
#         for column in sY_List:
#             newColumn = 'remaining {}'.format(column)
#             newColumn2 = 'error {}'.format(column)
            
#             columnsRemaining.append(newColumn)
#             censoredDf[newColumn] = censoredDf[column] - censoredDf['duration']
            
#             censoredDf[newColumn2] = censoredDf[newColumn] - censoredDf['Remaining_TS_True']
#             censoredDf[newColumn2] = censoredDf[newColumn2].abs()
#             columnsError.append(newColumn2)
            
#             # MAE for remaining time
#             # mae = mean_absolute_error(censoredDf['Remaining_TS_True'].values, censoredDf[newColumn].values)
#             # yamlDict['MAE {}'.format(newColumn)] = float(mae)

#         mean = censoredDf[columnsRemaining].mean()
#         median = censoredDf[columnsRemaining].median()

#         yamlDict['Mean remaining predicted time'] = mean.to_dict()
#         yamlDict['Median remaining predicted time'] = median.to_dict()
        
#         yamlDict['True Mean remaining time'] = censoredDf['Remaining_TS_True'].mean()
#         yamlDict['True Median remaining time'] = censoredDf['Remaining_TS_True'].median()

#         lib2.saveYaml(os.path.join(resultsSubDir, 'Metrics.yaml'), yamlDict, sort_keys=False)
        

#         from matplotlib import pyplot as plt
#         columnRemaining = columnsRemaining[1]
#         columnError = columnsError[-1]
#         plt.scatter(censoredDf['Remaining_TS_True'], censoredDf[columnRemaining])
        
        

        
#         dataOut = []
#         for column in columnsError:
#             df2 = censoredDf[column]
#             df2.dropna(how='any', inplace=True)                    
#             dataOut.append(df2.values)
#             plt.boxplot(dataOut, vert = 1)
        
        
#         # plt.boxplot(censoredDf[columnError], vert = 1)





        
        
#     maeTree = {}
#     for column in models:
#         mae = mean_absolute_error(censoredDf['D_TS_True'].values, censoredDf[column].values)
#         maeTree[column] = mae



#     s = model.predict_survival_function(xTrain) # not for all
#     h = model.predict_cumulative_hazard_function(xTrain) # not for all








    
 
"""    