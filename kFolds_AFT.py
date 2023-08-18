# survival refgression 
# takes feature from kFolds_binary
# plots




import os
# import libStent
# from datetime import datetime
import pandas as pd
# import libPlot
import numpy as np
# import lib3
from datetime import timedelta
import lib2
# from time import time
# import warnings
# from math import log
# from math import exp
# from Rank import Rank
# from datetime import date
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import datetime
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
# from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import cumulative_dynamic_auc
# from random import sample
# import libChurn2
# from matplotlib import pyplot as plt
# from scipy.stats import norm
# from Encoder import DummyEncoderRobust
# from sklearn.linear_model import LogisticRegression
# import lib4
# from xgboost import XGBClassifier
# from lifelines import CoxTimeVaryingFitter
# from sklearn.preprocessing import minmax_scale
# from pandas import Series
# from StentBase1 import StentBase1
# from sklearn.ensemble import RandomForestClassifier
from FrameList import FrameList
# from FrameArray import FrameArray
# from State import State
# from State import TS_STEP
# import numpy
# from sksurv.datasets import load_gbsg2
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.metrics import integrated_brier_score
# from sksurv.preprocessing import OneHotEncoder
# from sksurv.metrics import concordance_index_censored
# from lifelines.utils import concordance_index
# from copy import deepcopy
# from datetime import datetime
# from pathlib import Path
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import GRU
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense
# from libRecommender import METRICS_BINARY
# from tensorflow.keras.callbacks import EarlyStopping
# from collections.abc import Iterable
# from random import choice
# from random import sample
import lifelines
import libChurn2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
# from sksurv.datasets import get_x_y
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
import lib3
# import libChurn2
# from pycox.evaluation import EvalSurv
from State import TS_STEP

def fitAFT(fitterAFT, data, duration_col, event_col, 
    timeline=None, ancillary=True, fit_left_censoring=False):
    
    show_progress = False
        
    if isinstance(fitterAFT, lifelines.AalenAdditiveFitter) or \
        isinstance(fitterAFT, lifelines.fitters.coxph_fitter.CoxPHFitter):
        fitterAFT.fit(data, duration_col=duration_col, 
            event_col=event_col, show_progress=show_progress, timeline=timeline)
    elif isinstance(fitterAFT, lifelines.fitters.generalized_gamma_regression_fitter.GeneralizedGammaRegressionFitter):        
        if fit_left_censoring:
            fitterAFT.fit_left_censoring(data, duration_col=duration_col, 
                event_col=event_col, show_progress=show_progress, timeline=timeline)
        else:
            fitterAFT.fit(data, duration_col=duration_col, 
                event_col=event_col, show_progress=show_progress, timeline=timeline)
    else:
        if fit_left_censoring:
            fitterAFT.fit_left_censoring(data, duration_col=duration_col, 
                event_col=event_col, ancillary=ancillary, show_progress=show_progress, timeline=timeline)
        else:
            fitterAFT.fit(data, duration_col=duration_col, 
                event_col=event_col, ancillary=ancillary, show_progress=show_progress, timeline=timeline)  

    return fitterAFT

def predictAFT(fitterAFT, data):
    
    if isinstance(fitterAFT, lifelines.fitters.coxph_fitter.CoxPHFitter):
        aic = fitterAFT.AIC_partial_
    elif isinstance(fitterAFT, lifelines.AalenAdditiveFitter):
        aic = np.nan
    else:
        aic = fitterAFT.AIC_                        
    
    y_pred_exp = fitterAFT.predict_expectation(data)
    y_pred_exp.fillna(y_pred_exp.mean(), inplace=True)
                    
    y_pred_median = fitterAFT.predict_median(data)
    y_pred_median.fillna(y_pred_median.mean(), inplace=True)
    try:
        # replace with highest value except inf
        y_pred_median.replace([np.inf], np.nanmax(y_pred_median.values[y_pred_median.values != np.inf]), inplace=True)
        # replace with lowest value except -inf
        y_pred_median.replace([-np.inf], np.nanmin(y_pred_median.values[y_pred_median.values != -np.inf]), inplace=True)
    except:
        pass
    return {'AIC': aic, 'CI': fitterAFT.concordance_index_, 
            'yExpected': y_pred_exp, 'yMedian': y_pred_median}

def makeModelSksurv(modelName, n_estimators, max_depth, min_samples_split, 
                    min_samples_leaf, max_features, verbose, n_jobs, random_state):
    
    if modelName == 'RandomSurvivalForest':
        return RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=0.0, max_features=max_features, max_leaf_nodes=None, 
            bootstrap=True, oob_score=False, n_jobs=n_jobs, random_state=random_state, 
            verbose=verbose, warm_start=False, max_samples=None)

    if modelName == 'ExtraSurvivalTrees':
        return ExtraSurvivalTrees(n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=0.0, max_features=max_features, max_leaf_nodes=None, 
            bootstrap=True, oob_score=False, n_jobs=n_jobs, random_state=random_state, 
            verbose=verbose, warm_start=False, max_samples=None)

    if modelName == 'GradientBoostingSurvivalAnalysis':
        if max_depth is None:
            max_depth = 100
        return GradientBoostingSurvivalAnalysis(loss='coxph', learning_rate=0.1, 
            n_estimators=n_estimators, criterion='friedman_mse', 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=0.0, max_depth=max_depth, min_impurity_split=None, 
            min_impurity_decrease=0.0, random_state=random_state, max_features=max_features,
            max_leaf_nodes=None, subsample=1.0, dropout_rate=0.0, verbose=verbose, ccp_alpha=0.0)

    if modelName == 'ComponentwiseGradientBoostingSurvivalAnalysis':
        return ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', 
            learning_rate=0.05, n_estimators=n_estimators, subsample=1.0, 
            dropout_rate=0, random_state=random_state, verbose=verbose)
  
    return None

def predictSurvivalTime(estimator, xTest, T_List=None, probability=0.5):
    s = estimator.predict_survival_function(xTest)
    lifetimeList = []
    pAliveList = []
    for i in range(0, s.shape[0], 1):
        s1 = s[i]
        timeline1 = s1.x
        pAlive = s1.y
        idx = (np.abs(pAlive - probability)).argmin()
        lifetimeList.append(int(timeline1[idx]))
        if T_List is not None:             
            T = T_List[i]
            idx2 = findNearest(timeline1, T)
            pAliveList.append(pAlive[idx2])
    return lifetimeList, pAliveList

def findNearest(array, value):
    idx = lib2.argmin([abs(x - value) for x in array])
    return idx

#   make list of id for last state
#   dead        
def getIdDeadAll():
    idJustDied = []
    for dDate, state in statesDict.items():
        for client, clientObj in state.Clients.items():
            if clientObj.descriptors.get('status') == 2:
                # just died
                iD = '{} | {}'.format(client, dDate)
                idJustDied.append(iD)
    return idJustDied

#   dead last time    
def getIdDeadLast(lag=0):
    idLastDied = []
    for client in clientsAllList:
        for dDate in dStateReverseList:
            clientObj = statesDict[dDate].Clients.get(client)
            if clientObj is None:
                continue
            state = clientObj.descriptors['status']
            if state == 2:
                dState = dDate + timedelta(days=int(lag*TS_STEP))
                dState = min(dState, dStateReverseList[0]) # do not go beyond                     
                iD = '{} | {}'.format(client, dState)
                idLastDied.append(iD)                
                break
    return idLastDied
        
def getIdCensored():
#   censored at last state
    idCensored = []
    for client, clientObj in stateLast.Clients.items():
        if clientObj.descriptors.get('status') == 4:
            iD = '{} | {}'.format(client, dStateLast)
            idCensored.append(iD)
    return idCensored

def makeRows():
#   make rows for frameList
    rows = []
#   All clients, all states
    for client in clientsAllList:
        for dDate in dStateList:       
            #!!! do not remember why was from 1:
        # for dDate in dStateList[1:]:
            iD = '{} | {}'.format(client, dDate)
            rows.append(iD)
    return rows

def makeStructuredArray(durations, events):
    y = []
    for duration, event in zip(durations, events):
        y.append((bool(event), float(duration)))
    return  np.array(y, dtype=[('cens', '?'), ('time', '<f8')])

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

###############################################################################    
if __name__ == "__main__":

    # TEST_FRAC = 0.5
    random_state = 101
    nFolds = 3
    lag = 0
    
    sData = 'CDNOW' # CDNOW casa_2 plex_45 Retail Simulated1

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)    
    lib2.checkDir(resultsSubDir)
    
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    nStates = len(statesDict)
    transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])
    
    dStateList = sorted(list(statesDict.keys()))
    dStateLast = dStateList[-1]
    stateZero = statesDict[dStateList[0]]
    stateLast = statesDict[dStateLast]

    dStateReverseList = sorted(dStateList, reverse=True)
    # clientsAllList = stateLast.clientsAll

    yHatFiles = lib3.findFiles(dataSubDir, '* yHat.dat', exact=False)
    # if len(yHatFiles) == 0:
    #     raise RuntimeError('GRU first')
    # yHat = lib2.loadObject(yHatFiles[0])

    yHatList = []
    for file in yHatFiles:
        yHat1 = lib2.loadObject(file)
        yHatList.append(yHat1)
    yHat = pd.concat(yHatList, axis=0)
    yHat.sort_values(['user', 'date'], inplace=True)
    yHat.reset_index(drop=True, inplace=True)
    del(yHat1)
    del(yHatList)

    clientsAllList = list(yHat['user'].unique())
    foldsDict = libChurn2.splitList(nFolds, clientsAllList)

    columnsSpecial = ['client', 'date', 'duration', 'E', 'yHat']
    features = ['D_TS', 'status', 'frequency', 
        'C_Orig', 'moneySum', 'moneyMedian', 'moneyDailyStep',
        'r10_F', 'r10_C', 'r10_M', 'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC',
        'trend_r10_MC', 'trend_short_r10_MC', 'trend_r10_FM', 'trend_short_r10_FM',
        'trend_r10_FC', 'trend_short_r10_FC', 'trend_r10_FMC', 'trend_short_r10_FMC']
    
    rows = makeRows() # all states, all clients
    idCensored = getIdCensored() # censored
    idDeadLast = getIdDeadLast(lag=lag) # dead last time
    rows = idDeadLast + idCensored
    rows = sorted(rows)
    
    # make frameList and fill with nan
    columnsFrame = columnsSpecial + features
    frame = FrameList(rowNames=rows, columnNames=columnsFrame, df=None, dtype=None)
    for column in frame.columnNames:
        frame.fillColumn(column, np.nan)
        
#   make rich data for survival regression (lifelines)
#   one client can have many observations
    # for client in tqdm(clientsAllList):
    #     for dDate in dStateList[1:]: # skip zero state
    #         clientObj = statesDict[dDate].Clients.get(client)
    #         if clientObj is None:
    #             continue
    #         row = '{} | {}'.format(client, dDate)
    #         frame.put(row, 'client', client)
    #         frame.put(row, 'date', dDate)            
    #         for column in features:
    #             value = clientObj.descriptors.get(column)
    #             if value is None:
    #                 raise ValueError('Client: {} has no feature: {}'.format(client, column))            
    #             frame.put(row, column, value)


    for row in rows:
        client, sDate = row.split(' | ')    
        dDate = datetime.strptime(sDate, '%Y-%m-%d').date()    
        clientObj = statesDict[dDate].Clients.get(client)
        if clientObj is None:
            continue
        frame.put(row, 'client', client)
        frame.put(row, 'date', dDate)            
        for column in features:
            value = clientObj.descriptors.get(column)
            if value is None:
                raise ValueError('Client: {} has no feature: {}'.format(client, column))            
            frame.put(row, column, value)
                
            
# put predictions from ANN
    rowsSet = set(frame.rowNames)
    for i, row in tqdm(yHat.iterrows()):
        iD = '{} | {}'.format(row['user'], row['date'])
        if iD in rowsSet:
            frame.put(iD, 'yHat', row['yHat'])

#   frame to pandas
    df = frame.to_pandas()
    mapStatusToResponse = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}
    df['E'] = df['status'].map(mapStatusToResponse)
    df['duration'] = df['D_TS']
    # make duratin 1 for clients with duration zero
    df['duration'] = np.where(df['duration'] == 0, 1, df['duration']) 
    df = df[df['duration'] > 0]
    df = df[columnsFrame]
    del(df['D_TS'])
    df.sort_index(inplace=True)
    features.remove('D_TS')
    features.remove('status')
    features.append('yHat')
    




# fit survival; one observation - one client
    predictionsList = []
    fitResultsFoldDict = {} # AIC, CI for each model for each fold
    # coefficientsFoldList = []
    # # ibsDict = {}
    # ibsLastDict = {}
    sDict = {} # survival matrix
    hDict = {} # cumulative hazard matrix
    # s1Dict = {}
    # xTestScaledDict = {}
    # xTestScaledLastDict = {}
    # coefficientsDict = {}
    
    
    for fold, fold1_Dict in foldsDict.items():
        # break
        print('Fold: {} out of {}'.format(fold+1, nFolds))
    #   split train test    
        trainDf = df[df['client'].isin(fold1_Dict['clientsTrain'])].copy(deep=True)
        testDf = df[df['client'].isin(fold1_Dict['clientsTest'])].copy(deep=True)
        trainDf.sort_index(inplace=True)
        testDf.sort_index(inplace=True)            
    
    #   Scale features    
    #!!! tmp
        # features = ['yHat']
        scaler = StandardScaler()    
        xTrainScaled = scaler.fit_transform(trainDf[features])
        xTrainScaled = pd.DataFrame(xTrainScaled, index=trainDf.index, columns=features)
        xTrainScaled['duration'] = trainDf['duration']
        xTrainScaled['E'] = trainDf['E']
        
        xTestScaled = scaler.transform(testDf[features])
        xTestScaled = pd.DataFrame(xTestScaled, index=testDf.index, columns=features)
        xTestScaled['duration'] = testDf['duration']
        xTestScaled['E'] = testDf['E']

    #   AFT
        smoothing_penalizer = 0
        penalizer = 0.1
        ancillary = False    
        timeLine = list(range(0, nStates*5, 1))
        timeLineObserved = np.arange(0, nStates)
        predictionsDf = testDf[['client', 'date', 'duration', 'E', 'yHat', 'frequency']].copy(deep=True)    
        modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter'] #, 'GeneralizedGammaRegressionFitter']
        sY_List = [] # names for predicting columns
        for aftModelName in tqdm(modelsAFT):
        # aftModelName = modelsAFT[0]
        
            fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
                penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
                
            fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col='duration', event_col='E', 
                ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)
            
            fitResults = predictAFT(fitterAFT, xTestScaled)
            fitResults['yExpected'].name = 'yExpected {}'.format(aftModelName)
            fitResults['yMedian'].name = 'yMedian {}'.format(aftModelName)
            predictionsDf = predictionsDf.join(fitResults['yExpected'], how='left')
            predictionsDf = predictionsDf.join(fitResults['yMedian'], how='left')
            sY_List.append('yExpected {}'.format(aftModelName))
            sY_List.append('yMedian {}'.format(aftModelName))
            del(fitResults['yExpected'])
            del(fitResults['yMedian'])
                        
            oldValue = fitResultsFoldDict.get(aftModelName, [])
            oldValue.append(fitResults)
            fitResultsFoldDict[aftModelName] = oldValue                        

#   evaluate survival curve on observed timeline for all states
            s = fitterAFT.predict_survival_function(xTestScaled, times=timeLineObserved)
            h = fitterAFT.predict_cumulative_hazard(xTestScaled, times=timeLineObserved)            
            
            oldValue = sDict.get(aftModelName, [])
            oldValue.append(s)
            sDict[aftModelName] = oldValue

            oldValue = hDict.get(aftModelName, [])
            oldValue.append(h)
            hDict[aftModelName] = oldValue
            
            # oldValue = xTestScaledDict.get(aftModelName, [])
            # oldValue.append(xTestScaled)
            # xTestScaledDict[aftModelName] = oldValue            
                                            
        predictionsList.append(predictionsDf)
                
    predictionsDf = pd.concat(predictionsList, axis=0)
    predictionsDf.sort_index(inplace=True)
    
    
###########
#   fit everything to get coefficients, AIC, CI
    scaler = StandardScaler()    
    xScaled = scaler.fit_transform(df[features])
    xScaled = pd.DataFrame(xScaled, index=df.index, columns=features)
    xScaled['duration'] = df['duration']
    xScaled['E'] = df['E']    
    
    fitResultsDict = {}
    coefDict = {}
    for aftModelName in tqdm(modelsAFT):
    
        fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
            
        fitterAFT = fitAFT(fitterAFT, xScaled, duration_col='duration', event_col='E', 
            ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)
    
        fitResults = predictAFT(fitterAFT, xScaled)
        del(fitResults['yExpected'])
        del(fitResults['yMedian'])
        fitResultsDict[aftModelName] = fitResults        
        coefDict[aftModelName] = fitterAFT.summary
    
    # plot coefficiants
        fig, ax = plt.subplots(1, 1, figsize=(19,15))
        fitterAFT.plot(columns=None, parameter=None, ax=ax)
        title = 'Coefficients {}'.format(aftModelName)
        fig.suptitle(title, fontsize=16)
        fileName = '{}{}'.format(title, '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close()
    
# Metrics

#   CI for prediction obtained by folds
    ciDict = {}
    for column in sY_List: 
        ci = concordance_index(df['duration'].values, predictionsDf[column].values, df['E']) # 1 good
        ciDict[column] = ci

#   IBS for prediction obtained by folds
    # ibsDict = {}
    ibs2Dict = {}    
    aucTS_Dict = {}
    aucDict = {}
    bsDict = {}
    for model in modelsAFT:
        # model = modelsAFT[0]
        sList = sDict[model]        
        s = pd.concat(sList, axis=1)
        columns = sorted(list(s.columns)) # sort columns
        s = s[columns]  
        hList = hDict[model]
        h = pd.concat(hList, axis=1)
        columns = sorted(list(h.columns)) # sort columns
        h = h[columns]
                             
        # evalSurv = EvalSurv(s, xScaled['duration'].values, xScaled['E'].values, censor_surv='km')
        # ibs = evalSurv.integrated_brier_score(timeLineObserved)                    
        # ibsDict[model] = ibs
        # bsDict[model] = evalSurv.brier_score(timeLineObserved)

        survival_train = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
        survival_test = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
        times = sorted(list(xScaled['duration'].unique()))
        _ = times.pop(-1)
        estimate = s[s.index.isin(times)]
        estimate = estimate.values.T
        estimateH = h[h.index.isin(times)]
        estimateH = estimateH.values.T  
        
        ibs2 = integrated_brier_score(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimate, times=times)        
        ibs2Dict[model] = ibs2

# dynamic AUC for right censored data:
        aucTS, aucMean = cumulative_dynamic_auc(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimateH, times=times)
        aucTS_Dict[model] = aucTS
        aucDict[model] = aucMean


        bs = brier_score(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimate, times=times) 
        bsDict[model] = pd.Series(data=bs[1], index=bs[0])
            


# plot 4 diagrams on one pic
    plotBrierScore(bsDict, ibs2Dict)
    plot_cumulative_dynamic_auc(times, aucTS_Dict, aucDict)

#   Save coefficients to excel
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'Coefficients.xlsx')) as writer:
        for aftModelName, coef in coefDict.items():
            coef.to_excel(writer, sheet_name='{}'.format(aftModelName))
        
    
    for aftModelName, d in fitResultsDict.items():
        key = 'yMedian {}'.format(aftModelName)
        print('{} -- AIC: {}, CI: {} '.format(aftModelName, round(d['AIC'], 2), round(ciDict[key], 6)))
        
        
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'LifeTime.xlsx')) as writer:
        predictionsDf.to_excel(writer)
        
    print(ibs2Dict)

# CI calculated
# WeibullAFT -- AIC: 28926.2, CI: 0.908667 
# LogNormalAFT -- AIC: 28587.78, CI: 0.912267 
# LogLogisticAFT -- AIC: 28628.18, CI: 0.918628 
# CoxPHFitter -- AIC: 88457.66, CI: 0.859428
# COX has partial AIC

# no yHat
# WeibullAFT -- AIC: 31176.86, CI: 0.860702 
# LogNormalAFT -- AIC: 31096.62, CI: 0.874384 
# LogLogisticAFT -- AIC: 30670.93, CI: 0.889036 
# CoxPHFitter -- AIC: 82565.38, CI: 0.839903

# Simulated1
# WeibullAFT -- AIC: 57249.44, CI: 0.939302 
# LogNormalAFT -- AIC: 56615.05, CI: 0.940493 
# LogLogisticAFT -- AIC: 57722.8, CI: 0.945837 
# CoxPHFitter -- AIC: 228048.85, CI: 0.853044 
# {'WeibullAFT': 0.035529728468989795, 'LogNormalAFT': 0.03429841803032229, 'LogLogisticAFT': 0.03300671987817704, 'CoxPHFitter': 0.08625531769250679}


        
#     # lastStateIdList = idLastDied + idCensored
    
# # fit survival    
#     predictionsFoldList = []
#     fitResultsFoldList = []
#     coefficientsFoldList = []
#     # ibsDict = {}
#     ibsLastDict = {}
#     sDict = {}
#     s1Dict = {}
#     xTestScaledDict = {}
#     xTestScaledLastDict = {}
    
#     for fold, fold1_Dict in foldsDict.items():
#         print('Fold: {} out of {}'.format(fold+1, nFolds))
#     #   split train test    
#         trainDf = df[df['client'].isin(fold1_Dict['clientsTrain'])].copy(deep=True)
#         testDf = df[df['client'].isin(fold1_Dict['clientsTest'])].copy(deep=True)
#         trainDf.sort_index(inplace=True)
#         testDf.sort_index(inplace=True)            
    
#     #   Scale features    
#     #!!! tmp
#         # features = ['yHat']
#         scaler = StandardScaler()    
#         xTrainScaled = scaler.fit_transform(trainDf[features])
#         xTrainScaled = pd.DataFrame(xTrainScaled, index=trainDf.index, columns=features)
#         xTrainScaled['duration'] = trainDf['duration']
#         xTrainScaled['E'] = trainDf['E']
        
#         xTestScaled = scaler.transform(testDf[features])
#         xTestScaled = pd.DataFrame(xTestScaled, index=testDf.index, columns=features)
#         xTestScaled['duration'] = testDf['duration']
#         xTestScaled['E'] = testDf['E']

#     #   AFT
#         smoothing_penalizer = 0
#         penalizer = 0.1
#         ancillary = False    
#         timeLine = list(range(0, nStates*5, 1))
#         timeLineObserved = np.arange(0, nStates)
#         summaryDf = testDf[['client', 'date', 'duration', 'E', 'yHat', 'frequency']].copy(deep=True)    
#         modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter'] #, 'GeneralizedGammaRegressionFitter']
#         fitResults = {}
#         sY_List = []
#         coefficientsDict = {}
#         for aftModelName in tqdm(modelsAFT):
#         # aftModelName = modelsAFT[4]
        
#             fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
#                 penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
                
#             fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col='duration', event_col='E', 
#                 ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)
            
#             fitResults[aftModelName] = predictAFT(fitterAFT, xTestScaled)
                                
#             fitResults[aftModelName]['yExpected'].name = 'yExpected {}'.format(aftModelName)
#             fitResults[aftModelName]['yMedian'].name = 'yMedian {}'.format(aftModelName)        
#             summaryDf = summaryDf.join(fitResults[aftModelName]['yExpected'], how='left')
#             summaryDf = summaryDf.join(fitResults[aftModelName]['yMedian'], how='left')
#             sY_List.append('yExpected {}'.format(aftModelName))
#             sY_List.append('yMedian {}'.format(aftModelName))
#             coefficientsDict[aftModelName] = fitterAFT.summary

# #   evaluate survival curve on observed timeline for all states
#             s = fitterAFT.predict_survival_function(xTestScaled, times=timeLineObserved)
            
#             oldValue = sDict.get(aftModelName, [])
#             oldValue.append(s)
#             sDict[aftModelName] = oldValue

#             oldValue = xTestScaledDict.get(aftModelName, [])
#             oldValue.append(xTestScaled)
#             xTestScaledDict[aftModelName] = oldValue            
                                    
#             # IBS for last state
#             lastStateIdFold = list(set(lastStateIdList).intersection(set(xTestScaled.index)))            
#             xTestScaledLast = xTestScaled.loc[lastStateIdFold].copy(deep=True)
#             s1 = fitterAFT.predict_survival_function(xTestScaledLast, times=timeLineObserved)
            
#             oldValue = s1Dict.get(aftModelName, [])
#             oldValue.append(s1)
#             s1Dict[aftModelName] = oldValue

#             oldValue = xTestScaledLastDict.get(aftModelName, [])
#             oldValue.append(xTestScaledLast)
#             xTestScaledLastDict[aftModelName] = oldValue             
                        
#         coefficientsFoldList.append(coefficientsDict)
#         fitResultsFoldList.append(fitResults)
#         predictionsFoldList.append(summaryDf)
                
#     predictionsDf = pd.concat(predictionsFoldList, axis=0)
#     predictionsDf.sort_index(inplace=True)

#   concordance index
#     ciDict = {}
#     for column in sY_List: 
#         ci = concordance_index(df['duration'].values, predictionsDf[column].values, df['E']) # 1 good
#         ciDict[column] = ci
        
# #   select only last state
#     predictionsLastStateDf = predictionsDf.loc[lastStateIdList].copy(deep=True)
    
# #   concordance index last state
#     ciLastDict = {}
#     for column in sY_List: 
#         ci = concordance_index(predictionsLastStateDf['duration'].values, predictionsLastStateDf[column].values, predictionsLastStateDf['E']) # 1 good
#         ciLastDict[column] = ci    
    
    
#     ibsFullDict = {}
#     ibsLastDict = {}
#     for model in modelsAFT:
#         model = modelsAFT[4]
#         s = sDict[model]
#         s = pd.concat(s, axis=1)
#         s1 = s1Dict[model]
#         s1 = pd.concat(s1, axis=1)        
#         xTestScaled = xTestScaledDict[model]
#         xTestScaled = pd.concat(xTestScaled, axis=0)        
#         xTestScaledLast = xTestScaledLastDict[model]
#         xTestScaledLast = pd.concat(xTestScaledLast, axis=0)    
#         ev = EvalSurv(s, xTestScaled['duration'].values, xTestScaled['E'].values, censor_surv=s)
#         ibs = ev.integrated_brier_score(timeLineObserved)  
#         ev1 = EvalSurv(s1, xTestScaledLast['duration'].values, xTestScaledLast['E'].values, censor_surv=s1)
#         ibs1 = ev1.integrated_brier_score(timeLineObserved)
#         ibsFullDict[model] = ibs
#         ibsLastDict[model] = ibs1


#     predictionsLastDf = predictionsDf.loc[lastStateIdList].copy(deep=True)

#     ev1.brier_score(timeLineObserved).plot()
    
    
    
    # fit cox on last state
    #     last state + 1 TS
    #     last state + 2 TS
    #     ...
        
        
#!!! reconstruct all predictions and get CI and IBS

#!!! fit all data before plot
    # #   Plot hazard Rate or AFT 
    #     plt.figure(1, figsize=(19,10)) 
    #     ax = fitterAFT.plot()
    #     title = 'Stack {}'.format(aftModelName)
    #     plt.title(title)
    #     fileName = '{}{}'.format(title, '.png')
    #     plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    #     plt.close() 


    """
    #   survival trees
    #   slow
    #   predicts only whithin observed time length; usefull for clients with long history
    
        xTrain, yTrain = get_x_y(xTrainScaled, attr_labels=['E', 'duration'], pos_label=1, survival=True)
        xTest, yTest = get_x_y(xTestScaled, attr_labels=['E', 'duration'], pos_label=1, survival=True)
    
        # maxLen = len(timeLine)
        max_features='auto'
        n_estimators = 100
        max_depth = 5
        min_samples_split = 2
        min_samples_leaf = 1
        n_jobs = 6
        # models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
            # 'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
        
        models = ['ExtraSurvivalTrees', 'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
        
        for modelName in tqdm(models):
        # if True:
            # modelName = 'GradientBoostingSurvivalAnalysis'
            model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
                max_depth=max_depth, min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
                n_jobs=n_jobs, random_state=random_state) 
                            
            model.fit(xTrain, yTrain)
    
            # s = model.predict_survival_function(xTest)
    
            lifetime, _ = predictSurvivalTime(model, xTest, T_List=None, probability=0.5)
            summaryDf['yMedian {}'.format(modelName)] = lifetime
            sY_List.append('yMedian {}'.format(modelName))
    
    
    
    """

        
        