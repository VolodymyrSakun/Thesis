old

import os
import libStent
from datetime import datetime
import pandas as pd
import libPlot
import numpy as np
import lib3
from datetime import timedelta
import lib2
from time import time
import warnings
from math import log
from math import exp
from Rank import Rank
from datetime import date
from TrendFrame import TrendFrame
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from random import sample
import libChurn2
from matplotlib import pyplot as plt
from scipy.stats import norm
from Encoder import DummyEncoderRobust
from sklearn.linear_model import LogisticRegression
import lib4
from xgboost import XGBClassifier
from lifelines import CoxTimeVaryingFitter
from sklearn.preprocessing import minmax_scale
from pandas import Series
from StentBase1 import StentBase1
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# import sys
import pandas as pd
from datetime import date
# from datetime import datetime
from datetime import timedelta
import os
import numpy as np
#import traceback
# import matplotlib.pyplot as plt
# import random
import time

## append path to cfg
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'cfg'))
import gen_cfg as cfg
#sys.path.append(cfg.LIB_DIR)
#sys.path.append(cfg.CONTROL_DIR)
#sys.path.append(cfg.RECOMMENDATION_DIR)
import globalDef
import lib2
import lib3
import sqlGeneral
# import sqlDealership
import warnings

# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column
# from sqlalchemy import Integer, String, BigInteger, SmallInteger, Date, Float, Numeric, Text, TEXT
# from pandas.core.frame import DataFrame

from lifelines import WeibullAFTFitter    
from lifelines import LogLogisticAFTFitter
from lifelines import LogNormalAFTFitter
# from lifelines import WeibullFitter
# from lifelines import LogNormalFitter
# from lifelines import LogLogisticFitter
# from lifelines import ExponentialFitter  
from sklearn.metrics import mean_absolute_error
import lifelines
import libRecommender
import libChurn2
from Rank import Rank
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.datasets import get_x_y
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.preprocessing import MinMaxScaler


def findNearest(array, value):
    idx = lib2.argmin([abs(x - value) for x in array])
    return idx

def splitIndex(data):
    idList = list(data.index)
    return [x.split(' | ')[0] for x in idList]

def find(dDate, dStateList):
    for d in dStateList:
        if d >= dDate:
            return d
    return dStateList[-1]


def getConcordanceMetricAlive(tTrue, tPredicted):
    binScore = np.where(tTrue < tPredicted, 1, 0)
    return np.sum(binScore) / binScore.shape[0]
    
def scaleDf(data, columns=None, scaler=None):
    if columns is None:
        columnsScaled = list(data.columns)
    else:
        columnsScaled = columns
        
    columnsOrig = sorted(list(set(data.columns).difference(columnsScaled)))
    if len(columnsOrig) > 0:
        dataOrig = data[columnsOrig].copy(deep=True)
    else:
        dataOrig = None
        
    if scaler is None:
        scaler = StandardScaler()
        a = scaler.fit_transform(data[columnsScaled])
    else:
        a = scaler.transform(data[columnsScaled])
            
    scaledDf = pd.DataFrame(a, index=data.index, columns=columnsScaled)
    
    if dataOrig is not None:
        return pd.concat([dataOrig, scaledDf], axis=1), scaler
        
    return scaledDf, scaler


def findFeatureOut(fitterAFT):
    b = fitterAFT.params_.reset_index()
    columns = list(b.columns)
    featureOut = None
    minValue = np.inf
    for i, row in b.iterrows():
        feature = row['covariate']
        if feature == 'Intercept':
            continue
        value = abs(row[columns[-1]])
        if value < minValue:
            featureOut = feature
            minValue = value
    return featureOut
    
def fitAFT(fitterAFT, data, duration_col='T', event_col='E', 
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
    return {'AIC': aic, 'AUC': fitterAFT.concordance_index_, 
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
            verbose=0, warm_start=False, max_samples=None)

    if modelName == 'GradientBoostingSurvivalAnalysis':
        if max_depth is None:
            max_depth = 100
        return GradientBoostingSurvivalAnalysis(loss='coxph', learning_rate=0.1, 
            n_estimators=n_estimators, criterion='friedman_mse', 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=0.0, max_depth=max_depth, min_impurity_split=None, 
            min_impurity_decrease=0.0, random_state=random_state, max_features=max_features,
            max_leaf_nodes=None, subsample=1.0, dropout_rate=0.0, verbose=0, ccp_alpha=0.0)

    if modelName == 'ComponentwiseGradientBoostingSurvivalAnalysis':
        return ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', 
            learning_rate=0.1, n_estimators=n_estimators, subsample=1.0, 
            dropout_rate=0, random_state=random_state, verbose=verbose)
  
    return None



def getDeadLifetime(mapClientToLastState, lastState):
    mapDeadToLifeTime = {}
    for client, status in lastState.lostDict.items():
        if status == 1:
            first = lastState.firstEventDict.get(client)
            last = mapClientToLastState.get(client)
            if first is not None and last is not None:
                lifetime = (last - first).days
                mapDeadToLifeTime[client] = lifetime
    return mapDeadToLifeTime

def buildTrainFrame(clients, statesDict, mapClientToLastStateDate):
    clientFeaturesDict = {}
    dates = sorted(list(statesDict.keys()))
    
    for client in clients:
        E = statesDict[dates[-1]].lostDict.get(client) # from last state
        if E is None:
            continue
        
        dDate = mapClientToLastStateDate.get(client)
        if dDate is None:
            continue
        
        clientFeatures1 = statesDict[dDate].getFeatures(client)
        if clientFeatures1 is None:
            continue
        
        clientFeatures1['E'] = E
        clientFeaturesDict[client] = clientFeatures1
        
        
    clientsList = sorted(list(clientFeaturesDict.keys()))    
    for client, clientFeatures1 in clientFeaturesDict.items():
        if isinstance(clientFeatures1, dict):
            break    
    featuresList = sorted(list(clientFeatures1.keys()))

    featuresFrame = TrendFrame(clientsList, featuresList, dtype=float)

    for client, clientFeatures1 in clientFeaturesDict.items():
        featuresFrame.fillRow(client, clientFeatures1, toFloat=True)

    featuresDf = featuresFrame.to_pandas()
    
    return featuresDf

def buildTestExtendedFrame(clients, statesDict, mapClientToLastStateDate):
    """
    clients = clientsTestAliveList

    """
    
    dates = sorted(list(statesDict.keys()))

    clientFeatures0 = None
    clientFeaturesExtendedDict = {}
    for client in clients:
        # client = '10984'
        # if client == '10984':
            # raise ValueError()
    
        dLastState = mapClientToLastStateDate.get(client)
        if dLastState is None:
            continue
        
        E = statesDict[dates[-1]].lostDict.get(client)
        if E is None:
            continue
    
        shift = 0 # corresponds to last state: last transaction or death state
        idx = dates.index(dLastState)
        nSteps = idx
        censored = E # status = censoring indicator
        while idx >= 0:
            idExtended = '{} | {:03d}'.format(client, shift)
            dState = dates[idx]
            
            clientFeatures1 = statesDict[dState].getFeatures(client)
            if clientFeatures1 is None:
                break # birth riched
            clientFeatures1['E'] = E 
            clientFeatures1['_shift'] = shift
            clientFeatures1['_period'] = idx
            stop = clientFeatures1['T']
            if idx == 0:
                start = 0
            else:
                start = stop - TS_STEP                
            clientFeatures1['_start'] = start
            clientFeatures1['_stop'] = stop            
            clientFeatures1['_nSteps'] = nSteps
            clientFeatures1['_censored'] = censored            
            clientFeaturesExtendedDict[idExtended] = clientFeatures1
            if clientFeatures0 is None:
                clientFeatures0 = clientFeatures1.copy()
                
            E = 0 # change status to alive if was dead
            shift += 1 # number ts from last state back in time
            idx -= 1 # bsck in time to previos state
            
#   Dict to Frame                    
    clientsExtendedList = sorted(list(clientFeaturesExtendedDict.keys()))     
    featuresList = sorted(list(clientFeatures0.keys()))

    featuresExtendedFrame = TrendFrame(clientsExtendedList, featuresList, dtype=float)
    
    for idExtended, clientFeatures1 in tqdm(clientFeaturesExtendedDict.items()):
        featuresExtendedFrame.fillRow(idExtended, clientFeatures1, toFloat=True)
        
    featuresExtendedDf = featuresExtendedFrame.to_pandas()
    
    return featuresExtendedDf
    
def scoreCDF(yTrue, yHat):
    m = np.mean(yTrue)
    s = np.std(yTrue)
    mae = mean_absolute_error(yTrue, yHat)
    return norm.sf(mae, m, s)
    
    
def fitClassic(xTrain, yTrain, n_jobs=6, verbose=False, eval_set=None, eval_metric='auc'):
    """
    fit data with 3 classifiers
    
    Parameters
    ----------
    xTrain : numpy ndarray, columns = features, rows = observations
    yTrain : 1D numpy ndarray, len=xTrain.shape[0]

    Returns
    -------
    dict : trained classifiers
    """
    # calculate class weights
    total = len(yTrain)
    pos = sum(yTrain)
    neg = total - pos
    scale_pos_weight = neg/pos    
    
    classifierXG = XGBClassifier(max_depth=7, learning_rate=0.05, n_estimators=300,\
        objective='binary:logistic', booster='gbtree', n_jobs=n_jobs,\
        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,\
        colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,\
        scale_pos_weight=scale_pos_weight, base_score=0.5, random_state=101, seed=None, 
        missing=0, use_label_encoder=False)  
       
    if eval_set is not None:
        classifierXG.fit(xTrain, yTrain, verbose=verbose, eval_set=eval_set, eval_metric=eval_metric)
    else:
        classifierXG.fit(xTrain, yTrain, verbose=verbose)
        
    return classifierXG

def getSurvival(coxTimeVaryingFitter, data, T_List):
    """
    coxTimeVaryingFitter = ctv
    data = testAliveExtendedDf
    
    """
    h0 = coxTimeVaryingFitter.baseline_cumulative_hazard_
    s0 = coxTimeVaryingFitter.baseline_survival_
    h = coxTimeVaryingFitter.predict_partial_hazard(data)
    h.index = data.index
    h = h.to_dict()    
    f = h0['baseline hazard'] * s0['baseline survival']
    timeline = list(f.index)
    f = f.to_dict()
    h0 = h0['baseline hazard'].to_dict()
    s0 = s0['baseline survival'].to_dict()

    survivalList = []
    # T_List = list(data['T'])
    clients = list(data.index)
    # clientsOrig = list(data['_client'])
    # d1 = data[data['_client'] == '02359']
    for i in range(0, len(clients), 1):
        client = clients[i]
        if client == '02359':
            raise ValueError()
        T = T_List[i]
        t = timeline[findNearest(timeline, T)]
        s = f[t] / (h0[t] * h[client])
        s = min(1, s)
        survivalList.append(s)
    return survivalList

def makeHugeDTPO(data, columnsX, columnsTS):
    dataCopy = data.copy(deep=True)
    newColumns = []
    for column in tqdm(columnsX):
        tsDf = dataCopy[columnsTS].copy(deep=True)
        tsDf = tsDf.multiply(dataCopy[column], axis='index')
        columns = ['{}_{}'.format(column, x) for x in tsDf.columns]
        newColumns.extend(columns)
        tsDf.columns = columns
        dataCopy = pd.concat([dataCopy, tsDf], axis=1)
    dataCopy.drop(columns=columnsX, inplace=True)
    return dataCopy, newColumns
    
def getSurvivalDTPO(data, columnId, columnHazard, columnPeriod):
    """
    data sorted
    data = predictionsTestAliveDf
    client = '07644'
    """
    # client = '00005'
    tmp = data.set_index(columnId).copy(deep=True)
    clients = sorted(list(set(tmp.index)))
    S_List = []
    P_List = []
    for client in clients:
        tmp1 = tmp.loc[client]
        if isinstance(tmp1, Series):
            tmp1 = tmp.loc[[client]]
    
        hazards = list(tmp1[columnHazard].values)
        periods = list(tmp1[columnPeriod].values)
        nPeriods = len(periods)
        sPrev = 1
        s_list = []
        s_list.append(float(sPrev))
        p = 0
        p_list = []
        p_list.append(float(p))
        for i in range(0, nPeriods-1, 1):
            idx = nPeriods - i -1
            h = hazards[idx]
            s = sPrev * (1 - h)
            s_list.insert(0, float(s))
            p = sPrev - s
            p_list.insert(0, float(p))
            sPrev = s
        S_List.extend(s_list)
        P_List.extend(p_list)
    return S_List, P_List
    
def predictSurvivalProbability(fitter, data):
    S_Array = fitter.predict_survival_function(data)
    S_Frame = TrendFrame(clients=None, dates=None, df=S_Array, dtype=float)
    T_List = list(data['T'])
    idxList = list(data.index)
    timeline = S_Frame.clients
    pAliveList = []
    for i in range(0, len(idxList), 1):
        client = idxList[i]
        T = T_List[i]
        idxRow = findNearest(timeline, T)
        idxColumn = S_Frame.mapDateToIndex.get(client)
        pAlive = S_Frame.array[idxRow, idxColumn]
        pAliveList.append(pAlive)        
    return pAliveList

def plotClient(client, data, visitsDict, mapClientToFirstState, dFirst, 
    features=None, prefix='', plotPath=None):
    """
    data = predictionsTestDeadDf
    """    
    visitsList = visitsDict.get(client)
    df1 = data[data['_client'] == client].copy(deep=True)
    dates = list(df1['dDate'])
    sCOX = list(df1['S_COX'])
    s_DTPO_LR2 = list(df1['S_DTPO_LR2'])
    s_DTPO_LR = list(df1['S_DTPO_LR'])
    s_DTPO_XG = list(df1['S_DTPO_XG'])
    s_DTPO_RF = list(df1['S_DTPO_RF'])

    plt.figure(1, figsize=(19,10)) 

    plt.plot(dates, sCOX, c='b', linestyle='-', label='survival COX')
    plt.plot(dates, s_DTPO_LR, c='g', linestyle='-', label='survival LR')
    plt.plot(dates, s_DTPO_LR2, c='g', linestyle='-.', label='survival LR2')
    plt.plot(dates, s_DTPO_XG, c='r', linestyle='-', label='survival XG')
    plt.plot(dates, s_DTPO_RF, c='y', linestyle='-', label='survival RF')
    if visitsList is not None:
        for visit in visitsList:
            plt.axvline(visit, linestyle=':', c='k')   
    
    plt.axvline(dFirst, linestyle='-', c='k', label='FMC start')
    
    dFirstState = mapClientToFirstState.get(client, None)
    if dFirstState is not None:
        plt.axvline(dFirstState, linestyle='-.', c='k', label='First State')            
    
    if isinstance(features, list):
        for feature in features:
            y = df1[feature].values
            y = minmax_scale(y.reshape(-1, 1), feature_range=(0, 1))
            y = y.reshape(-1)
            plt.plot(dates, y, linestyle='-', c='m', label=feature)
            
    plt.legend()

    title = '{} {} survival'.format(prefix, client)
    plt.title(title)
    
    fileName = '{}{}'.format(title, '.png')
    if plotPath is not None:
        path = os.path.join(plotPath, fileName)
    else:
        path = fileName
        
    plt.savefig(path, bbox_inches='tight')
    plt.close() 
    
    return 
    
def predictSurvivalTime(estimator, xTest, T_List, probability=0.5):
    s = estimator.predict_survival_function(xTest)
    lifetimeList = []
    pAliveList = []
    for i in range(0, s.shape[0], 1):
        s1 = s[i]
        timeline1 = s1.x
        pAlive = s1.y
        idx = (np.abs(pAlive - probability)).argmin()
        lifetimeList.append(int(timeline1[idx]))
        T = T_List[i]
        idx2 = findNearest(timeline1, T)
        pAliveList.append(pAlive[idx2])
    return lifetimeList, pAliveList

def plotClient2(client, data, visitsDict, mapClientToFirstState, dFirst, 
    prefix='', plotPath=None):
    """
    data = predictionsTestDeadDf
    '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    """    
    visitsList = visitsDict.get(client)
    df1 = data[data['_client'] == client].copy(deep=True)
    dates = list(df1['dDate'])
    sCOX = list(df1['S_COX'])
    s_CoxPHFitter = list(df1['S_CoxPHFitter'])
    s_DTPO_LR2 = list(df1['S_DTPO_LR2'])
    s_DTPO_LR = list(df1['S_DTPO_LR'])
    s_DTPO_XG = list(df1['S_DTPO_XG'])
    s_DTPO_RF = list(df1['S_DTPO_RF'])

    s_WeibullAFT = list(df1['S_WeibullAFT'])
    s_LogNormalAFT = list(df1['S_LogNormalAFT'])
    s_LogLogisticAFT = list(df1['S_LogLogisticAFT'])

    s_RandomSurvivalForest = list(df1['S_RandomSurvivalForest'])
    s_ExtraSurvivalTrees = list(df1['S_ExtraSurvivalTrees'])
    s_GradientBoostingSurvivalAnalysis = list(df1['S_GradientBoostingSurvivalAnalysis'])
    s_ComponentwiseGradientBoostingSurvivalAnalysis = list(df1['S_ComponentwiseGradientBoostingSurvivalAnalysis'])

    plt.figure(1, figsize=(19,10)) 

    plt.plot(dates, sCOX, c='b', linestyle='-', label='survival COX Time-varying')
    plt.plot(dates, s_CoxPHFitter, c='b', linestyle='--', label='survival COX last state')
    
    plt.plot(dates, s_DTPO_LR, c='g', linestyle='-', label='survival LR')
    plt.plot(dates, s_DTPO_LR2, c='g', linestyle='--', label='survival LR Huge')
    
    plt.plot(dates, s_DTPO_XG, c='r', linestyle='-', label='survival XG')
    plt.plot(dates, s_DTPO_RF, c='r', linestyle='--', label='survival RF')
    
    plt.plot(dates, s_WeibullAFT, c='b', linestyle=':', label='Weibull AFT')
    plt.plot(dates, s_LogNormalAFT, c='g', linestyle=':', label='LogNormal AFT')
    plt.plot(dates, s_LogLogisticAFT, c='r', linestyle=':', label='LogLogistic AFT')
    
    plt.plot(dates, s_RandomSurvivalForest, c='b', linestyle='-.', label='RandomSurvivalForest')
    plt.plot(dates, s_ExtraSurvivalTrees, c='g', linestyle='-.', label='ExtraSurvivalTrees')
    plt.plot(dates, s_GradientBoostingSurvivalAnalysis, c='r', linestyle='-.', label='GradientBoostingSurvivalAnalysis')
    plt.plot(dates, s_ComponentwiseGradientBoostingSurvivalAnalysis, c='k', linestyle='-.', label='ComponentwiseGradientBoostingSurvivalAnalysis')    
    
    if visitsList is not None:
        for visit in visitsList:
            plt.axvline(visit, linestyle=':', c='k')   
    
    plt.axvline(dFirst, linestyle='-', c='k', label='FMC start')
    
    dFirstState = mapClientToFirstState.get(client, None)
    if dFirstState is not None:
        plt.axvline(dFirstState, linestyle='-.', c='k', label='First State')            
            
    plt.legend()

    title = '{} {} pAlive'.format(prefix, client)
    plt.title(title)
    
    fileName = '{}{}'.format(title, '.png')
    if plotPath is not None:
        path = os.path.join(plotPath, fileName)
    else:
        path = fileName
        
    plt.savefig(path, bbox_inches='tight')
    plt.close() 
    
    return

###############################################################################

if __name__ == "__main__":
    
    MIN_FREQUENCY = 3
    mergeTransactionsPeriod=7
    RFM_HOLDOUT = 180
    TS_STEP = 7
    K = 1.5
    INCLUDE_START = True
    INCLUDE_END = True
    SHORT_SLOPE_SPAN = 10 # SHORT_SLOPE_SPAN * TS_STEP days
    TEST_DEAD_FRACTION = 0.3
    TEST_ALIVE_FRACTION = 0.2
    random_state = 101
    
    eventType = 'final' # instant delayed final
    
    # columnId = 'iId'
    # columnEvent = 'ds'
    # columnSale = 'Sale'
    dataName = 'CDNOW'
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    StentBase1.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, dataName)    
    StentBase1.checkDir(resultsSubDir)
        
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])
    g = transactionsDf.groupby('iId')['ds'].apply(list)
    visitsDict = g.to_dict() 

    dStateList = sorted(list(statesDict.keys()))
    dStateReverseList = sorted(dStateList, reverse=True)
    dFirst = dStateList[0]
    dLast = dStateList[-1]
    span = (dLast - dFirst).days
    lastState = statesDict[dLast]
    firstState = statesDict[dFirst]
    
    E_Dict = lastState.lostDict
    firstEventDict = lastState.firstEventDict
    lastEventDict = lastState.lastEventDict
    lostDict = lastState.lostDict

    mapDateToIdx = {x: y for y, x in enumerate(dStateList)}
    mapIdxToDate = {y: x for x, y in mapDateToIdx.items()}

#   1. death time == last event
#   2. death time == last event + q05
    mapClientToLastStateDateInstant = {}
    dropInstantList = [] # died before start RFM == first state
    mapClientToLastStateDateDelayed = {}
    dropDelayedList = [] # died before start RFM == first state
    for client, status in E_Dict.items():
        if status == 0:
            mapClientToLastStateDateInstant[client] = dLast
            mapClientToLastStateDateDelayed[client] = dLast            
        else:
            lastEvent = lastState.lastEventDict.get(client)
            if lastEvent < dFirst:
                dropInstantList.append(client)
            dState = find(lastEvent, dStateList)
            mapClientToLastStateDateInstant[client] = dState
            
            q05 = lastState.q05Dict.get(client)
            lastEventExpected = lastEvent + timedelta(days=q05)
            if lastEventExpected < dFirst:
                dropDelayedList.append(client)
            dStateExpected = find(lastEventExpected, dStateList)
            mapClientToLastStateDateDelayed[client] = dStateExpected
        
#   3. death time == last event + q09 * K 
    mapClientToLastStateDateFinal = {}    
    dropFinalList = []    
    for client, E in lostDict.items():
        if E == 0:
            mapClientToLastStateDateFinal[client] = dLast
        else:
            # dead in last state; find when died
            dPrevios = dStateReverseList[0] # from recent to old
            found = False
            for dDate in dStateReverseList[1:]:
                status = statesDict[dDate].lostDict.get(client)
                if status == 0:
                    # active here; dead in one more recent state
                    mapClientToLastStateDateFinal[client] = dPrevios
                    found = True
                    break
                else:
                    dPrevios = dDate
            if not found:
                dropFinalList.append(client)
           
#   good    
    mapClientToFirstState = {}
    for client in lostDict.keys():
        for dState in dStateList:
            if client in statesDict[dState].clientsAll:            
                mapClientToFirstState[client] = dState
                break
           
            
    if eventType == 'instant':
        dropList = dropInstantList
        mapClientToLastStateDate = mapClientToLastStateDateInstant
    elif eventType == 'delayed':
        dropList = dropDelayedList
        mapClientToLastStateDate = mapClientToLastStateDateDelayed
    elif eventType == 'final':
        dropList = dropFinalList
        mapClientToLastStateDate = mapClientToLastStateDateFinal        
        
###############################################################################
#   split clients
    clientsAllList = sorted(set(lostDict.keys()).difference(set(dropList)))
    clientsDeadList = sorted([x for x, y in lostDict.items() if y == 1])
    clientsAliveList = sorted(list(set(clientsAllList).difference(set(clientsDeadList))))
    
    clientsTestDeadList = sample(clientsDeadList, int(len(clientsDeadList) * TEST_DEAD_FRACTION))
    clientsTestAliveList = sample(clientsAliveList, int(len(clientsAliveList) * TEST_ALIVE_FRACTION))
    
    clientsTrainList = sorted(list(set(clientsAllList).difference(set(clientsTestDeadList))))
    clientsTrainList = sorted(list(set(clientsTrainList).difference(set(clientsTestAliveList))))
            
###############################################################################
#   Build frames
    trainDf = buildTrainFrame(clientsTrainList, statesDict, mapClientToLastStateDate)
    trainExtendedDf = buildTestExtendedFrame(clientsTrainList, statesDict, mapClientToLastStateDate)
    testDeadExtendedDf = buildTestExtendedFrame(clientsTestDeadList, statesDict, mapClientToLastStateDate)
    testAliveExtendedDf = buildTestExtendedFrame(clientsTestAliveList, statesDict, mapClientToLastStateDate)
    trainExtendedDf['_client'] = splitIndex(trainExtendedDf)    
    testDeadExtendedDf['_client'] = splitIndex(testDeadExtendedDf)    
    testAliveExtendedDf['_client'] = splitIndex(testAliveExtendedDf)    

    clientsTrainList = sorted(list(trainExtendedDf['_client'].unique()))
    clientsTestDeadList = sorted(list(testDeadExtendedDf['_client'].unique()))
    clientsTestAliveList = sorted(list(testAliveExtendedDf['_client'].unique()))

    nTimeStepsMax = max(trainExtendedDf['_nSteps'])

    trainExtendedDf = trainExtendedDf.append(testDeadExtendedDf.iloc[0])
    trainExtendedDf['_period'].iloc[len(trainExtendedDf)-1] = nTimeStepsMax+1
    
    dummyEncoder = DummyEncoderRobust(unknownValue=0, prefix_sep='_')
    prefix = 'ts'
    trainDummyDf = dummyEncoder.fit_transform(trainExtendedDf, '_period', prefix=prefix, drop=False)
    testDeadDummyDf = dummyEncoder.transform(testDeadExtendedDf, drop=False)
    testAliveDummyDf = dummyEncoder.transform(testAliveExtendedDf, drop=False)

    trainExtendedDf.drop(index=[trainExtendedDf.index[-1]], inplace=True)
    trainDummyDf.drop(index=[trainDummyDf.index[-1]], inplace=True)


    # trainDummyDf.columns
    # dummyEncoder.columnsDummy
    
    columnsX = ['r10_Clumpiness', 'r10_FC', 'r10_FM', 'r10_FMC',
        'r10_MC', 'r10_MoneySum', 'r10_Q05', 'r10_Q09', 
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC']
    
    columnsTS = dummyEncoder.columnsDummy
    


    trainHugeDf, newColumns = makeHugeDTPO(trainDummyDf, columnsX, columnsTS)
    testDeadHugeDf, _ = makeHugeDTPO(testDeadDummyDf, columnsX, columnsTS)
    testAliveHugeDf, _ = makeHugeDTPO(testAliveDummyDf, columnsX, columnsTS)


    a = list(trainHugeDf.columns)

    columnsX = columnsTS.copy()
    columnsX.extend(newColumns)
    columnsX.extend(['clumpiness', 'moneyDaily', 'moneyMedian', 'moneySum', 'q05', 'q09'])
       
    columnY = 'E'
    y = trainHugeDf[columnY].values
    class_weight = lib4.getClassWeights(trainHugeDf[columnY].values)
    class_weight = None
    
    predictionsTestDeadDf = testDeadExtendedDf[['_client', 'E', 'T', '_period', 'frequency', 'r10_RFMC']]
    predictionsTestDeadDf['dDate'] = predictionsTestDeadDf['_period'].map(mapIdxToDate)

    predictionsTestAliveDf = testAliveExtendedDf[['_client', 'E', 'T', '_period', 'frequency', 'r10_RFMC']]
    predictionsTestAliveDf['dDate'] = predictionsTestAliveDf['_period'].map(mapIdxToDate)

    testDeadDf = testDeadExtendedDf.copy(deep=True)
    testAliveDf = testAliveExtendedDf.copy(deep=True)
               
    # true T for dead    
    # for dead clients True lifetime == mapClientToLastStateDate - firstEventDict       
    mapDeadToLifeTime = getDeadLifetime(mapClientToLastStateDate, lastState)
   
    testDeadDf['yTrue'] = testDeadDf['_client'].map(mapDeadToLifeTime)
    testAliveDf['yTrue'] = testAliveDf['_client'].map(lastState.T_Dict)
    
###############################################################################
#   LR large
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(trainHugeDf[columnsX].values)
    xTestDeadScaled = scaler.transform(testDeadHugeDf[columnsX].values)
    xTestAliveScaled = scaler.transform(testAliveHugeDf[columnsX].values)
    
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=random_state, solver='saga', max_iter=300, 
        multi_class='auto', verbose=1, warm_start=False, n_jobs=6, l1_ratio=None)
     
    lr.fit(xTrainScaled, trainHugeDf[columnY].values)
    
    yHatTestDead_LR2 = lr.predict_proba(xTestDeadScaled)
    yHatTestAlive_LR2 = lr.predict_proba(xTestAliveScaled)

    predictionsTestDeadDf['H_DTPO_LR2'] = yHatTestDead_LR2[:, 1]
    predictionsTestAliveDf['H_DTPO_LR2'] = yHatTestAlive_LR2[:, 1]    
    
###############################################################################
#   Cox time-varying    
    
    # columnsXY = ['clumpiness', 'moneyDaily', 'moneyMedian', 'moneySum',
    #    'q05', 'q09', 'r10_Clumpiness', 'r10_FC', 'r10_FM', 'r10_FMC',
    #    'r10_Loyalty', 'r10_MC', 'r10_MoneyDaily',
    #    'r10_MoneyMedian', 'r10_MoneySum', 'r10_Q05', 'r10_Q09', 'r10_RF',
    #    'r10_RFM', 'r10_RFMC', 'timeTo05', 'timeTo09',
    #    'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
    #    'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 'E', '_start', '_stop', '_client']
    
    columnsX = ['clumpiness', 'moneyDaily', 'moneyMedian', 'moneySum',
       'q05', 'q09', 'r10_Clumpiness', 'r10_FC', 'r10_FM', 'r10_FMC',
       'r10_Loyalty', 'r10_MC', 'r10_MoneyDaily',
       'r10_MoneyMedian', 'r10_MoneySum', 'r10_Q05', 'r10_Q09', 'r10_RF',
       'r10_RFM', 'r10_RFMC', 'timeTo05', 'timeTo09',
       'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
       'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC']
    
    columnY = 'E'    
    columnStart = '_start'
    columnStop = '_stop'    
    columnId = '_client'
    
    xTrainScaled = scaler.fit_transform(trainExtendedDf[columnsX])
    xTrainScaled = pd.DataFrame(xTrainScaled, index=trainExtendedDf.index, columns=columnsX)
    xTrainScaled['E'] = trainExtendedDf['E']
    xTrainScaled['_start'] = trainExtendedDf['_start']
    xTrainScaled['_stop'] = trainExtendedDf['_stop']
    xTrainScaled['_client'] = trainExtendedDf['_client']
    
    xTestDeadScaled = scaler.transform(testDeadExtendedDf[columnsX])
    xTestDeadScaled = pd.DataFrame(xTestDeadScaled, index=testDeadExtendedDf.index, columns=columnsX)
    xTestDeadScaled['E'] = testDeadExtendedDf['E']
    xTestDeadScaled['_start'] = testDeadExtendedDf['_start']
    xTestDeadScaled['_stop'] = testDeadExtendedDf['_stop']
    xTestDeadScaled['_client'] = testDeadExtendedDf['_client']    
    
    xTestAliveScaled = scaler.transform(testAliveExtendedDf[columnsX])
    xTestAliveScaled = pd.DataFrame(xTestAliveScaled, index=testAliveExtendedDf.index, columns=columnsX)
    xTestAliveScaled['E'] = testAliveExtendedDf['E']
    xTestAliveScaled['_start'] = testAliveExtendedDf['_start']
    xTestAliveScaled['_stop'] = testAliveExtendedDf['_stop']
    xTestAliveScaled['_client'] = testAliveExtendedDf['_client']      
       
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(xTrainScaled, id_col=columnId, event_col=columnY, 
        start_col=columnStart, stop_col=columnStop, show_progress=True)
    
#   Plot hazard Rate    
    plt.figure(1, figsize=(19,10)) 
    ax = ctv.plot()
    fileName = '{}{}'.format('HR', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
#   Save coefficients to excel    
    summary = ctv.summary    
    summary.to_excel(os.path.join(resultsSubDir, 'summary.xlsx'))
    
    survivalTestDead = getSurvival(ctv, xTestDeadScaled, list(testDeadExtendedDf['T']))
    survivalTestAlive = getSurvival(ctv, xTestAliveScaled, list(testAliveExtendedDf['T']))

    predictionsTestDeadDf['S_COX'] = survivalTestDead    
    predictionsTestAliveDf['S_COX'] = survivalTestAlive

################################################################################

    # Discrete Time Proportional Odds Model (DTPO)
    
    columnsX = ['clumpiness',
       'moneyDaily', 'moneyMedian', 'moneySum',
       'q05', 'q09', 'r10_Clumpiness', 'r10_FC', 'r10_FM', 'r10_FMC',
       'r10_Loyalty', 'r10_MC', 'r10_MoneyDaily',
       'r10_MoneyMedian', 'r10_MoneySum', 'r10_Q05', 'r10_Q09', 'r10_RF',
       'r10_RFM', 'r10_RFMC', 'r10_Recency', 'recency', 'timeTo05', 'timeTo09',
       'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
       'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC']
    columnsX.extend(dummyEncoder.columnsDummy)
    
    columnY = 'E'
    
    xTrainScaled = scaler.fit_transform(trainDummyDf[columnsX].values)
    xTestDeadScaled = scaler.transform(testDeadDummyDf[columnsX].values)
    xTestAliveScaled = scaler.transform(testAliveDummyDf[columnsX].values)
    
    # class_weight = lib4.getClassWeights(trainDummyDf[columnY].values)
    
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=random_state, solver='lbfgs', max_iter=100, 
        multi_class='auto', verbose=0, warm_start=False, n_jobs=6, l1_ratio=None)
     
    lr.fit(xTrainScaled, trainDummyDf[columnY].values)
        
    yHatTestDead_LR = lr.predict_proba(xTestDeadScaled)
    yHatTestAlive_LR = lr.predict_proba(xTestAliveScaled)

    predictionsTestDeadDf['H_DTPO_LR'] = yHatTestDead_LR[:, 1]
    predictionsTestAliveDf['H_DTPO_LR'] = yHatTestAlive_LR[:, 1]

###############################################################################

    xg = fitClassic(trainDummyDf[columnsX].values, trainDummyDf[columnY].values, 
        n_jobs=6, verbose=False, eval_set=None, eval_metric='auc')

    yHatTestDead_XG = xg.predict_proba(testDeadDummyDf[columnsX].values)
    yHatTestAlive_XG = xg.predict_proba(testAliveDummyDf[columnsX].values)
    
    predictionsTestDeadDf['H_DTPO_XG'] = yHatTestDead_XG[:, 1]
    predictionsTestAliveDf['H_DTPO_XG'] = yHatTestAlive_XG[:, 1]    
    predictionsTestDeadDf['H_DTPO_XG'] = predictionsTestDeadDf['H_DTPO_XG'].astype(float)
    predictionsTestAliveDf['H_DTPO_XG'] = predictionsTestAliveDf['H_DTPO_XG'].astype(float)

###############################################################################    

    rf = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
        bootstrap=True, oob_score=False, n_jobs=6, random_state=random_state, 
        verbose=1, warm_start=False, class_weight=class_weight, ccp_alpha=0.0, 
        max_samples=None)
                                        
    rf.fit(trainDummyDf[columnsX].values, trainDummyDf[columnY].values)

    yHatTestDead_RF = rf.predict_proba(testDeadDummyDf[columnsX].values)
    yHatTestAlive_RF = rf.predict_proba(testAliveDummyDf[columnsX].values)
    predictionsTestDeadDf['H_DTPO_RF'] = yHatTestDead_RF[:, 1]
    predictionsTestAliveDf['H_DTPO_RF'] = yHatTestAlive_RF[:, 1]  
                                        
###############################################################################

#   Dead
    S_List, P_List = getSurvivalDTPO(predictionsTestDeadDf, columnId='_client',
        columnHazard='H_DTPO_LR2', columnPeriod='_period')
    predictionsTestDeadDf['S_DTPO_LR2'] = S_List
    predictionsTestDeadDf['P_DTPO_LR2'] = P_List
    
    S_List, P_List = getSurvivalDTPO(predictionsTestDeadDf, columnId='_client',
        columnHazard='H_DTPO_LR', columnPeriod='_period')
    predictionsTestDeadDf['S_DTPO_LR'] = S_List
    predictionsTestDeadDf['P_DTPO_LR'] = P_List

    S_List, P_List = getSurvivalDTPO(predictionsTestDeadDf, columnId='_client',
        columnHazard='H_DTPO_XG', columnPeriod='_period')
    predictionsTestDeadDf['S_DTPO_XG'] = S_List
    predictionsTestDeadDf['P_DTPO_XG'] = P_List
    
    S_List, P_List = getSurvivalDTPO(predictionsTestDeadDf, columnId='_client',
        columnHazard='H_DTPO_RF', columnPeriod='_period')
    predictionsTestDeadDf['S_DTPO_RF'] = S_List
    predictionsTestDeadDf['P_DTPO_RF'] = P_List
    
#   Alive    
    S_List, P_List = getSurvivalDTPO(predictionsTestAliveDf, columnId='_client',
        columnHazard='H_DTPO_LR2', columnPeriod='_period')
    predictionsTestAliveDf['S_DTPO_LR2'] = S_List
    predictionsTestAliveDf['P_DTPO_LR2'] = P_List
    
    S_List, P_List = getSurvivalDTPO(predictionsTestAliveDf, columnId='_client',
        columnHazard='H_DTPO_LR', columnPeriod='_period')
    predictionsTestAliveDf['S_DTPO_LR'] = S_List
    predictionsTestAliveDf['P_DTPO_LR'] = P_List

    S_List, P_List = getSurvivalDTPO(predictionsTestAliveDf, columnId='_client',
        columnHazard='H_DTPO_XG', columnPeriod='_period')
    predictionsTestAliveDf['S_DTPO_XG'] = S_List
    predictionsTestAliveDf['P_DTPO_XG'] = P_List    

    S_List, P_List = getSurvivalDTPO(predictionsTestAliveDf, columnId='_client',
        columnHazard='H_DTPO_RF', columnPeriod='_period')
    predictionsTestAliveDf['S_DTPO_RF'] = S_List
    predictionsTestAliveDf['P_DTPO_RF'] = P_List    
    
   
###############################################################################
   
    modelsU = ['KaplanMeier', 'Weibull', 'Exponential', 'LogNormal','LogLogistic', 'GeneralizedGamma']
    
    maeExpectedList = []
    binScoreExpectedList = []
    
    for modelNameU in modelsU:
        # break
        fitterU = libChurn2.makeUnivariateModel(modelNameU)
        timeLine = list(range(0, int(span*5), 1))
        fitterU = fitterU.fit(trainDf['T'].values,
            trainDf['E'].values, timeline=timeLine, label=modelNameU, entry=None) 
            
        meanSurvivalTime = fitterU.median_survival_time_
        print(modelNameU, meanSurvivalTime)
        
        yHatDead = np.full(shape=(len(testDeadDf)), fill_value=meanSurvivalTime)
        yHatAlive = np.full(shape=(len(testAliveDf)), fill_value=meanSurvivalTime)
        
        if np.isinf(yHatDead).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadDf['yTrue'].values, yHatDead))
        else:
            maeExpectedList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf['T'].values, yHatAlive))

    for i in range(0, len(modelsU), 1):
        print('Model: {}'.format(modelsU[i]))
        print('Dead mean: {}'.format(maeExpectedList[i]))
        print('Alive Mean CI: {}'.format(binScoreExpectedList[i]))

   
        
#   determine baseline: mean survival time balance between CI for alive and MAE for dead (transformed to [0..1])
    # mean = 519
    f1_list = []
    s1_list = []
    s2_list = []
    mae_list = []
    for mean in list(range(0, 1000, 1)):
        yHatDead = np.full(shape=(len(testDeadDf)), fill_value=mean)
        yHatAlive = np.full(shape=(len(testAliveDf)), fill_value=mean)  
        mae = mean_absolute_error(testDeadDf['yTrue'].values, yHatDead)
        score1 = scoreCDF(testDeadDf['yTrue'].values, yHatDead)
        score2 = getConcordanceMetricAlive(testAliveDf['T'].values, yHatAlive)        
        f1 = 2*score1*score2 / (score1 + score2)
        s1_list.append(score1)
        s2_list.append(score2)
        f1_list.append(f1)
        mae_list.append(mae)

    idx = lib2.argmax(f1_list)
    mae_list[idx]
    s2_list[idx]
    f1_list[idx]
    print('Best mean survival time:', idx, 'days')
    
###############################################################################
#   Cox + AFT
    
    columnsX = ['clumpiness',
 'frequency',
 'moneyDaily',
 'moneyMedian',
 'moneySum',
 'pEvent',
 'q05',
 'q09',
 'r10_Clumpiness',
 'r10_FC',
 'r10_FM',
 'r10_FMC',
 'r10_Frequency',
 'r10_MC',
 'r10_MoneyMedian',
 'r10_MoneySum',
 'r10_Q05',
 'r10_Q09',
 'trendShort_FC',
 'trendShort_FM',
 'trendShort_FMC',
 'trendShort_MC',
 'trend_FC',
 'trend_FM',
 'trend_FMC',
 'trend_MC']

    duration_col = 'T'
    event_col = 'E'

    
    # columnsTestXY = columnsTrainXY.copy()
    # columnsTestXY.remove('E')
    # columnsTestXY.remove('T')


    xTrainScaled = scaler.fit_transform(trainDf[columnsX])
    xTrainScaled = pd.DataFrame(xTrainScaled, index=trainDf.index, columns=columnsX)
    xTrainScaled['E'] = trainDf['E']
    xTrainScaled['T'] = trainDf['T']

    xTestDeadScaled = scaler.transform(testDeadDf[columnsX])
    xTestDeadScaled = pd.DataFrame(xTestDeadScaled, index=testDeadDf.index, columns=columnsX)
    xTestDeadScaled['E'] = testDeadDf['E']
    xTestDeadScaled['T'] = testDeadDf['T']  
    
    xTestAliveScaled = scaler.transform(testAliveDf[columnsX])
    xTestAliveScaled = pd.DataFrame(xTestAliveScaled, index=testAliveDf.index, columns=columnsX)
    xTestAliveScaled['E'] = testAliveDf['E']
    xTestAliveScaled['T'] = testAliveDf['T']
 
    smoothing_penalizer = 0
    penalizer = 0.1
    ancillary = False
    maeExpectedList = []
    maeMedianList = []
    binScoreExpectedList = []
    binScoreMedianList = []
    f1Expected_list = []
    f1Median_list = []
    modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter']
    # models = ['LogNormalAFT']
    timeline = list(range(0, 1000, 1))
    timeline = None
    for aftModelName in modelsAFT:        
        
        fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)

        fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col=duration_col, event_col=event_col, 
            ancillary=ancillary, fit_left_censoring=False, timeline=timeline)

        testDeadDict = predictAFT(fitterAFT, xTestDeadScaled)
        testAliveDict = predictAFT(fitterAFT, xTestAliveScaled)

        pAliveTestDeadList = predictSurvivalProbability(fitterAFT, xTestDeadScaled)
        pAliveTestAliveList = predictSurvivalProbability(fitterAFT, xTestAliveScaled)
        predictionsTestDeadDf['S_{}'.format(aftModelName)] = pAliveTestDeadList
        predictionsTestAliveDf['S_{}'.format(aftModelName)] = pAliveTestAliveList

        if np.isinf(testDeadDict['yExpected']).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadDf['yTrue'].values, testDeadDict['yExpected']))
        else:
            maeExpectedList.append(np.inf)
        if np.isinf(testDeadDict['yMedian']).sum() == 0:
            maeMedianList.append(mean_absolute_error(testDeadDf['yTrue'].values, testDeadDict['yMedian']))
        else:
            maeMedianList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yExpected']))
        binScoreMedianList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yMedian']))

        scoreExpected1 = scoreCDF(testDeadDf['yTrue'].values, testDeadDict['yExpected'])
        scoreExpected2 = getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yExpected'])        
        f1Expected = 2*scoreExpected1*scoreExpected2 / (scoreExpected1 + scoreExpected2)
        f1Expected_list.append(f1Expected)

        scoreMedian1 = scoreCDF(testDeadDf['yTrue'].values, testDeadDict['yMedian'])
        scoreMedian2 = getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yMedian'])        
        f1Median = 2*scoreMedian1*scoreMedian2 / (scoreMedian1 + scoreMedian2)
        f1Median_list.append(f1Median)
        
    for i in range(0, len(modelsAFT), 1):
        print('Model: {}'.format(modelsAFT[i]))
        print('Dead mean: {}. Dead median: {}'.format(maeExpectedList[i], maeMedianList[i]))
        print('Alive Mean CI: {}. Alive Median CI: {}'.format(binScoreExpectedList[i], binScoreMedianList[i]))
        print('f1:', f1Expected_list[i])
        print('f1:', f1Median_list[i])

    # fitterAFT.print_summary()
    # fitterAFT.plot()

    # dir(fitterAFT)
    
    



    
###############################################################################
#   Tree    
    max_features='auto'
    n_estimators = 300
    max_depth = 5
    min_samples_split = 2
    min_samples_leaf = 1
    n_jobs = 6
    random_state = 101
    maeList = []
    binScoreList = []
    models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
        'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    # models = ['GradientBoostingSurvivalAnalysis']
    for modelName in models:
        
        model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
            max_depth=max_depth, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
            n_jobs=n_jobs, random_state=random_state) 
                        
        xTrain, yTrain = get_x_y(xTrainScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
        xTestDead, _ = get_x_y(xTestDeadScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
        xTestAlive, _ = get_x_y(xTestAliveScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)

        model.fit(xTrain, yTrain)
        lifetimeTestDead, pAliveTestDead = predictSurvivalTime(model, xTestDead, list(testDeadDf['T'].values), probability=0.5)
        mae = mean_absolute_error(testDeadDf['yTrue'].values, lifetimeTestDead)

        lifetimeTestAlive, pAliveTestAlive = predictSurvivalTime(model, xTestAlive, list(testAliveDf['T'].values), probability=0.5)
        binScore = getConcordanceMetricAlive(testAliveDf['yTrue'].values, lifetimeTestAlive)

        maeList.append(mae)
        binScoreList.append(binScore)

        predictionsTestDeadDf['S_{}'.format(modelName)] = pAliveTestDead
        predictionsTestAliveDf['S_{}'.format(modelName)] = pAliveTestAlive

    print(maeList)
    print(binScoreList)
    





        

##############################################################################    

    clients = sample(clientsTestDeadList, 20)
    for client in clients:
        plotClient2(client, predictionsTestDeadDf, visitsDict, mapClientToFirstState, dFirst, 
            prefix='lost', plotPath=resultsSubDir)


    clients = sample(clientsTestAliveList, 20)
    for client in clients:
        plotClient2(client, predictionsTestAliveDf, visitsDict, mapClientToFirstState, dFirst, 
            prefix='active', plotPath=resultsSubDir)



    """
    
    from math import exp
    from math import factorial
    import numpy as np
    from matplotlib import pyplot as plt

    x = [3, 5, 10, 4, 2, 3, 5]
    
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    s = sum(x)
    r = n/s
    
def poisson(k, t, r):    
    return (r*t)**k * exp(-r*t)/factorial(k)
    
def waiting(t, r):    
    return np.exp(-r*t)
    
    poisson(1, 100, r)
    1 - waiting(10, r)
    
    y = np.arange(1, 10)
    
    plt.plot(y, 1 - waiting(y, r))
    plt.show()
    
    
    """
    