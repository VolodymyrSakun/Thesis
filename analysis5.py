# looks like survival built in Feb 2022

import os
# import libStent
# from datetime import datetime
import pandas as pd
# import libPlot
import numpy as np
import lib3
from datetime import timedelta
import lib2
# from time import time
import warnings
# from math import log
from math import exp
from Rank import Rank
# from datetime import date
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
from FrameList import FrameList
from FrameArray import FrameArray
from State import State
import numpy
from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# from State import K

# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column
# from sqlalchemy import Integer, String, BigInteger, SmallInteger, Date, Float, Numeric, Text, TEXT
# from pandas.core.frame import DataFrame

# from lifelines import WeibullAFTFitter    
# from lifelines import LogLogisticAFTFitter
# from lifelines import LogNormalAFTFitter
# from lifelines import WeibullFitter
# from lifelines import LogNormalFitter
# from lifelines import LogLogisticFitter
# from lifelines import ExponentialFitter  
from sklearn.metrics import mean_absolute_error
import lifelines
# import libRecommender
# import libChurn2
# from Rank import Rank
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.datasets import get_x_y
# from statsmodels.stats.outliers_influence import variance_inflation_factor    
# from statsmodels.tools.tools import add_constant
# from sklearn.metrics import r2_score
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import max_error
# from sklearn.preprocessing import MinMaxScaler


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

def buildFrame(clients, statesDict, mapClientToLastState):
    clientFeaturesDict = {}
    
    for client in clients:
        
        dDate = mapClientToLastState.get(client)
        if dDate is None:
            continue
        
        clientFeatures1 = statesDict[dDate].getFeatures(client)
        if clientFeatures1 is None:
            continue
        
        # clientFeatures1['E'] = E
        clientFeaturesDict[client] = clientFeatures1
        
        
    clientsList = sorted(list(clientFeaturesDict.keys()))    
    for client, clientFeatures1 in clientFeaturesDict.items():
        if isinstance(clientFeatures1, dict):
            break    
    featuresList = sorted(list(clientFeatures1.keys()))

    featuresFrame = FrameList(rowNames=clientsList, columnNames=featuresList, 
        df=None, dtype=None)
    for client, clientFeatures1 in clientFeaturesDict.items():
        featuresFrame.fillRow(rowName=client, data=clientFeatures1)

    featuresDf = featuresFrame.to_pandas()
    featuresDf = featuresDf[featuresDf['D_State'] > 0]
    return featuresDf

def buildPersonPeriodFrame(clients, statesDict, mapClientToLastState):
    """
    Make person-period frame for CoxTV
    Input for preprocessing for DTPO models (LR)
    
    Index: (client_id | period)
    Column: Features + complementary
    
    clients = clientsTestDeadList
    clients = clientsTrainList

    testDeadExtendedDf

    """
    mapClientToFirstState = lastState.birthStateDict
    
    featuresList = None
    clientFeaturesExtendedDict = {}
    for client in clients:
        # client = '00203'

        dFirstState = mapClientToFirstState.get(client)
        if dFirstState is None:
            raise RuntimeError('buildPersonPeriodFrame, Crap 1. Client: {}, dFirstState: {}'.format(client, dFirstState))
        
        dLastState = mapClientToLastState.get(client)
        if dLastState is None:
            raise RuntimeError('buildPersonPeriodFrame, Crap 2. Client: {}, dLastState: {}'.format(client, dLastState))
        
        E_last = statesDict[dLastState].lostDict.get(client)
        if E_last is None:
            raise RuntimeError('buildPersonPeriodFrame, Crap 3. Client: {}, dLastState: {}'.format(client, dLastState))
    
        dState = dFirstState + timedelta(days=TS_STEP) # step forward
        while dState <= dLastState:
            clientFeatures1 = statesDict[dState].getFeatures(client).copy()            
            # State.checkDictForNaN(clientFeatures1)            
            
            if clientFeatures1 is None:
                raise RuntimeError('buildPersonPeriodFrame, Crap 4. Client: {}, dState: {}'.format(client, dState))
                # break # birth riched
            if dState == dLastState:
                E = E_last # active or dead
            else:
                E = 0 # active until last state
                
            # if client in statesDict[dState].clientsNotBorn:
            #     stop = clientFeatures1['T_TS']
            # else:
            #     stop = clientFeatures1['D_TS']
            
            if E == 0:
                stop = clientFeatures1['T_TS'] # alive by fact from last state's view
                # fix dead that will be reborn
                clientFeatures1['D'] = clientFeatures1['T']
                clientFeatures1['D_State'] = clientFeatures1['T_State']
                clientFeatures1['D_TS'] = clientFeatures1['T_TS']
            else:
                stop = clientFeatures1['D_TS'] # dead
                            
            clientFeatures1[columnEvent] = E
            idExtended = '{} | {:03d}'.format(client, stop)  
            clientFeatures1['_date'] = dState            
            clientFeatures1['_start'] = stop - 1
            clientFeatures1['_stop'] = stop
            clientFeatures1['_nSteps'] = int((dLastState - dFirstState).days / TS_STEP)
            clientFeatures1['_censored'] = E_last
            clientFeatures1[columnId] = client  
            
            clientFeaturesExtendedDict[idExtended] = clientFeatures1
            if featuresList is None:
                featuresList = sorted(list(clientFeatures1.keys()))

            # print(dState)
            # print(clientFeatures1)
            
            dState += timedelta(days=TS_STEP)
                       
#   Dict to Frame
    clientsExtendedList = sorted(list(clientFeaturesExtendedDict.keys()))     

    featuresExtendedFrame = FrameList(rowNames=clientsExtendedList, columnNames=featuresList, 
        df=None, dtype=None)
            
    for idExtended, clientFeatures1 in tqdm(clientFeaturesExtendedDict.items()):
        featuresExtendedFrame.fillRow(rowName=idExtended, data=clientFeatures1)        

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

# def getSurvival(coxTimeVaryingFitter, data, T_List):
#     """
#     probably wrong
#     coxTimeVaryingFitter = ctv
#     data = testAliveExtendedDf
#     a = sum(h[0:49])
#     """
#     h0 = coxTimeVaryingFitter.baseline_cumulative_hazard_
#     s0 = coxTimeVaryingFitter.baseline_survival_
#     h = coxTimeVaryingFitter.predict_partial_hazard(data)
#     h.index = data.index
#     h = h.to_dict()    
#     f = h0['baseline hazard'] * s0['baseline survival']
#     timeline = list(f.index)
#     f = f.to_dict()
#     h0 = h0['baseline hazard'].to_dict()
#     s0 = s0['baseline survival'].to_dict()

#     survivalList = []
#     clients = list(data.index)
#     # clientsOrig = list(data['_client'])
#     # d1 = data[data['_client'] == '02359']
#     for i in range(0, len(clients), 1):
#         client = clients[i]
#         if client == '02359':
#             raise ValueError()
#         T = T_List[i]
#         t = timeline[findNearest(timeline, T)]
#         s = f[t] / (h0[t] * h[client])
#         s = min(1, s)
#         survivalList.append(s)
#     return survivalList

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
    data = summaryTestDeadExtendedDf
    client = '00203'
    columnHazard = 'H_DTPO_LR2'
    columnPeriod = '_start'
    """
    # client = '00203' # 01605
    tmp = data.set_index(columnId).copy(deep=True)
    clients = sorted(list(set(tmp.index)))
    rowNames = sorted(list(data[columnPeriod].unique()))
    S_List = []
    P_List = []
    mapClientToMedianTime = {}
    S = FrameArray(rowNames=rowNames, columnNames=clients, df=None, dtype=float)            

    for client in clients:
        tmp1 = tmp.loc[client]
        if isinstance(tmp1, Series):
            tmp1 = tmp.loc[[client]]                
        hazards = list(tmp1[columnHazard].values)            
        periods = list(tmp1[columnPeriod].values)
        if periods[0] != 0:
            print(client)            
            raise RuntimeError('2')    
        nPeriods = len(hazards)
        sPrev = 1 # s(0) = 1
        s_list = []
        p_list = []
        for i in range(0, nPeriods, 1):
            # t = periods[i]
            s = sPrev * (1 - hazards[i]) # s(t)
            p = sPrev - s
            s_list.append(s)
            p_list.append(p)
            sPrev = s
        # mapClientToMeanTime[client] = sum(s_list) # cannot since t does nt go to inf; no features
        if s_list[-1] > 0.5:
            mapClientToMedianTime[client] = np.inf # survival did not cross 0.5
        else:
            t = np.argmin(np.abs([0.5 - x for x in s_list]))
            period = periods[t]
            mapClientToMedianTime[client] = period
            
        for i in range(0, len(periods), 1):          
            S.put(periods[i], client, s_list[i])

        S_List.extend(s_list)
        P_List.extend(p_list)
    return S_List, P_List, mapClientToMedianTime, S
    
def predictSurvivalProbability(fitter, data, T_List):
    S_Array = fitter.predict_survival_function(data)
    S_Frame = FrameArray(rowNames=None, columnNames=None, df=S_Array, dtype=float)            
    idxList = list(data.index)
    timeline = S_Frame.rowNames
    pAliveList = []
    for i in range(0, len(idxList), 1):
        client = idxList[i]
        T = T_List[i]
        idxRow = findNearest(timeline, T)
        idxColumn = S_Frame.mapColumnNameToIndex.get(client)
        pAlive = S_Frame.array[idxRow, idxColumn]
        pAliveList.append(pAlive)        
    return pAliveList

# def plotClient(client, data, visitsDict, mapClientToFirstState, dFirst, 
#     features=None, prefix='', plotPath=None):
#     """
#     data = predictionsTestDeadDf
#     """    
#     visitsList = visitsDict.get(client)
#     df1 = data[data['_client'] == client].copy(deep=True)
#     dates = list(df1['dDate'])
#     sCOX = list(df1['S_COX'])
#     s_DTPO_LR2 = list(df1['S_DTPO_LR2'])
#     s_DTPO_LR = list(df1['S_DTPO_LR'])
#     s_DTPO_XG = list(df1['S_DTPO_XG'])
#     s_DTPO_RF = list(df1['S_DTPO_RF'])

#     plt.figure(1, figsize=(19,10)) 

#     plt.plot(dates, sCOX, c='b', linestyle='-', label='survival COX')
#     plt.plot(dates, s_DTPO_LR, c='g', linestyle='-', label='survival LR')
#     plt.plot(dates, s_DTPO_LR2, c='g', linestyle='-.', label='survival LR2')
#     plt.plot(dates, s_DTPO_XG, c='r', linestyle='-', label='survival XG')
#     plt.plot(dates, s_DTPO_RF, c='y', linestyle='-', label='survival RF')
#     if visitsList is not None:
#         for visit in visitsList:
#             plt.axvline(visit, linestyle=':', c='k')   
    
#     plt.axvline(dFirst, linestyle='-', c='k', label='FMC start')
    
#     dFirstState = mapClientToFirstState.get(client, None)
#     if dFirstState is not None:
#         plt.axvline(dFirstState, linestyle='-.', c='k', label='First State')            
    
#     if isinstance(features, list):
#         for feature in features:
#             y = df1[feature].values
#             y = minmax_scale(y.reshape(-1, 1), feature_range=(0, 1))
#             y = y.reshape(-1)
#             plt.plot(dates, y, linestyle='-', c='m', label=feature)
            
#     plt.legend()

#     title = '{} {} survival'.format(prefix, client)
#     plt.title(title)
    
#     fileName = '{}{}'.format(title, '.png')
#     if plotPath is not None:
#         path = os.path.join(plotPath, fileName)
#     else:
#         path = fileName
        
#     plt.savefig(path, bbox_inches='tight')
#     plt.close() 
    
#     return 
    
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



def getSurvivalHazard(fitter, data, columnId=None):
    """
    fitter: CoxTV fitted model
    Predict hazard array and survival array
    
    Column : (user_id | step)
    Row: time step after birth
    
    columnId=None
    fitter = ctv
    data = xTestDeadExtendedScaled
    """
    
    if columnId is None:
        clients = list(data.index)
    else:
        clients = list(data[columnId].values)
        

    h0 = fitter.baseline_cumulative_hazard_
    # s0 = fitter.baseline_survival_
    h = fitter.predict_partial_hazard(data)
    h0 = h0['baseline hazard'].to_dict()

    timeline = [x for x in h0.keys()]
    
    # make cumulative hazard array: rows == timeline, columns == (user_id | step)
    H = FrameArray(rowNames=timeline, columnNames=clients, df=None, dtype=float)    
    for i in range(0, len(clients), 1):
        client = clients[i]
        hPartial = h[i]
        for j, hBase in h0.items():
            hazard = float(hBase * hPartial)
            H.put(j, client, hazard)
   
    # a = H.to_pandas()
    
    # make survival array: rows == timeline, columns == id
    S = FrameArray(rowNames=timeline, columnNames=clients, df=None, dtype=float)
    for client in tqdm(clients):
        # client = clients[0]
        # idxColumn = S.mapColumnNameToIndex.get(client)
        for t in timeline:
            # idxRow = S.mapRowNameToIndex.get(t)
            hCum = H.get(t, client)
            s = float(exp(-hCum))
            S.put(t, client, s)
            
    # b = S.to_pandas()
            
    return S, H

def getMeanSurvival(S):
    meanSurvivalTime = S.array.sum(axis=0)
    mapClientToMeanSurvival = {}
    for i in range(0, meanSurvivalTime.shape[0], 1):
        client = S.columnNames[i]
        mapClientToMeanSurvival[client] = int(meanSurvivalTime[i])
    return mapClientToMeanSurvival

def getMedianSurvival(S, D_Dict=None):

    mapClientToMedianSurvivalTime = {}
    if isinstance(D_Dict, dict):
        mapClientToP_Alive = {}

    for client in S.columnNames:
        idxColumn = S.mapColumnNameToIndex.get(client)
        idx = np.argmin(np.abs(0.5 - S.array[:, idxColumn]))
        idxMax = S.array.shape[0] -1
        tMin = S.rowNames[0]
        tMax = S.rowNames[-1]

        found05 = True
        if S.array[idx, idxColumn] == 0.5:
            t = S.rowNames[idx]
        else:
            if S.array[idx, idxColumn] > 0.5:
                if idx < idxMax:
                    x2 = S.rowNames[idx+1]
                    x1 = S.rowNames[idx]
                    y2 = S.array[idx+1, idxColumn]
                    y1 = S.array[idx, idxColumn]
                else:
                    found05 = False
            elif S.array[idx, idxColumn] < 0.5:
                x2 = S.rowNames[idx]
                x1 = S.rowNames[idx-1]        
                y2 = S.array[idx, idxColumn]
                y1 = S.array[idx-1, idxColumn]        
            if found05:
                t = x2 - (x2 - x1) * (0.5 - y2) / (y1 - y2)
                
        if found05:
            mapClientToMedianSurvivalTime[client] = int(t)
        else:
            mapClientToMedianSurvivalTime[client] = np.inf
    
        if isinstance(D_Dict, dict):
            t = D_Dict.get(client)
            if t is None:
                raise ValueError('{} not in keys'.format(client))
                
            t = min(int(t), tMax) # if client survived longer than max
            t = max(t, tMin)
            mapClientToP_Alive[client] = S.get(t, client)
    
    if isinstance(D_Dict, dict):
        return mapClientToMedianSurvivalTime, mapClientToP_Alive
    
    return mapClientToMedianSurvivalTime

def makeExtended_D_Dict(dataExtended, columnId, columnD, D_Dict):
    tmp = dataExtended[[columnId]].copy(deep=True)
    tmp[columnD] = tmp[columnId].map(D_Dict)
    tmp[columnId] = tmp.index
    return tmp[columnD].to_dict()

def getMAE_Dead(data, duration_col, predictors):
    maeDict = {}
    for column in predictors:
        if column in data.columns:
            mae = getMAE(data[duration_col].values, data[column].values, skipInf=True)
            if mae is not None:
                maeDict[column] = mae
    return maeDict
    
def getMAE(yTrue, yPred, skipInf=True):
    if not skipInf:
        return mean_absolute_error(yTrue, yPred)
    
    aeList = []
    for y_true, y_pred in zip(yTrue, yPred):
        if lib2.isNaN(y_true) or lib2.isNaN(y_pred):
            continue
        if y_pred == np.inf or y_pred == -np.inf:
            continue
        if y_true == np.inf or y_true == -np.inf:
            continue
        aeList.append(abs(y_true - y_pred))
    if len(aeList) == 0:
        return None
    return np.mean(aeList)
    
def getCI(yTrue, yPred):
    
    ciList = []
    for y_true, y_pred in zip(yTrue, yPred):
        if lib2.isNaN(y_true) or lib2.isNaN(y_pred):
            continue
        if y_pred == np.inf:
            ci = 1            
        elif y_pred >= y_true:
            ci = 1
        else:
            ci = 0            
        ciList.append(ci)
    if len(ciList) == 0:
        return None
    return sum(ciList) / len(ciList)


def getCI_Alive(dataAlive, duration_col, predictors):
    CI_Dict = {}
    for column in predictors:
        if column in dataAlive.columns:
            ciMean = getCI(dataAlive[duration_col].values, dataAlive[column].values)
            if ciMean is not None:
                CI_Dict[column] = ciMean        
    return CI_Dict
    
def makeSurvivalFrame(data, maxLen, mapClientToTS):
    # data = S_TestDeadDict[aftModelName]
    # maxLen = max(mapClientToTS.values()) + 1
    nRows = min(len(data), maxLen)
    rowNames = list(range(0, nRows, 1))
    df = data.iloc[0 : nRows].copy(deep=True)
    S = FrameArray(rowNames=rowNames, columnNames=list(data.columns), df=df.values, dtype=float)            
    for client in S.columnNames:
        idxLast = mapClientToTS.get(client)
        if idxLast < nRows:
            idxColumn = S.mapColumnNameToIndex.get(client)
            S.array[idxLast+1:, idxColumn] = 0
    return S

def makeSurvivalFrameTree(s, clients, maxLen, mapClientToTS):
    # clients = list(testDeadDf.index)
    # s = model.predict_survival_function(xTestDead)    
    rowNames = list(range(0, maxLen, 1))
    S = FrameArray(rowNames=rowNames, columnNames=clients, df=None, dtype=float)            
    S.fillRow(0, 1)
    for i in range(0, s.shape[0], 1):
        s1 = s[i]
        timeline1 = s1.x
        pAlive = s1.y
        client = clients[i]
        idxColumn = S.mapColumnNameToIndex.get(client)
        idxLast = mapClientToTS.get(client)
        for j in range(0, timeline1.shape[0], 1):
            rowName = timeline1[j]
            idxRow = S.mapRowNameToIndex.get(rowName)
            if idxRow is None:
                break
            if idxRow > idxLast:
                break
            S.array[idxRow, idxColumn] = pAlive[j]        
    return S



def makeStructuredArray(durations, events):
    y = []
    for duration, event in zip(durations, events):
        y.append((bool(event), float(duration)))
    return  np.array(y, dtype=[('cens', '?'), ('time', '<f8')])

def getMetrics(trainDuration, trainEvent, testDataList, S_List, sTestDuration, 
               sTestEvent, sPred, predKind, times=None):
    """
    trainDuration = trainForwardDf[duration_col]
    trainEvent=trainForwardDf[event_col]
    testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf]
    S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]]
    sTestDuration=duration_col
    sTestEvent=event_col
    sPred='pAlive_{}'.format(modelName)
    sPred = 'pAlive_COXTV'
    sPred = 'TimeMean_COXTV'
    sPred = 'TimeMedian_COXTV'
    predKind = 'time'
    predKind = 'survival'
    predKind = 'hazard'

     

    a1 = sorted(list(set(trainDuration)))
    a21 = deepcopy(S_List[0].rowNames)
    a22 = deepcopy(S_List[1].rowNames)
    a3 = sorted(list(set(testData[sTestDuration]))) 
    
    a = list()
    a.append(len(sorted(list(set(trainDuration)))))
    a.append(len(deepcopy(S_List[0].rowNames)))
    a.append(len(deepcopy(S_List[1].rowNames)))
    a.append(len(sorted(list(set(testData[sTestDuration])))))


    
    
    """
    testData = pd.concat(testDataList, axis=0)    
    if times is None:
        times = sorted(list(set(testData[sTestDuration])))
        _ = times.pop(-1)

    if times[0] == 0:
        times.pop(0)
    
    S = pd.concat([S_List[0].to_pandas().T, S_List[1].to_pandas().T], axis=0)
    S = S[times]
    S = S.values

    yTrain =  makeStructuredArray(trainDuration, trainEvent)    
    yTest =  makeStructuredArray(testData[sTestDuration], testData[sTestEvent])
    ibs = integrated_brier_score(yTrain, yTest, S, times)

    if predKind == 'survival':
        pred = [1-x for x in testData[sPred]]
    elif predKind == 'time' or predKind == 'hazard':
        pred = [x for x in testData[sPred]]

    ci = concordance_index(testData[sTestDuration].values, pred, testData[sTestEvent]) # 1 good
   
    return ibs, ci

def getMapClientToEvents(transactions):
    g = transactions.groupby('iId')['ds'].apply(list)
    mapClientToEventsList = {}
    for client, dates in g.iteritems():
        mapClientToEventsList[client] = sorted(dates)
    return mapClientToEventsList
    
def getMapClientToDeath(statesDict, mapClientToEventsList):
    dStateList = sorted(list(statesDict.keys()))  
    # dZeroState = dStateList.pop(0)
    dLastState = dStateList[-1]
    lastState = statesDict[dLastState]    
    
    mapClientToDeathList = {}
    client = '00016'
    for client in lastState.clientsAll:
        for dState, state in statesDict.items():
            dLastEvent = state.lastEventDict.get(client) # local death
            if dLastEvent is None:
                continue # client does not exist (not yet)         
            dNextEvent = dLastEvent + timedelta(days=1)
            dates = mapClientToEventsList[client]
            dNextEvent = State.roundToNext(dNextEvent, dates)
            if dNextEvent is None:
                dNextEvent = dLastState # end of TS
            t09 = state.q09Dict.get(client)
            dDeath = dLastEvent + timedelta(days=int(t09 * K))
            
            if dDeath < dNextEvent:
                dDeath = State.roundToNext(dDeath, dStateList)
                l = mapClientToDeathList.get(client, [])
                if dDeath not in l:
                    l.append(dDeath)
                mapClientToDeathList[client] = l
            
    return mapClientToDeathList

def buildPersonPeriodForwardFrame(clients, statesDict, mapClientToDeathList):
    """
    clients = clientsTestDeadList
    clients = clientsTrainList
    """
    
    clientFeaturesExtendedDict = {}
    dStateList = sorted(list(statesDict.keys()))  
    _ = dStateList.pop(0) # remove zero state without trends
    dLastState = dStateList[-1]
    
    for dState in tqdm(dStateList):

        state = statesDict.get(dState)
        clientsList = sorted([x for x in state.clientsAll if x in set(clients)])
        
        for client in clientsList:
            # client = clientsList[0]                            
            if client not in state.clientsActive: 
                # dead already; usefull only for test data to see predictions
                E = -1
                D_ForwardState = 0
                D_ForwardTS = 0                
            else:                
                dDeathList = mapClientToDeathList.get(client)
                if dDeathList is None:
                    E = 0
                    dLastState1 = dLastState
                else:
                    dDeath = State.roundToNext(dState, dDeathList) # to next death
                    if dDeath is None:
                        # alive
                        E = 0
                        dLastState1 = dLastState
                    else:
                        E = 1
                        dLastState1 = State.roundToNext(dDeath, dStateList) # to next state    

                D_ForwardState = round((dLastState1 - dState).days)
                D_ForwardTS = int(D_ForwardState / TS_STEP)
                
            clientFeatures1 = statesDict[dState].getFeatures(client).copy()            
            clientFeatures1['E'] = E # update
            clientFeatures1['D_ForwardState'] = D_ForwardState # duration forward
            clientFeatures1['D_ForwardTS'] = D_ForwardTS # in TS units
            clientFeatures1['_client'] = client
            clientFeatures1['_date'] = dState
            idExtended = '{} | {}'.format(client, dState)
            
            clientFeaturesExtendedDict[idExtended] = clientFeatures1

    for featuresList in clientFeaturesExtendedDict.values():
        break                          
    clientsExtendedList = sorted(list(clientFeaturesExtendedDict.keys()))     
#   Dict to Frame
    featuresExtendedFrame = FrameList(rowNames=clientsExtendedList, 
        columnNames=featuresList, df=None, dtype=None)
#   Fill Frame            
    for idExtended, clientFeatures1 in tqdm(clientFeaturesExtendedDict.items()):
        featuresExtendedFrame.fillRow(rowName=idExtended, data=clientFeatures1)        

    featuresExtendedDf = featuresExtendedFrame.to_pandas()    
    return featuresExtendedDf
    
def makeSurvivalFrameForward(fitter, data, duration_col, timeLine):
    """
    fitter =fitterAFT
    data = xTestDeadScaled
    
    """
    # s = fitter.predict_survival_function(data)
    # rowNames = sorted(list(data[duration_col].unique()))
    # rowNames = timeLine
    # df = s.loc[rowNames].copy(deep=True)
    # S = FrameArray(rowNames=None, columnNames=None, df=df, dtype=float)
    
    s = fitter.predict_survival_function(data, times=timeLine)
    S = FrameArray(rowNames=None, columnNames=None, df=s, dtype=float)
    
    return S

def makeSurvivalFrameTreeForward(model, data):
    # data = xTestDead
    
    clients = list(data.index)
    s = model.predict_survival_function(data)
    
    
    # times = model.event_times_
    # preds = numpy.asarray([[fn(t) for t in times] for fn in s])
    
    rowNames = list(model.event_times_)    
    S = FrameArray(rowNames=rowNames, columnNames=clients, df=None, dtype=float)            
    for i in range(0, s.shape[0], 1):
        s1 = s[i]
        timeline1 = s1.x
        pAlive = s1.y
        client = clients[i]
        idxColumn = S.mapColumnNameToIndex.get(client)                
        for j in range(0, timeline1.shape[0], 1):
            rowName = timeline1[j]
            idxRow = S.mapRowNameToIndex.get(rowName)
            if idxRow is None:
                raise ValueError('1')
            S.array[idxRow, idxColumn] = pAlive[j]
    return S

def plotProbability(client, data, features, visitsDict, mapClientToFirstState, 
        mapClientToLastState, dZero, prefix='', plotPath=None):
    """
    plot probability
    
    data = summaryTestDeadForwardDf
    '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    """    
    visitsList = visitsDict.get(client)
    df1 = data[data['_client'] == client].copy(deep=True)
    dates = list(df1['_date'].values)

    plt.figure(1, figsize=(19,10)) 

    for feature, setting in features.items():
        if feature in data.columns:
            y = list(df1[feature].values)
            plt.plot(dates, y, c=setting['c'], linestyle=setting['linestyle'], 
                     label=setting['label'])
    
    if visitsList is not None:
        for visit in visitsList:
            plt.axvline(visit, linestyle=':', c='k')   
    
    plt.axvline(dZero, linestyle='-', c='m', label='FMC start')
    
    dFirstState = mapClientToFirstState.get(client, None)
    if dFirstState is not None:
        plt.axvline(dFirstState, linestyle='-.', c='k', label='First State')            

    dLastState = mapClientToLastState.get(client, None)
    if dLastState is not None:
        plt.axvline(dLastState, linestyle='-', c='k', label='Last State') 
            
    plt.legend()

    title = '{} {} pAlive'.format(prefix, client)
    plt.title(title)
    
    fileName = '{}{}'.format(title, '.png')
    if plotPath is not None:
        path = os.path.join(plotPath, fileName)
    else:
        path = fileName
        
    # print(path)
    plt.savefig(path, bbox_inches='tight')
    plt.close() 
    
    return

def plotSurvival(client, features, mapClientToFirstState, 
        mapClientToLastState, visitsDict, prefix='', plotPath=None):
    """
    plot survival curve 
    features = featuresDead
    birthDateDict = lastState.birthStateDict
    '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    """    
    
    visitsList = visitsDict.get(client)

    plt.figure(1, figsize=(19,10)) 

    dBirth = mapClientToFirstState.get(client)
    dLast = mapClientToLastState.get(client)

    dDate = dBirth
    dates = []
    while dDate <= dLast:
        dates.append(dDate)
        dDate += timedelta(days=TS_STEP)

    nSteps = len(dates)
    for frame, setting in features.items():
        idxColumn = frame.mapColumnNameToIndex.get(client)
        if idxColumn is None:
            raise ValueError('No client: {} in plotSurvival'.format(client))
        y = frame.array[0 : nSteps, idxColumn]
        # y = [x for x in frame.array[:, idxColumn] if x > 0]
        # x = list(range(0, len(y), 1))  
        if len(dates) > len(y):
            dates = dates[0:len(y)]
        if len(y) > len(dates):
            y = y[0:len(dates)]
            
        plt.plot(dates, y, c=setting['c'], linestyle=setting['linestyle'], 
             label=setting['label'])
        
    if visitsList is not None:
        for visit in visitsList:
            plt.axvline(visit, linestyle=':', c='k')            
            

    plt.axvline(dBirth, linestyle='-.', c='k', label='First State')            
    plt.axvline(dLast, linestyle='-', c='k', label='Last State') 
    
    plt.legend()
    title = '{} {} Survival Curve'.format(prefix, client)
    plt.title(title)
    
    fileName = '{}{}'.format(title, '.png')
    if plotPath is not None:
        path = os.path.join(plotPath, fileName)
    else:
        path = fileName
        
    plt.savefig(path, bbox_inches='tight')
    plt.close() 
    
    return

def plotSurvivalForward(client, features, mapClientToFirstState, 
        mapClientToLastState, visitsDict, prefix='', plotPath=None):
    """
    Forward plot survival curve
    features = featuresDead
    birthDateDict = lastState.birthStateDict
    '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    """    
    
    visitsList = visitsDict.get(client)

    plt.figure(1, figsize=(19,10))   

    for frame, setting in features.items():
        found = False
        for key, idxColumn in frame.mapColumnNameToIndex.items():
            x = key.split(' | ')
            if x[0] == client:            
                found = True    
                dFirst = datetime.fromisoformat(x[1]).date()                
                break
        if not found:
            raise ValueError('client: {} not in frame'.format(client))
        dDate = dFirst
        dates = []
        while len(dates) < len(frame.rowNames):
            dates.append(dDate)
            dDate += timedelta(days=TS_STEP)
            
        # idxColumn = frame.mapColumnNameToIndex.get(client)
        # if idxColumn is None:
            # raise ValueError('No client: {} in plotSurvivalForward'.format(client))
        y = frame.array[:, idxColumn]

        # y = [x for x in frame.array[:, idxColumn] if x > 0]
        # x = list(range(0, len(y), 1))  
        if len(dates) > len(y):
            dates = dates[0:len(y)]
        if len(y) > len(dates):
            y = y[0:len(dates)]
            
        plt.plot(dates, y, c=setting['c'], linestyle=setting['linestyle'], 
             label=setting['label'])
        
    if visitsList is not None:
        for visit in visitsList:
            plt.axvline(visit, linestyle=':', c='k')  
           
    dFirstState = mapClientToFirstState.get(client, None)
    if dFirstState is not None:
        plt.axvline(dFirstState, linestyle='-.', c='k', label='First State')            

    dLastState = mapClientToLastState.get(client, None)
    if dLastState is not None:
        plt.axvline(dLastState, linestyle='-', c='k', label='Last State') 
            
    plt.legend()
    title = '{} {} Survival Curve'.format(prefix, client)
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
    TEST_DEAD_FRACTION = 0.25
    TEST_ALIVE_FRACTION = 0.15
    random_state = 101
    class_weight = None
    columnId = '_client'
    columnDuration = 'D_TS'
    columnDurationForward = 'D_ForwardTS'
    # columnDurationTrue = ''
    columnEvent = 'E'
    columnStart = '_start'
    columnStop = '_stop'  
    
    eventType = 'final' # instant delayed final
    dataName = 'CDNOW'
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, dataName)    
    lib2.checkDir(resultsSubDir)
        
    plotActiveProbaPath = os.path.join(resultsSubDir, 'plotActiveProba')
    plotActiveSurvPath = os.path.join(resultsSubDir, 'plotActiveSurvPath')
    plotForwardActiveProbaPath = os.path.join(resultsSubDir, 'plotForwardActiveProbaPath')
    plotForwardActiveSurvPath = os.path.join(resultsSubDir, 'plotForwardActiveSurvPath')
    
    plotLostProbaPath = os.path.join(resultsSubDir, 'plotLostProba')
    plotLostSurvPath = os.path.join(resultsSubDir, 'plotLostSurvPath')
    plotForwardLostProbaPath = os.path.join(resultsSubDir, 'plotForwardLostProbaPath')
    plotForwardLostSurvPath = os.path.join(resultsSubDir, 'plotForwardLostSurvPath')
    
    StentBase1.checkDir(plotActiveProbaPath)
    StentBase1.checkDir(plotActiveSurvPath)
    StentBase1.checkDir(plotForwardActiveProbaPath)
    StentBase1.checkDir(plotForwardActiveSurvPath)

    StentBase1.checkDir(plotLostProbaPath)
    StentBase1.checkDir(plotLostSurvPath)
    StentBase1.checkDir(plotForwardLostProbaPath)
    StentBase1.checkDir(plotForwardLostSurvPath)
    
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    nStates = len(statesDict)
    transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])
    g = transactionsDf.groupby('iId')['ds'].apply(list)
    visitsDict = g.to_dict() 

    dStateList = sorted(list(statesDict.keys()))
    dStateReverseList = sorted(dStateList, reverse=True)
    dZero = dStateList[0]
    dFirst = dStateList[1]
    dLast = dStateList[-1]
    span = (dLast - dFirst).days
    zeroState = statesDict[dZero]
    firstState = statesDict[dFirst]
    lastState = statesDict[dLast]
    
    E_Dict = lastState.lostDict
    firstEventDict = lastState.firstEventDict
    lastEventDict = lastState.lastEventDict
    lostDict = lastState.lostDict

    mapDateToIdx = {x: y for y, x in enumerate(dStateList)}
    mapIdxToDate = {y: x for x, y in mapDateToIdx.items()}      
    
    # D_StateDict = lastState.D_StateDict
    
    mapClientToLastState = {}
    for client in lastState.clientsActive:
        mapClientToLastState[client] = lastState.dState # alive clients, all to last state

    for dState in dStateReverseList: # from recent to old 
        for client in statesDict[dState].clientsJustLost:
            value = mapClientToLastState.get(client)
            if value is None: # more recent info is absent
                mapClientToLastState[client] = dState            

 # true lifetime in TS units; all dead and alive
    mapClientToTS = {}
    for client in lastState.clientsAll:
        dDeath = lastState.deathStateDict.get(client)
        if dDeath is None:
            mapClientToTS[client] = nStates - 1
        else:
            mapClientToTS[client] = lastState.datesList.index(dDeath)

###############################################################################
#   split clients
    clientsAllList = sorted(list(mapClientToLastState.keys()))
    clientsDeadList = sorted(list(set([x for x, y in lostDict.items() if y == 1]).intersection(set(clientsAllList))))
    clientsAliveList = sorted(list(set(clientsAllList).difference(set(clientsDeadList))))
    
    clientsTestDeadList = sample(clientsDeadList, int(len(clientsDeadList) * TEST_DEAD_FRACTION))
    clientsTestAliveList = sample(clientsAliveList, int(len(clientsAliveList) * TEST_ALIVE_FRACTION))
    
    clientsTrainList = sorted(list(set(clientsAllList).difference(set(clientsTestDeadList))))
    clientsTrainList = sorted(list(set(clientsTrainList).difference(set(clientsTestAliveList))))
            
###############################################################################
#   Build frames
    trainDf = buildFrame(clientsTrainList, statesDict, mapClientToLastState)
    testDeadDf = buildFrame(clientsTestDeadList, statesDict, mapClientToLastState)
    testAliveDf = buildFrame(clientsTestAliveList, statesDict, mapClientToLastState)

    trainExtendedDf = buildPersonPeriodFrame(clientsTrainList, statesDict, mapClientToLastState)
    testDeadExtendedDf = buildPersonPeriodFrame(clientsTestDeadList, statesDict, mapClientToLastState)
    testAliveExtendedDf = buildPersonPeriodFrame(clientsTestAliveList, statesDict, mapClientToLastState)
    # trainExtendedDf['_client'] = splitIndex(trainExtendedDf)    
    # testDeadExtendedDf['_client'] = splitIndex(testDeadExtendedDf)    
    # testAliveExtendedDf['_client'] = splitIndex(testAliveExtendedDf)    
    testDeadDf['_client'] = testDeadDf.index
    testAliveDf['_client'] = testAliveDf.index

    clientsTrainList = sorted(list(trainExtendedDf['_client'].unique()))
    clientsTestDeadList = sorted(list(testDeadExtendedDf['_client'].unique()))
    clientsTestAliveList = sorted(list(testAliveExtendedDf['_client'].unique()))

    nTimeStepsMax = max(trainExtendedDf['_nSteps'])
    
    dummyEncoder = DummyEncoderRobust(unknownValue=0, prefix_sep='_')
    prefix = 'ts'
    trainDummyDf = dummyEncoder.fit_transform(trainExtendedDf, '_start', prefix=prefix, drop=False)
    testDeadDummyDf = dummyEncoder.transform(testDeadExtendedDf, drop=False)
    testAliveDummyDf = dummyEncoder.transform(testAliveExtendedDf, drop=False)

    # trainExtendedDf.drop(index=[trainExtendedDf.index[-1]], inplace=True)
    # trainDummyDf.drop(index=[trainDummyDf.index[-1]], inplace=True)

    columnsPredict = ['_client', '_date', '_start', 'E', 'D_TS', 'frequency']
    summaryTestDeadExtendedDf = testDeadExtendedDf[columnsPredict].copy(deep=True)
    summaryTestAliveExtendedDf = testAliveExtendedDf[columnsPredict].copy(deep=True)
    summaryTestDeadExtendedDf['D_TS_True'] = summaryTestDeadExtendedDf['_client'].map(lastState.D_TS_Dict)

    columnsPredict = ['E', 'D_TS', 'frequency']
    summaryTestDeadDf = testDeadDf[columnsPredict].copy(deep=True)
    summaryTestAliveDf = testAliveDf[columnsPredict].copy(deep=True)
    
    metricsDict = {}
###############################################################################
#   Cox time-varying        
    
    columnsX = ['r10_FC', 'r10_FM', 'r10_FMC', 'r10_MC',  
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 
        'clumpiness', 'frequency', 'loyaltyState', 
        'moneyDaily', 'moneyMedian', 'moneySum', 
        'pEvent', 'pEventPoisson', 'q05', 'q09', 'q05Poisson', 'q09Poisson']
    
    scaler = StandardScaler()
    xTrainExtendedScaled = scaler.fit_transform(trainExtendedDf[columnsX])
    xTrainExtendedScaled = pd.DataFrame(xTrainExtendedScaled, index=trainExtendedDf.index, columns=columnsX)
    xTrainExtendedScaled['E'] = trainExtendedDf['E']
    xTrainExtendedScaled['_start'] = trainExtendedDf['_start']
    xTrainExtendedScaled['_stop'] = trainExtendedDf['_stop']
    xTrainExtendedScaled['_client'] = trainExtendedDf['_client']
    
    xTestDeadExtendedScaled = scaler.transform(testDeadExtendedDf[columnsX])
    xTestDeadExtendedScaled = pd.DataFrame(xTestDeadExtendedScaled, index=testDeadExtendedDf.index, columns=columnsX)
    xTestDeadExtendedScaled['E'] = testDeadExtendedDf['E']  
    
    xTestAliveExtendedScaled = scaler.transform(testAliveExtendedDf[columnsX])
    xTestAliveExtendedScaled = pd.DataFrame(xTestAliveExtendedScaled, index=testAliveExtendedDf.index, columns=columnsX)
    xTestAliveExtendedScaled['E'] = testAliveExtendedDf['E']   
       
    xTestDeadScaled = scaler.transform(testDeadDf[columnsX])
    xTestDeadScaled = pd.DataFrame(xTestDeadScaled, index=testDeadDf.index, columns=columnsX)
    xTestDeadScaled['E'] = testDeadDf['E']    
    
    xTestAliveScaled = scaler.transform(testAliveDf[columnsX])
    xTestAliveScaled = pd.DataFrame(xTestAliveScaled, index=testAliveDf.index, columns=columnsX)
    xTestAliveScaled['E'] = testAliveDf['E']     
    
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(xTrainExtendedScaled, id_col=columnId, event_col=columnEvent, 
        start_col=columnStart, stop_col=columnStop, show_progress=True, step_size=0.1)       
    
#   Plot hazard Rate    
    plt.figure(1, figsize=(19,10)) 
    ax = ctv.plot()
    title = 'Hazard rate Cox time-varying'
    plt.title(title)
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
#   Save coefficients to excel    
    summary = ctv.summary    
    summary.to_excel(os.path.join(resultsSubDir, 'summary Cox time-varying.xlsx'))

    testDeadD_Dict = makeExtended_D_Dict(testDeadExtendedDf, columnId, columnDuration, lastState.D_TS_Dict)
    testAliveD_Dict = makeExtended_D_Dict(testAliveExtendedDf, columnId, columnDuration, lastState.D_TS_Dict)
    
    S_TestDeadExtendedCOXTV, H_TestDeadExtended = getSurvivalHazard(ctv, xTestDeadExtendedScaled, columnId=None)
    S_TestAliveExtendedCOXTV, H_TestAliveExtended = getSurvivalHazard(ctv, xTestAliveExtendedScaled, columnId=None)
    meanSurvivalTestDeadExtended = getMeanSurvival(S_TestDeadExtendedCOXTV)
    meanSurvivalTestAliveExtended = getMeanSurvival(S_TestAliveExtendedCOXTV)
    medianSurvivalTestDeadExtended, pAliveTestDeadExtended = getMedianSurvival(S_TestDeadExtendedCOXTV, D_Dict=testDeadD_Dict)
    medianSurvivalTestAliveExtended, pAliveTestAliveExtended = getMedianSurvival(S_TestAliveExtendedCOXTV, D_Dict=testAliveD_Dict)
    
    S_TestDeadCOXTV, H_TestDead = getSurvivalHazard(ctv, xTestDeadScaled, columnId=None)
    S_TestAliveCOXTV, H_TestAlive = getSurvivalHazard(ctv, xTestAliveScaled, columnId=None)
    meanSurvivalTestDead = getMeanSurvival(S_TestDeadCOXTV)
    meanSurvivalTestAlive = getMeanSurvival(S_TestAliveCOXTV)
    medianSurvivalTestDead, pAliveTestDead = getMedianSurvival(S_TestDeadCOXTV, D_Dict=lastState.D_TS_Dict)
    medianSurvivalTestAlive, pAliveTestAlive = getMedianSurvival(S_TestAliveCOXTV, D_Dict=lastState.D_TS_Dict)
    
    summaryTestDeadExtendedDf['TimeMean_COXTV'] = summaryTestDeadExtendedDf.index.map(meanSurvivalTestDeadExtended)
    summaryTestDeadExtendedDf['TimeMedian_COXTV'] = summaryTestDeadExtendedDf.index.map(medianSurvivalTestDeadExtended)
    summaryTestDeadExtendedDf['pAlive_COXTV'] = summaryTestDeadExtendedDf.index.map(pAliveTestDeadExtended)

    summaryTestAliveExtendedDf['TimeMean_COXTV'] = summaryTestAliveExtendedDf.index.map(meanSurvivalTestAliveExtended)
    summaryTestAliveExtendedDf['TimeMedian_COXTV'] = summaryTestAliveExtendedDf.index.map(medianSurvivalTestAliveExtended)
    summaryTestAliveExtendedDf['pAlive_COXTV'] = summaryTestAliveExtendedDf.index.map(pAliveTestAliveExtended)
    
    summaryTestDeadDf['TimeMean_COXTV'] = summaryTestDeadDf.index.map(meanSurvivalTestDead)
    summaryTestDeadDf['TimeMedian_COXTV'] = summaryTestDeadDf.index.map(medianSurvivalTestDead)
    summaryTestDeadDf['pAlive_COXTV'] = summaryTestDeadDf.index.map(pAliveTestDead)

    summaryTestAliveDf['TimeMean_COXTV'] = summaryTestAliveDf.index.map(meanSurvivalTestAlive)
    summaryTestAliveDf['TimeMedian_COXTV'] = summaryTestAliveDf.index.map(medianSurvivalTestAlive)
    summaryTestAliveDf['pAlive_COXTV'] = summaryTestAliveDf.index.map(pAliveTestAlive)
    
    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadCOXTV, S_TestAliveCOXTV], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='pAlive_COXTV', 
                         predKind='survival')    
    metricsDict['COXTV_S'] = (ibs, ci)
    
    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadCOXTV, S_TestAliveCOXTV], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMean_COXTV', 
                         predKind='time')    
    metricsDict['COXTV_Mean'] = (ibs, ci)
    
    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadCOXTV, S_TestAliveCOXTV], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_COXTV', 
                         predKind='time')    
    metricsDict['COXTV_Median'] = (ibs, ci)                                             

###############################################################################
# Discrete Time Proportional Odds Model (DTPO)
#   time-varying effects
#   LogisticRegression large

    columnsX = ['r10_FC', 'r10_FM', 'r10_FMC', 'r10_MC',  
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 
        'clumpiness', 'frequency', 'loyaltyState', 
        'moneyDaily', 'moneyMedian', 'moneySum', 
        'pEvent', 'pEventPoisson', 'q05', 'q09', 'q05Poisson', 'q09Poisson']
    
    columnsAlone = []
            
    columnsTS = dummyEncoder.columnsDummy
    
    trainHugeDf, newColumns = makeHugeDTPO(trainDummyDf, columnsX, columnsTS)
    testDeadHugeDf, _ = makeHugeDTPO(testDeadDummyDf, columnsX, columnsTS)
    testAliveHugeDf, _ = makeHugeDTPO(testAliveDummyDf, columnsX, columnsTS)
    
    columnsX = newColumns + columnsAlone    
    columnY = columnEvent
    # class_weight = lib4.getClassWeights(trainHugeDf[columnY].values)
    
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(trainHugeDf[columnsX].values)
    xTestDeadScaled = scaler.transform(testDeadHugeDf[columnsX].values)
    xTestAliveScaled = scaler.transform(testAliveHugeDf[columnsX].values)
    
    xTrainScaled = np.concatenate((xTrainScaled, trainDummyDf[dummyEncoder.columnsDummy].values), axis=1)
    xTestDeadScaled = np.concatenate((xTestDeadScaled, testDeadDummyDf[dummyEncoder.columnsDummy].values), axis=1)
    xTestAliveScaled = np.concatenate((xTestAliveScaled, testAliveDummyDf[dummyEncoder.columnsDummy].values), axis=1)

    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=random_state, solver='newton-cg', max_iter=500, 
        multi_class='auto', verbose=1, warm_start=False, n_jobs=1, l1_ratio=None)
     
    lr.fit(xTrainScaled, trainHugeDf[columnY].values)
    
    yHatTestDead_LR2 = lr.predict_proba(xTestDeadScaled)
    yHatTestAlive_LR2 = lr.predict_proba(xTestAliveScaled)

    summaryTestDeadExtendedDf['H_DTPO_LR2'] = yHatTestDead_LR2[:, 1]
    summaryTestAliveExtendedDf['H_DTPO_LR2'] = yHatTestAlive_LR2[:, 1]    

################################################################################
    # Discrete Time Proportional Odds Model (DTPO)
#   LogisticRegression
    columnsX = ['r10_FC', 'r10_FM', 'r10_FMC', 'r10_MC',  
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 
        'clumpiness', 'frequency', 'loyaltyState', 
        'moneyDaily', 'moneyMedian', 'moneySum', 
        'pEvent', 'pEventPoisson', 'q05', 'q09', 'q05Poisson', 'q09Poisson']
    
    scaler = StandardScaler()    
    xTrainScaled = scaler.fit_transform(trainDummyDf[columnsX].values)
    xTestDeadScaled = scaler.transform(testDeadDummyDf[columnsX].values)
    xTestAliveScaled = scaler.transform(testAliveDummyDf[columnsX].values)
    
    xTrainScaled = np.concatenate((xTrainScaled, trainDummyDf[dummyEncoder.columnsDummy].values), axis=1)
    xTestDeadScaled = np.concatenate((xTestDeadScaled, testDeadDummyDf[dummyEncoder.columnsDummy].values), axis=1)
    xTestAliveScaled = np.concatenate((xTestAliveScaled, testAliveDummyDf[dummyEncoder.columnsDummy].values), axis=1)
        
    columnY = columnEvent    
        
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=random_state, solver='newton-cg', max_iter=500, 
        multi_class='auto', verbose=0, warm_start=False, n_jobs=1, l1_ratio=None)
     
    lr.fit(xTrainScaled, trainDummyDf[columnY].values)

    yHatTestDead_LR = lr.predict_proba(xTestDeadScaled)
    yHatTestAlive_LR = lr.predict_proba(xTestAliveScaled)

    summaryTestDeadExtendedDf['H_DTPO_LR'] = yHatTestDead_LR[:, 1]
    summaryTestAliveExtendedDf['H_DTPO_LR'] = yHatTestAlive_LR[:, 1]

###############################################################################
    # Discrete Time Proportional Odds Model (DTPO)
#   XG
    xg = fitClassic(trainDummyDf[columnsX].values, trainDummyDf[columnY].values, 
        n_jobs=6, verbose=False, eval_set=None, eval_metric='auc')

    yHatTestDead_XG = xg.predict_proba(testDeadDummyDf[columnsX].values)
    yHatTestAlive_XG = xg.predict_proba(testAliveDummyDf[columnsX].values)
        
    summaryTestDeadExtendedDf['H_DTPO_XG'] = yHatTestDead_XG[:, 1].astype(float)
    summaryTestAliveExtendedDf['H_DTPO_XG'] = yHatTestAlive_XG[:, 1].astype(float)      

###############################################################################    
    # Discrete Time Proportional Odds Model (DTPO)
#   RF
    rf = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
        bootstrap=True, oob_score=False, n_jobs=6, random_state=random_state, 
        verbose=1, warm_start=False, class_weight=class_weight, ccp_alpha=0.0, 
        max_samples=None)
                                        
    rf.fit(trainDummyDf[columnsX].values, trainDummyDf[columnY].values)

    yHatTestDead_RF = rf.predict_proba(testDeadDummyDf[columnsX].values)
    yHatTestAlive_RF = rf.predict_proba(testAliveDummyDf[columnsX].values)
    
    summaryTestDeadExtendedDf['H_DTPO_RF'] = yHatTestDead_RF[:, 1].astype(float)
    summaryTestAliveExtendedDf['H_DTPO_RF'] = yHatTestAlive_RF[:, 1].astype(float)
                                        
###############################################################################
#   extract survival function and expected surival time from DTPO models
#   Dead
    S_List, P_List, mapClientToMedianTime, S_TestDead_LR2 = getSurvivalDTPO(summaryTestDeadExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_LR2', columnPeriod=columnStart)
    summaryTestDeadExtendedDf['S_DTPO_LR2'] = S_List
    summaryTestDeadExtendedDf['P_DTPO_LR2'] = P_List
    summaryTestDeadDf['TimeMedian_LR2'] = summaryTestDeadDf.index.map(mapClientToMedianTime)

    S_List, P_List, mapClientToMedianTime, S_TestDead_LR = getSurvivalDTPO(summaryTestDeadExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_LR', columnPeriod=columnStart)
    summaryTestDeadExtendedDf['S_DTPO_LR'] = S_List
    summaryTestDeadExtendedDf['P_DTPO_LR'] = P_List
    summaryTestDeadDf['TimeMedian_LR'] = summaryTestDeadDf.index.map(mapClientToMedianTime)

    S_List, P_List, mapClientToMedianTime, S_TestDead_XG = getSurvivalDTPO(summaryTestDeadExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_XG', columnPeriod=columnStart)
    summaryTestDeadExtendedDf['S_DTPO_XG'] = S_List
    summaryTestDeadExtendedDf['P_DTPO_XG'] = P_List
    summaryTestDeadDf['TimeMedian_XG'] = summaryTestDeadDf.index.map(mapClientToMedianTime)
    
    S_List, P_List, mapClientToMedianTime, S_TestDead_RF = getSurvivalDTPO(summaryTestDeadExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_RF', columnPeriod=columnStart)
    summaryTestDeadExtendedDf['S_DTPO_RF'] = S_List
    summaryTestDeadExtendedDf['P_DTPO_RF'] = P_List
    summaryTestDeadDf['TimeMedian_RF'] = summaryTestDeadDf.index.map(mapClientToMedianTime)
    
#   Alive    
    S_List, P_List, mapClientToMedianTime, S_TestAlive_LR2 = getSurvivalDTPO(summaryTestAliveExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_LR2', columnPeriod=columnStart)
    summaryTestAliveExtendedDf['S_DTPO_LR2'] = S_List
    summaryTestAliveExtendedDf['P_DTPO_LR2'] = P_List
    summaryTestAliveDf['TimeMedian_LR2'] = summaryTestAliveDf.index.map(mapClientToMedianTime)
    
    S_List, P_List, mapClientToMedianTime, S_TestAlive_LR = getSurvivalDTPO(summaryTestAliveExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_LR', columnPeriod=columnStart)
    summaryTestAliveExtendedDf['S_DTPO_LR'] = S_List
    summaryTestAliveExtendedDf['P_DTPO_LR'] = P_List
    summaryTestAliveDf['TimeMedian_LR'] = summaryTestAliveDf.index.map(mapClientToMedianTime)

    S_List, P_List, mapClientToMedianTime, S_TestAlive_XG = getSurvivalDTPO(summaryTestAliveExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_XG', columnPeriod=columnStart)
    summaryTestAliveExtendedDf['S_DTPO_XG'] = S_List
    summaryTestAliveExtendedDf['P_DTPO_XG'] = P_List    
    summaryTestAliveDf['TimeMedian_XG'] = summaryTestAliveDf.index.map(mapClientToMedianTime)

    S_List, P_List, mapClientToMedianTimex, S_TestAlive_RF = getSurvivalDTPO(summaryTestAliveExtendedDf, 
        columnId=columnId, columnHazard='H_DTPO_RF', columnPeriod=columnStart)
    summaryTestAliveExtendedDf['S_DTPO_RF'] = S_List
    summaryTestAliveExtendedDf['P_DTPO_RF'] = P_List    
    summaryTestAliveDf['TimeMedian_RF'] = summaryTestAliveDf.index.map(mapClientToMedianTime)

    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDead_LR2, S_TestAlive_LR2], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_LR2', 
                         predKind='time', 
                         times=deepcopy(S_TestDead_LR2.rowNames))
    metricsDict['LR2_Median'] = (ibs, ci)
    
    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDead_LR, S_TestAlive_LR], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_LR', 
                         predKind='time',
                         times=deepcopy(S_TestDead_LR.rowNames))
    metricsDict['LR_Median'] = (ibs, ci)

    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDead_XG, S_TestAlive_XG], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_XG', 
                         predKind='time',
                         times=deepcopy(S_TestDead_XG.rowNames))
    metricsDict['XG_Median'] = (ibs, ci)

    ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDead_RF, S_TestAlive_RF], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_RF', 
                         predKind='time',
                         times=deepcopy(S_TestDead_RF.rowNames))
    metricsDict['RF_Median'] = (ibs, ci)    
            
###############################################################################
#   Univariate survival models
   
    duration_col = columnDuration
    event_col = columnEvent
    timeLine = list(range(0, int(nStates*5), 1))

    modelsU = ['KaplanMeier', 'Weibull', 'Exponential', 'LogNormal','LogLogistic', 'GeneralizedGamma']
    
    maeExpectedList = []
    binScoreExpectedList = []
    meanSurvivalTimeList = []
    for modelNameU in modelsU:
        # break
        fitterU = libChurn2.makeUnivariateModel(modelNameU)
        fitterU = fitterU.fit(trainDf[duration_col].values,
            trainDf[event_col].values, timeline=timeLine, label=modelNameU, entry=None) 
        
        meanSurvivalTime = fitterU.median_survival_time_
        meanSurvivalTimeList.append(meanSurvivalTime)
        
        print(modelNameU, meanSurvivalTime)
        
        yHatDead = np.full(shape=(len(testDeadDf)), fill_value=meanSurvivalTime)
        yHatAlive = np.full(shape=(len(testAliveDf)), fill_value=meanSurvivalTime)
        
        if np.isinf(yHatDead).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadDf[duration_col].values, yHatDead))
        else:
            maeExpectedList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf[duration_col].values, yHatAlive))


    scoresUnivariateDict = {}
    for i in range(0, len(modelsU), 1):
        modelU = modelsU[i]        
        scoresUnivariateDict[modelU] = {}
        scoresUnivariateDict[modelU]['MAE Dead'] = float(maeExpectedList[i])
        scoresUnivariateDict[modelU]['CI Alive'] = float(binScoreExpectedList[i])
        scoresUnivariateDict[modelU]['Mean survival time'] = float(meanSurvivalTimeList[i])
        
    lib2.saveYaml(os.path.join(resultsSubDir, 'scores Univariate.yaml'), scoresUnivariateDict, sort_keys=False)
               
#   determine baseline: mean survival time balance between CI for alive and MAE for dead (transformed to [0..1])
    # mean = 519
    f1_list = []
    s1_list = []
    s2_list = []
    mae_list = []
    for mean in list(range(0, nStates*5, 1)):
        yHatDead = np.full(shape=(len(testDeadDf)), fill_value=mean)
        yHatAlive = np.full(shape=(len(testAliveDf)), fill_value=mean)  
        mae = mean_absolute_error(testDeadDf[duration_col].values, yHatDead)
        score1 = scoreCDF(testDeadDf[duration_col].values, yHatDead)
        score2 = getConcordanceMetricAlive(testAliveDf[duration_col].values, yHatAlive)        
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
#   last state only
    
    columnsX = ['r10_FC', 'r10_FM', 'r10_FMC', 'r10_MC',  
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 
        'clumpiness', 'frequency', 'loyaltyState', 
        'moneyDaily', 'moneyMedian', 'moneySum', 
        'pEvent', 'pEventPoisson', 'q05', 'q09', 'q05Poisson', 'q09Poisson']
    
    duration_col = columnDuration
    event_col = columnEvent
    
    scaler = StandardScaler()    
    xTrainScaled = scaler.fit_transform(trainDf[columnsX])
    xTrainScaled = pd.DataFrame(xTrainScaled, index=trainDf.index, columns=columnsX)
    xTrainScaled[event_col] = trainDf[event_col]
    xTrainScaled[duration_col] = trainDf[duration_col]

    xTestDeadScaled = scaler.transform(testDeadDf[columnsX])
    xTestDeadScaled = pd.DataFrame(xTestDeadScaled, index=testDeadDf.index, columns=columnsX)
    xTestDeadScaled[event_col] = testDeadDf[event_col]
    xTestDeadScaled[duration_col] = testDeadDf[duration_col]  
    
    xTestAliveScaled = scaler.transform(testAliveDf[columnsX])
    xTestAliveScaled = pd.DataFrame(xTestAliveScaled, index=testAliveDf.index, columns=columnsX)
    xTestAliveScaled[event_col] = testAliveDf[event_col]
    xTestAliveScaled[duration_col] = testAliveDf[duration_col]

    xTestDeadExtendedScaled = scaler.transform(testDeadExtendedDf[columnsX])
    xTestDeadExtendedScaled = pd.DataFrame(xTestDeadExtendedScaled, index=testDeadExtendedDf.index, columns=columnsX)
    xTestDeadExtendedScaled[event_col] = testDeadExtendedDf[event_col]
    xTestDeadExtendedScaled[duration_col] = testDeadExtendedDf[duration_col]  
    
    xTestAliveExtendedScaled = scaler.transform(testAliveExtendedDf[columnsX])
    xTestAliveExtendedScaled = pd.DataFrame(xTestAliveExtendedScaled, index=testAliveExtendedDf.index, columns=columnsX)
    xTestAliveExtendedScaled[event_col] = testAliveExtendedDf[event_col]
    xTestAliveExtendedScaled[duration_col] = testAliveExtendedDf[duration_col]
 
    maxLen = max(mapClientToTS.values()) + 1
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
    timeLine = list(range(0, nStates*5, 1))
    S_TestDeadDict = {}
    S_TestAliveDict = {}
    for aftModelName in tqdm(modelsAFT):
        fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)

        fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col=duration_col, event_col=event_col, 
            ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)

        S_TestDeadDict[aftModelName] =  makeSurvivalFrame( \
            fitterAFT.predict_survival_function(xTestDeadScaled), maxLen, mapClientToTS)

        S_TestAliveDict[aftModelName] = makeSurvivalFrame( \
            fitterAFT.predict_survival_function(xTestAliveScaled), maxLen, mapClientToTS)
    
        testDeadDict = predictAFT(fitterAFT, xTestDeadScaled)
        testAliveDict = predictAFT(fitterAFT, xTestAliveScaled)
        testDeadExtendedDict = predictAFT(fitterAFT, xTestDeadExtendedScaled)
        testAliveExtendedDict = predictAFT(fitterAFT, xTestAliveExtendedScaled)

        pAliveTestDeadList = predictSurvivalProbability(fitterAFT, xTestDeadScaled, list(xTestDeadScaled[duration_col]))
        pAliveTestAliveList = predictSurvivalProbability(fitterAFT, xTestAliveScaled, list(xTestAliveScaled[duration_col]))
        pAliveTestDeadExtendedList = predictSurvivalProbability(fitterAFT, xTestDeadExtendedScaled, list(xTestDeadExtendedScaled[duration_col]))
        pAliveTestAliveExtendedList = predictSurvivalProbability(fitterAFT, xTestAliveExtendedScaled, list(xTestAliveExtendedScaled[duration_col]))

        summaryTestDeadDf['TimeMean_{}'.format(aftModelName)] = testDeadDict['yExpected'].values
        summaryTestDeadDf['TimeMedian_{}'.format(aftModelName)] = testDeadDict['yMedian'].values
        summaryTestDeadDf['pAlive_{}'.format(aftModelName)] = pAliveTestDeadList

        summaryTestAliveDf['TimeMean_{}'.format(aftModelName)] = testAliveDict['yExpected'].values
        summaryTestAliveDf['TimeMedian_{}'.format(aftModelName)] = testAliveDict['yMedian'].values
        summaryTestAliveDf['pAlive_{}'.format(aftModelName)] = pAliveTestAliveList

        summaryTestDeadExtendedDf['TimeMean_{}'.format(aftModelName)] = testDeadExtendedDict['yExpected'].values
        summaryTestDeadExtendedDf['TimeMedian_{}'.format(aftModelName)] = testDeadExtendedDict['yMedian'].values
        summaryTestDeadExtendedDf['pAlive_{}'.format(aftModelName)] = pAliveTestDeadExtendedList
        
        summaryTestAliveExtendedDf['TimeMean_{}'.format(aftModelName)] = testAliveExtendedDict['yExpected'].values
        summaryTestAliveExtendedDf['TimeMedian_{}'.format(aftModelName)] = testAliveExtendedDict['yMedian'].values
        summaryTestAliveExtendedDf['pAlive_{}'.format(aftModelName)] = pAliveTestAliveExtendedList
        
        if np.isinf(testDeadDict['yExpected']).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadDf[duration_col].values, testDeadDict['yExpected']))
        else:
            maeExpectedList.append(np.inf)
        if np.isinf(testDeadDict['yMedian']).sum() == 0:
            maeMedianList.append(mean_absolute_error(testDeadDf[duration_col].values, testDeadDict['yMedian']))
        else:
            maeMedianList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf[duration_col].values, testAliveDict['yExpected']))
        binScoreMedianList.append(getConcordanceMetricAlive(testAliveDf[duration_col].values, testAliveDict['yMedian']))

        scoreExpected1 = scoreCDF(testDeadDf[duration_col].values, testDeadDict['yExpected'])
        scoreExpected2 = getConcordanceMetricAlive(testAliveDf[duration_col].values, testAliveDict['yExpected'])        
        f1Expected = 2*scoreExpected1*scoreExpected2 / (scoreExpected1 + scoreExpected2)
        f1Expected_list.append(f1Expected)

        scoreMedian1 = scoreCDF(testDeadDf[duration_col].values, testDeadDict['yMedian'])
        scoreMedian2 = getConcordanceMetricAlive(testAliveDf[duration_col].values, testAliveDict['yMedian'])        
        f1Median = 2*scoreMedian1*scoreMedian2 / (scoreMedian1 + scoreMedian2)
        f1Median_list.append(f1Median)
        
        ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadDict[aftModelName], S_TestAliveDict[aftModelName]], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMean_{}'.format(aftModelName), 
                         predKind='time')    
        metricsDict['{}_Mean'.format(aftModelName)] = (ibs, ci)  
    
        ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadDict[aftModelName], S_TestAliveDict[aftModelName]], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='TimeMedian_{}'.format(aftModelName), 
                         predKind='time')    
        metricsDict['{}_Median'.format(aftModelName)] = (ibs, ci) 
        
        ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadDict[aftModelName], S_TestAliveDict[aftModelName]], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='pAlive_{}'.format(aftModelName), 
                         predKind='survival')    
        metricsDict['{}_S'.format(aftModelName)] = (ibs, ci) 
        
        
        #   Plot hazard Rate    
        plt.figure(1, figsize=(19,10)) 
        ax = fitterAFT.plot()
        title = 'Hazard rate {}'.format(aftModelName)
        plt.title(title)
        fileName = '{}{}'.format(title, '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close() 
    
        #   Save coefficients to excel    
        summary = fitterAFT.summary    
        summary.to_excel(os.path.join(resultsSubDir, 'Summary {}.xlsx'.format(aftModelName)))

    for i in range(0, len(modelsAFT), 1):
        print('Model: {}'.format(modelsAFT[i]))
        print('Dead mean: {}. Dead median: {}'.format(maeExpectedList[i], maeMedianList[i]))
        print('Alive Mean CI: {}. Alive Median CI: {}'.format(binScoreExpectedList[i], binScoreMedianList[i]))
        print('f1:', f1Expected_list[i])
        print('f1:', f1Median_list[i])

###############################################################################
#   Survival Tree    

    # same features as for COX and AFT

    xTrain, yTrain = get_x_y(xTrainScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestDead, _ = get_x_y(xTestDeadScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestAlive, _ = get_x_y(xTestAliveScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestDeadExtended, _ = get_x_y(xTestDeadExtendedScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestAliveExtended, _ = get_x_y(xTestAliveExtendedScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
   
    maxLen = max(mapClientToTS.values()) + 1
    max_features='auto'
    n_estimators = 500
    max_depth = 5
    min_samples_split = 2
    min_samples_leaf = 1
    n_jobs = 6
    random_state = 101
    maeList = []
    binScoreList = []
    models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
        'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    # models = ['ComponentwiseGradientBoostingSurvivalAnalysis']
    for modelName in tqdm(models):
        
        model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
            max_depth=max_depth, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
            n_jobs=n_jobs, random_state=random_state) 
                        
        model.fit(xTrain, yTrain)
                
        S_TestDeadDict[modelName] = makeSurvivalFrameTree( \
            model.predict_survival_function(xTestDead), list(testDeadDf.index), 
            maxLen, mapClientToTS)

        S_TestAliveDict[modelName] = makeSurvivalFrameTree( \
            model.predict_survival_function(xTestAlive), list(testAliveDf.index), 
            maxLen, mapClientToTS)                
        
        timeTestDead, pAliveTestDead = predictSurvivalTime(model, xTestDead, list(testDeadDf[duration_col].values), probability=0.5)
        timeTestAlive, pAliveTestAlive = predictSurvivalTime(model, xTestAlive, list(testAliveDf[duration_col].values), probability=0.5)
        timeTestDeadExtended, pAliveTestDeadExtended = predictSurvivalTime(model, xTestDeadExtended, list(testDeadExtendedDf[duration_col].values), probability=0.5)
        timeTestAliveExtended, pAliveTestAliveExtended = predictSurvivalTime(model, xTestAliveExtended, list(testAliveExtendedDf[duration_col].values), probability=0.5)
                
        summaryTestDeadDf['Time_{}'.format(modelName)] = timeTestDead
        summaryTestDeadDf['pAlive_{}'.format(modelName)] = pAliveTestDead

        summaryTestAliveDf['Time_{}'.format(modelName)] = timeTestAlive
        summaryTestAliveDf['pAlive_{}'.format(modelName)] = pAliveTestAlive

        summaryTestDeadExtendedDf['Time_{}'.format(modelName)] = timeTestDeadExtended
        summaryTestDeadExtendedDf['pAlive_{}'.format(modelName)] = pAliveTestDeadExtended

        summaryTestAliveExtendedDf['Time_{}'.format(modelName)] = timeTestAliveExtended
        summaryTestAliveExtendedDf['pAlive_{}'.format(modelName)] = pAliveTestAliveExtended
       
        mae = mean_absolute_error(testDeadDf[duration_col].values, timeTestDead)
        binScore = getConcordanceMetricAlive(testAliveDf[duration_col].values, timeTestAlive)
        maeList.append(mae)
        binScoreList.append(binScore)

        ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='pAlive_{}'.format(modelName), 
                         predKind='survival')    
        metricsDict['{}_S'.format(modelName)] = (ibs, ci) 
        
        ibs, ci = getMetrics(trainDuration=trainDf['D_TS'],
                         trainEvent=trainDf['E'],
                         testDataList=[summaryTestDeadDf, summaryTestAliveDf],
                         S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]], 
                         sTestDuration='D_TS', 
                         sTestEvent='E',
                         sPred='Time_{}'.format(modelName), 
                         predKind='time')    
        metricsDict['{}_Median'.format(modelName)] = (ibs, ci) 

    print(maeList)
    print(binScoreList)



        
    
    
##############################################################################    
    #   Plots
    #   Probability of being alive for random clients for all states
    # survival functions for random clients
    
    features = {'pAlive_COXTV': {'c': 'b', 'linestyle': '-', 'label': 'COX Time-varying'},
       'S_DTPO_LR2': {'c': 'g', 'linestyle': '--', 'label': 'LR time-variant'},
       'S_DTPO_LR': {'c': 'g', 'linestyle': '-', 'label': 'LR time-invariant'},
       'S_DTPO_XG': {'c': 'r', 'linestyle': '-', 'label': 'XG'},
       'S_DTPO_RF': {'c': 'r', 'linestyle': '--', 'label': 'RF'},
       'pAlive_WeibullAFT': {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       'pAlive_LogNormalAFT': {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       'pAlive_LogLogisticAFT': {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       'pAlive_CoxPHFitter': {'c': 'b', 'linestyle': '--', 'label': 'COX last state'},
       'pAlive_RandomSurvivalForest': {'c': 'b', 'linestyle': '-.', 'label': 'RandomSurvivalForest'},
       'pAlive_ExtraSurvivalTrees': {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'},
       'pAlive_GradientBoostingSurvivalAnalysis': {'c': 'r', 'linestyle': '-.', 'label': 'GradientBoostingSurvivalAnalysis'},
       'pAlive_ComponentwiseGradientBoostingSurvivalAnalysis': {'c': 'k', 'linestyle': '-.', 'label': 'ComponentwiseGradientBoostingSurvivalAnalysis'}}
    
    featuresDead = {S_TestDeadCOXTV: {'c': 'b', 'linestyle': '-', 'label': 'COX Time-varying'},
       S_TestDead_LR2: {'c': 'g', 'linestyle': '--', 'label': 'LR time-variant'},
       S_TestDead_LR: {'c': 'g', 'linestyle': '-', 'label': 'LR time-invariant'},
       S_TestDead_XG: {'c': 'r', 'linestyle': '-', 'label': 'XG'},
       S_TestDead_RF: {'c': 'r', 'linestyle': '--', 'label': 'RF'},
       S_TestDeadDict['WeibullAFT']: {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       S_TestDeadDict['LogNormalAFT']: {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       S_TestDeadDict['LogLogisticAFT']: {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       S_TestDeadDict['CoxPHFitter']: {'c': 'b', 'linestyle': '--', 'label': 'COX last state'},
       S_TestDeadDict['RandomSurvivalForest']: {'c': 'b', 'linestyle': '-.', 'label': 'RandomSurvivalForest'},
       S_TestDeadDict['ExtraSurvivalTrees']: {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'},
       S_TestDeadDict['GradientBoostingSurvivalAnalysis']: {'c': 'r', 'linestyle': '-.', 'label': 'GradientBoostingSurvivalAnalysis'},
       S_TestDeadDict['ComponentwiseGradientBoostingSurvivalAnalysis']: {'c': 'k', 'linestyle': '-.', 'label': 'ComponentwiseGradientBoostingSurvivalAnalysis'}}
    
    featuresAlive = {S_TestAliveCOXTV: {'c': 'b', 'linestyle': '-', 'label': 'COX Time-varying'},
       S_TestAlive_LR2: {'c': 'g', 'linestyle': '--', 'label': 'LR time-variant'},
       S_TestAlive_LR: {'c': 'g', 'linestyle': '-', 'label': 'LR time-invariant'},
       S_TestAlive_XG: {'c': 'r', 'linestyle': '-', 'label': 'XG'},
       S_TestAlive_RF: {'c': 'r', 'linestyle': '--', 'label': 'RF'},
       S_TestAliveDict['WeibullAFT']: {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       S_TestAliveDict['LogNormalAFT']: {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       S_TestAliveDict['LogLogisticAFT']: {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       S_TestAliveDict['CoxPHFitter']: {'c': 'b', 'linestyle': '--', 'label': 'COX last state'},
       S_TestAliveDict['RandomSurvivalForest']: {'c': 'b', 'linestyle': '-.', 'label': 'RandomSurvivalForest'},
       S_TestAliveDict['ExtraSurvivalTrees']: {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'},
       S_TestAliveDict['GradientBoostingSurvivalAnalysis']: {'c': 'r', 'linestyle': '-.', 'label': 'GradientBoostingSurvivalAnalysis'},
       S_TestAliveDict['ComponentwiseGradientBoostingSurvivalAnalysis']: {'c': 'k', 'linestyle': '-.', 'label': 'ComponentwiseGradientBoostingSurvivalAnalysis'}}        
    
    clients = sample(clientsTestDeadList, 20)
    for client in clients:
        plotProbability(client, summaryTestDeadExtendedDf, features, visitsDict, 
            lastState.birthStateDict, mapClientToLastState, dZero, prefix='lost', 
            plotPath=plotLostProbaPath)
        
        plotSurvival(client, featuresDead, lastState.birthStateDict, 
            mapClientToLastState, visitsDict, prefix='lost', plotPath=plotLostSurvPath)

    clients = sample(clientsTestAliveList, 20)
    for client in clients:
        plotProbability(client, summaryTestAliveExtendedDf, features, visitsDict, 
            lastState.birthStateDict, mapClientToLastState, dZero, prefix='active', 
            plotPath=plotActiveProbaPath)

        plotSurvival(client, featuresAlive, lastState.birthStateDict, 
            mapClientToLastState, visitsDict, prefix='active', plotPath=plotActiveSurvPath)
    
###############################################################################
#   Scores    
    predictors = ['TimeMean_COXTV', 'TimeMedian_COXTV',
       'TimeMean_WeibullAFT', 'TimeMedian_WeibullAFT',
       'TimeMean_LogNormalAFT', 'TimeMedian_LogNormalAFT',
       'TimeMean_LogLogisticAFT',
       'TimeMedian_LogLogisticAFT', 
       'TimeMean_CoxPHFitter', 'TimeMedian_CoxPHFitter',
       'Time_RandomSurvivalForest', 
       'Time_ExtraSurvivalTrees', 
       'Time_GradientBoostingSurvivalAnalysis',
       'Time_ComponentwiseGradientBoostingSurvivalAnalysis',
       'TimeMedian_LR2',
       'TimeMedian_LR',
       'TimeMedian_XG',
       'TimeMedian_RF']

    maeDict = getMAE_Dead(summaryTestDeadDf, duration_col, predictors)
    maeExtendedDict = getMAE_Dead(summaryTestDeadExtendedDf, duration_col, predictors)

    ciDict = getCI_Alive(summaryTestAliveDf, duration_col, predictors)
    ciExtendedDict = getCI_Alive(summaryTestAliveExtendedDf, duration_col, predictors)


###############################################################################
#   Forward models


    
    mapClientToEventsList = getMapClientToEvents(transactionsDf)
    mapClientToDeathList = getMapClientToDeath(statesDict, mapClientToEventsList)

    trainForwardDf = buildPersonPeriodForwardFrame(clientsTrainList, statesDict, mapClientToDeathList)
    trainForwardDf = trainForwardDf[trainForwardDf[columnEvent] != -1]
    trainForwardDf = trainForwardDf[trainForwardDf[columnDurationForward] > 0]
   
    testDeadForwardDf = buildPersonPeriodForwardFrame(clientsTestDeadList, statesDict, mapClientToDeathList)
    testAliveForwardDf = buildPersonPeriodForwardFrame(clientsTestAliveList, statesDict, mapClientToDeathList)
          
    columnsPredict = [columnEvent, columnDurationForward, 'frequency', '_client', '_date']
    summaryTestDeadForwardDf = testDeadForwardDf[columnsPredict].copy(deep=True)
    summaryTestAliveForwardDf = testAliveForwardDf[columnsPredict].copy(deep=True)
    
    columnsX = ['r10_FC', 'r10_FM', 'r10_FMC', 'r10_MC',  
        'trendShort_FC', 'trendShort_FM', 'trendShort_FMC', 'trendShort_MC',
        'trend_FC', 'trend_FM', 'trend_FMC', 'trend_MC', 
        'clumpiness', 'frequency', 
        'moneyDaily', 'moneyMedian', 'moneySum', 
        'pEvent', 'pEventPoisson', 'q05', 'q09', 'q05Poisson', 'q09Poisson']
    
    duration_col = columnDurationForward
    event_col = columnEvent
    
    scaler = StandardScaler()    
    xTrainScaled = scaler.fit_transform(trainForwardDf[columnsX])
    xTrainScaled = pd.DataFrame(xTrainScaled, index=trainForwardDf.index, columns=columnsX)
    xTrainScaled[event_col] = trainForwardDf[event_col]
    xTrainScaled[duration_col] = trainForwardDf[duration_col]

    xTestDeadScaled = scaler.transform(testDeadForwardDf[columnsX])
    xTestDeadScaled = pd.DataFrame(xTestDeadScaled, index=testDeadForwardDf.index, columns=columnsX)
    xTestDeadScaled[event_col] = testDeadForwardDf[event_col]
    xTestDeadScaled[duration_col] = testDeadForwardDf[duration_col]  
    
    xTestAliveScaled = scaler.transform(testAliveForwardDf[columnsX])
    xTestAliveScaled = pd.DataFrame(xTestAliveScaled, index=testAliveForwardDf.index, columns=columnsX)
    xTestAliveScaled[event_col] = testAliveForwardDf[event_col]
    xTestAliveScaled[duration_col] = testAliveForwardDf[duration_col]

    maxLen = max(mapClientToTS.values()) + 1
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
    timeLine = list(range(0, nStates*3, 1))

    for aftModelName in tqdm(modelsAFT):
        modelName = '{}_Forward'.format(aftModelName)
        fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)

        fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col=duration_col, event_col=event_col, 
            ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)

# same CI
        # fitterAFT.score(pd.concat([xTestDeadScaled, xTestAliveScaled], axis=0), scoring_method='concordance_index')

        S_TestDeadDict[modelName] =  makeSurvivalFrameForward(fitterAFT,
            xTestDeadScaled, duration_col, timeLine)

        S_TestAliveDict[modelName] =  makeSurvivalFrameForward(fitterAFT,
            xTestAliveScaled, duration_col, timeLine)
            
        testDeadDict = predictAFT(fitterAFT, xTestDeadScaled)
        testAliveDict = predictAFT(fitterAFT, xTestAliveScaled)

        pAliveTestDeadList = predictSurvivalProbability(fitterAFT, xTestDeadScaled, list(xTestDeadScaled[duration_col]))
        pAliveTestAliveList = predictSurvivalProbability(fitterAFT, xTestAliveScaled, list(xTestAliveScaled[duration_col]))

        summaryTestDeadForwardDf['TimeMean_{}'.format(modelName)] = testDeadDict['yExpected'].values
        summaryTestDeadForwardDf['TimeMedian_{}'.format(modelName)] = testDeadDict['yMedian'].values
        summaryTestDeadForwardDf['pAlive_{}'.format(modelName)] = pAliveTestDeadList

        summaryTestAliveForwardDf['TimeMean_{}'.format(modelName)] = testAliveDict['yExpected'].values
        summaryTestAliveForwardDf['TimeMedian_{}'.format(modelName)] = testAliveDict['yMedian'].values
        summaryTestAliveForwardDf['pAlive_{}'.format(modelName)] = pAliveTestAliveList
      
        if np.isinf(testDeadDict['yExpected']).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadForwardDf[duration_col].values, testDeadDict['yExpected']))
        else:
            maeExpectedList.append(np.inf)
        if np.isinf(testDeadDict['yMedian']).sum() == 0:
            maeMedianList.append(mean_absolute_error(testDeadForwardDf[duration_col].values, testDeadDict['yMedian']))
        else:
            maeMedianList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveForwardDf[duration_col].values, testAliveDict['yExpected']))
        binScoreMedianList.append(getConcordanceMetricAlive(testAliveForwardDf[duration_col].values, testAliveDict['yMedian']))

        scoreExpected1 = scoreCDF(testDeadForwardDf[duration_col].values, testDeadDict['yExpected'])
        scoreExpected2 = getConcordanceMetricAlive(testAliveForwardDf[duration_col].values, testAliveDict['yExpected'])        
        f1Expected = 2*scoreExpected1*scoreExpected2 / (scoreExpected1 + scoreExpected2)
        f1Expected_list.append(f1Expected)

        scoreMedian1 = scoreCDF(testDeadForwardDf[duration_col].values, testDeadDict['yMedian'])
        scoreMedian2 = getConcordanceMetricAlive(testAliveForwardDf[duration_col].values, testAliveDict['yMedian'])        
        f1Median = 2*scoreMedian1*scoreMedian2 / (scoreMedian1 + scoreMedian2)
        f1Median_list.append(f1Median)
        
        ibs, ci = getMetrics(trainDuration=trainForwardDf[duration_col],
                         trainEvent=trainForwardDf[event_col],
                         testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf],
                         S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]],
                         sTestDuration=duration_col,
                         sTestEvent=event_col,
                         sPred='TimeMean_{}'.format(modelName),
                         predKind='time')
        metricsDict['{}_Mean'.format(modelName)] = (ibs, ci)  
    
        ibs, ci = getMetrics(trainDuration=trainForwardDf[duration_col],
                         trainEvent=trainForwardDf[event_col],
                         testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf],
                         S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]], 
                         sTestDuration=duration_col, 
                         sTestEvent=event_col,
                         sPred='TimeMedian_{}'.format(modelName), 
                         predKind='time')    
        metricsDict['{}_Median'.format(modelName)] = (ibs, ci) 
        
        ibs, ci = getMetrics(trainDuration=trainForwardDf[duration_col],
                         trainEvent=trainForwardDf[event_col],
                         testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf],
                         S_List=[S_TestDeadDict[modelName], S_TestAliveDict[modelName]], 
                         sTestDuration=duration_col, 
                         sTestEvent=event_col,
                         sPred='pAlive_{}'.format(modelName), 
                         predKind='survival')    
        metricsDict['{}_S'.format(modelName)] = (ibs, ci) 
        
    
        #   Plot hazard Rate    
        plt.figure(1, figsize=(19,10)) 
        ax = fitterAFT.plot()
        title = 'Hazard rate {}'.format(modelName)
        plt.title(title)
        fileName = '{}{}'.format(title, '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close() 
    
        #   Save coefficients to excel    
        summary = fitterAFT.summary    
        summary.to_excel(os.path.join(resultsSubDir, 'Summary {}.xlsx'.format(modelName)))

    for i in range(0, len(modelsAFT), 1):
        print('Model: {}'.format(modelsAFT[i]))
        print('MAE Dead mean: {}. MAE Dead median: {}'.format(maeExpectedList[i], maeMedianList[i]))
        print('Alive Mean CI: {}. Alive Median CI: {}'.format(binScoreExpectedList[i], binScoreMedianList[i]))
        print('f1Expected:', f1Expected_list[i])
        print('f1Median:', f1Median_list[i])

    

###############################################################################
       
        
#   Survival Tree    

    # same features as for COX and AFT

    xTrain, yTrain = get_x_y(xTrainScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestDead, _ = get_x_y(xTestDeadScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
    xTestAlive, _ = get_x_y(xTestAliveScaled, attr_labels=[event_col, duration_col], pos_label=1, survival=True)
   
    max_features='auto'
    n_estimators = 100
    max_depth = 3
    min_samples_split = 20
    min_samples_leaf = 10
    n_jobs = 6
    random_state = 101
    maeList = []
    binScoreList = []
    # models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
        # 'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']
    models = ['ExtraSurvivalTrees']
    for modelName in tqdm(models):
        modelName_ = '{}_Forward'.format(modelName)
        model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
            max_depth=max_depth, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
            n_jobs=n_jobs, random_state=random_state) 
                        
        model.fit(xTrain, yTrain)
                
        S_TestDeadDict[modelName_] = makeSurvivalFrameTreeForward(model, xTestDead)
        S_TestAliveDict[modelName_] = makeSurvivalFrameTreeForward(model, xTestAlive)

        timeTestDead, pAliveTestDead = predictSurvivalTime(model, xTestDead, list(xTestDeadScaled[duration_col].values), probability=0.5)
        timeTestAlive, pAliveTestAlive = predictSurvivalTime(model, xTestAlive, list(xTestAliveScaled[duration_col].values), probability=0.5)
                
        summaryTestDeadForwardDf['Time_{}'.format(modelName_)] = timeTestDead
        summaryTestDeadForwardDf['pAlive_{}'.format(modelName_)] = pAliveTestDead

        summaryTestAliveForwardDf['Time_{}'.format(modelName_)] = timeTestAlive
        summaryTestAliveForwardDf['pAlive_{}'.format(modelName_)] = pAliveTestAlive
      
        mae = mean_absolute_error(xTestDeadScaled[duration_col].values, timeTestDead)
        binScore = getConcordanceMetricAlive(xTestAliveScaled[duration_col].values, timeTestAlive)
        maeList.append(mae)
        binScoreList.append(binScore)

        ibs, ci = getMetrics(trainDuration=trainForwardDf[duration_col],
                         trainEvent=trainForwardDf[event_col],
                         testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf],
                         S_List=[S_TestDeadDict[modelName_], S_TestAliveDict[modelName_]], 
                         sTestDuration=duration_col, 
                         sTestEvent=event_col,
                         sPred='pAlive_{}'.format(modelName_), 
                         predKind='survival',
                         times=model.event_times_)
        metricsDict['{}_S'.format(modelName_)] = (ibs, ci)        
        
        ibs, ci = getMetrics(trainDuration=trainForwardDf[duration_col],
                         trainEvent=trainForwardDf[event_col],
                         testDataList=[summaryTestDeadForwardDf, summaryTestAliveForwardDf],
                         S_List=[S_TestDeadDict[modelName_], S_TestAliveDict[modelName_]], 
                         sTestDuration=duration_col, 
                         sTestEvent=event_col,
                         sPred='Time_{}'.format(modelName_), 
                         predKind='time',
                         times=model.event_times_)    
        metricsDict['{}_Median'.format(modelName_)] = (ibs, ci) 

    print(maeList)
    print(binScoreList)
        
    
    
    featuresForward = {
       'pAlive_WeibullAFT_Forward': {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       'pAlive_LogNormalAFT_Forward': {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       'pAlive_LogLogisticAFT_Forward': {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       'pAlive_CoxPHFitter_Forward': {'c': 'b', 'linestyle': '--', 'label': 'COX'},
       'pAlive_ExtraSurvivalTrees_Forward': {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'}}
    
    featuresDeadForward = {
       S_TestDeadDict['WeibullAFT_Forward']: {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       S_TestDeadDict['LogNormalAFT_Forward']: {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       S_TestDeadDict['LogLogisticAFT_Forward']: {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       S_TestDeadDict['CoxPHFitter_Forward']: {'c': 'b', 'linestyle': '--', 'label': 'COX'},
       S_TestDeadDict['ExtraSurvivalTrees_Forward']: {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'}}
    
    featuresAliveForward = {
       S_TestAliveDict['WeibullAFT_Forward']: {'c': 'b', 'linestyle': ':', 'label': 'Weibull AFT'},
       S_TestAliveDict['LogNormalAFT_Forward']: {'c': 'g', 'linestyle': ':', 'label': 'LogNormal AFT'},
       S_TestAliveDict['LogLogisticAFT_Forward']: {'c': 'r', 'linestyle': ':', 'label': 'LogLogistic AFT'},
       S_TestAliveDict['CoxPHFitter_Forward']: {'c': 'b', 'linestyle': '--', 'label': 'COX'},
       S_TestAliveDict['ExtraSurvivalTrees_Forward']: {'c': 'g', 'linestyle': '-.', 'label': 'ExtraSurvivalTrees'}}
        

    # summaryTestDeadForwardDf['_client'], summaryTestDeadForwardDf['_date'] = splitIndex2(summaryTestDeadForwardDf)
    # summaryTestAliveForwardDf['_client'], summaryTestAliveForwardDf['_date'] = splitIndex2(summaryTestAliveForwardDf)

    clients = sample(clientsTestDeadList, 20)
    for client in clients:
        plotProbability(client, summaryTestDeadForwardDf, featuresForward, visitsDict, 
            lastState.birthStateDict, mapClientToLastState, dZero, prefix='lostF', 
            plotPath=plotForwardLostProbaPath)
        
        plotSurvivalForward(client, featuresDeadForward, lastState.birthStateDict, 
            mapClientToLastState, visitsDict, prefix='lostF', 
            plotPath=plotForwardLostSurvPath)

    clients = sample(clientsTestAliveList, 20)
    for client in clients:
        plotProbability(client, summaryTestAliveForwardDf, featuresForward, visitsDict, 
            lastState.birthStateDict, mapClientToLastState, dZero, prefix='activeF', 
            plotPath=plotForwardActiveProbaPath)

        plotSurvivalForward(client, featuresAliveForward, lastState.birthStateDict, 
            mapClientToLastState, visitsDict, prefix='activeF', 
            plotPath=plotForwardActiveSurvPath)
    
    # for key in featuresDead.keys():
        # break
    
    predictors = ['TimeMean_WeibullAFT_Forward', 'TimeMedian_WeibullAFT_Forward',
       'TimeMean_LogNormalAFT_Forward', 'TimeMedian_LogNormalAFT_Forward',
       'TimeMean_LogLogisticAFT_Forward', 'TimeMedian_LogLogisticAFT_Forward',
       'TimeMean_CoxPHFitter_Forward',
       'TimeMedian_CoxPHFitter_Forward', 
       'Time_ExtraSurvivalTrees_Forward']
    
    maeForwardDict = getMAE_Dead(summaryTestDeadForwardDf, duration_col, predictors)
    ciForwardDict = getCI_Alive(summaryTestAliveForwardDf, duration_col, predictors)

    summaryTestDeadForward1Df = summaryTestDeadForwardDf.drop_duplicates(subset=['_client'], keep='first')
    maeForward1Dict = getMAE_Dead(summaryTestDeadForward1Df, duration_col, predictors)

    summaryTestAliveForward1Df = summaryTestAliveForwardDf.drop_duplicates(subset=['_client'], keep='first')
    ciForward1Dict = getCI_Alive(summaryTestAliveForward1Df, duration_col, predictors)

    
    metrics1Dict = {key: {'IBS': float(value[0]), 'CI': float(value[1])} for key, value in metricsDict.items()}

    lib2.saveYaml(os.path.join(resultsSubDir, 'metrics.yaml'), metrics1Dict, sort_keys=False)




# date_string = "2012-12-15"
# a = datetime.fromisoformat(date_string).date()



# def splitIndex2(data):
#     idList = list(data.index)
#     clients = []
#     dates = []
#     for x in idList:
#         y = x.split(' | ')
#         clients.append(y[0])
#         dates.append(datetime.fromisoformat(y[1]).date())
#     return clients, dates
    

    
    # import csv 
    

    