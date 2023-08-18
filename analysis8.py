
# COX and LR Jul 2022

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

def getFeatures(state, client):
    """
    """
    featuresDict = {}
    featuresDict['C'] = state.r10_C.get(client)
    featuresDict['F'] = state.r10_Frequency.get(client)
    featuresDict['M'] = state.r10_MoneyDailyStep.get(client)       
    featuresDict['r10_FM'] = state.r10_FM.get(client)
    featuresDict['r10_MC'] = state.r10_MC.get(client)
    featuresDict['r10_FC'] = state.r10_FC.get(client)
    featuresDict['r10_FMC'] = state.r10_FMC.get(client)                               
    featuresDict['trendShort_FM'] = state.trendShort_FM.get(client, 0)
    featuresDict['trendShort_MC'] = state.trendShort_MC.get(client, 0)
    featuresDict['trendShort_FC'] = state.trendShort_FC.get(client, 0)
    featuresDict['trendShort_FMC'] = state.trendShort_FMC.get(client, 0)
    featuresDict['trend_FM'] = state.trend_FM.get(client, 0)
    featuresDict['trend_MC'] = state.trend_MC.get(client, 0)
    featuresDict['trend_FC'] = state.trend_FC.get(client, 0)
    featuresDict['trend_FMC'] = state.trend_FMC.get(client, 0)
    if not State.checkDictForNaN(featuresDict):
        raise ValueError('NaN in features, client: {}, date: {}'.format(client, state.dState))
    return featuresDict  
    
def build_TV_data1(client):
    dStateBirth = stateLast.birthStateDict[client]    
    D, E, dStateDeath = mapClientToStatus[client]
    rows = sorted([x for x in dStateList if x >= dStateBirth and x <= dStateDeath])
    if len(rows) < 2:
        return None
    
    featuresDict = getFeatures(stateLast, client)
    columns = ['id', 'start', 'stop', 'E'] + list(featuresDict.keys())

    frame = FrameList(rowNames=rows[1:], columnNames=columns, 
        df=None, dtype=None)
    
    frame.fillColumn('id', client)
    frame.fillColumn('E', 0)
    frame.put(rows[-1], 'E', E) # last row-state status
    
    for i, dState in enumerate(rows):
        if i == 0:
            continue
        frame.put(dState, 'start', i - 1)
        frame.put(dState, 'stop', i)
        featuresDict = getFeatures(statesDict[dState], client)        
        frame.fillRow(dState, featuresDict)
        
    df = frame.to_pandas()
    df['dState'] = df.index
    columns.insert(1, 'dState')
    df = df[columns]
    df.index = df['id'] + '_' + df['dState'].astype(str)
    # df.reset_index(drop=True, inplace=True)
    return df

def buildPersonPeriodFrame(clients):
    dataList = []
    for client in tqdm(clients):
        df1 = build_TV_data1(client)
        if df1 is not None:
            dataList.append(df1)    
    df = pd.concat(dataList, axis=0)
    # df.reset_index(drop=True, inplace=True)
    return df
    
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

def makeExtended_D_Dict(dataExtended, columnId, columnD, D_Dict):
    tmp = dataExtended[[columnId]].copy(deep=True)
    tmp[columnD] = tmp[columnId].map(D_Dict)
    tmp[columnId] = tmp.index
    return tmp[columnD].to_dict()

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

def getMeanSurvival(S):
    meanSurvivalTime = S.array.sum(axis=0)
    mapClientToMeanSurvival = {}
    for i in range(0, meanSurvivalTime.shape[0], 1):
        client = S.columnNames[i]
        mapClientToMeanSurvival[client] = int(meanSurvivalTime[i])
    return mapClientToMeanSurvival

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
        # if periods[0] != 0:
        #     print(client)            
        #     raise RuntimeError('2')
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
    return {'S': S_List, 'P': P_List, 'Time': mapClientToMedianTime, 'Frame': S}

###############################################################################    
if __name__ == "__main__":

    TEST_DEAD_FRACTION = 0.25
    TEST_ALIVE_FRACTION = 0.15
    random_state = 101

    sData = 'casa_2' # CDNOW casa_2 plex_45

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
    stateLast = statesDict[dStateLast]

    dStateReverseList = sorted(dStateList, reverse=True)
    clientsAllList = stateLast.clientsAll

    mapClientToStatus = {}
#   dead clients, get death time and durtion
    clientsDeadList = []
    for client, dStateDeath in stateLast.deathStateDict.items():
        D = statesDict[dStateDeath].D_TS_Dict.get(client)
        if D is None:
            raise ValueError(client)
        mapClientToStatus[client] = (D, 1, dStateDeath)
        clientsDeadList.append(client)

#   alive clients, get duration
    clientsAliveList = []
    for client in stateLast.clientsCensored:
        D = stateLast.D_TS_Dict.get(client)
        if D is None:
            raise ValueError(client)
        mapClientToStatus[client] = (D, 0, dStateLast)    
        clientsAliveList.append(client)
    
#   split clients
    clientsTestDeadList = sample(clientsDeadList, int(len(clientsDeadList) * TEST_DEAD_FRACTION))
    clientsTestAliveList = sample(clientsAliveList, int(len(clientsAliveList) * TEST_ALIVE_FRACTION))    
    clientsTrainList = sorted(list(set(clientsAllList).difference(set(clientsTestDeadList))))
    clientsTrainList = sorted(list(set(clientsTrainList).difference(set(clientsTestAliveList))))
     
# build person-period frames for COX time-varying regression
    trainDf = buildPersonPeriodFrame(clientsTrainList)
    testDeadDf = buildPersonPeriodFrame(clientsTestDeadList)
    testAliveDf = buildPersonPeriodFrame(clientsTestAliveList)

    columnsX = ['C', 'F', 'M', 'r10_FM', 'r10_MC',
       'r10_FC', 'r10_FMC', 'trendShort_FM', 'trendShort_MC', 'trendShort_FC',
       'trendShort_FMC', 'trend_FM', 'trend_MC', 'trend_FC', 'trend_FMC']
    
    # trainDf.columns
 
#   scale features    
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(trainDf[columnsX])
    xTrain = pd.DataFrame(xTrain, index=trainDf.index, columns=columnsX)
    xTrain['E'] = trainDf['E']
    xTrain['id'] = trainDf['id']
    xTrain['start'] = trainDf['start']
    xTrain['stop'] = trainDf['stop']

    xTestDead = scaler.transform(testDeadDf[columnsX])
    xTestDead = pd.DataFrame(xTestDead, index=testDeadDf.index, columns=columnsX)
    xTestDead['id'] = testDeadDf['id']
    xTestDead['E'] = testDeadDf['E']
    xTestDead['start'] = testDeadDf['start']
    xTestDead['stop'] = testDeadDf['stop']    
    xTestDead['ID'] = testDeadDf['id'] + '_' + testDeadDf['dState'].astype(str)

    xTestAlive = scaler.transform(testAliveDf[columnsX])
    xTestAlive = pd.DataFrame(xTestAlive, index=testAliveDf.index, columns=columnsX)
    xTestAlive['id'] = testAliveDf['id']    
    xTestAlive['E'] = testAliveDf['E']  
    xTestAlive['start'] = testAliveDf['start']
    xTestAlive['stop'] = testAliveDf['stop']      
    xTestAlive['ID'] = testAliveDf['id'] + '_' + testAliveDf['dState'].astype(str)
    
#   COX time-varying regression fit
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(xTrain, id_col='id', event_col='E', 
        start_col='start', stop_col='stop', show_progress=True, fit_options={'step_size': 0.1})       
    coxSummary = ctv.summary


#   Plot hazard Rate    
    plt.figure(1, figsize=(19,10)) 
    ax = ctv.plot()
    title = 'Hazard rate Cox time-varying'
    plt.title(title)
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    

#   make data for LR    
    dummyEncoder = DummyEncoderRobust(unknownValue=0, prefix_sep='_')
    prefix = 'ts'
    trainDummyDf = dummyEncoder.fit_transform(trainDf, 'start', prefix=prefix, drop=False)
    testDeadDummyDf = dummyEncoder.transform(testDeadDf, drop=False)
    testAliveDummyDf = dummyEncoder.transform(testAliveDf, drop=False)
    columnsDummy = list(dummyEncoder.columnsDummy)

    scaler = StandardScaler()    
    xTrainLR = scaler.fit_transform(trainDummyDf[columnsX].values)
    xTestDeadLR = scaler.transform(testDeadDummyDf[columnsX].values)
    xTestAliveLR = scaler.transform(testAliveDummyDf[columnsX].values)
    
    xTrainLR = np.concatenate((xTrainLR, trainDummyDf[columnsDummy].values), axis=1)
    xTestDeadLR = np.concatenate((xTestDeadLR, testDeadDummyDf[columnsDummy].values), axis=1)
    xTestAliveLR = np.concatenate((xTestAliveLR, testAliveDummyDf[columnsDummy].values), axis=1)
    
    class_weight = None
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=random_state, solver='newton-cg', max_iter=500, 
        multi_class='auto', verbose=0, warm_start=False, n_jobs=1, l1_ratio=None)
     
    lr.fit(xTrainLR, trainDummyDf['E'].values)

    yHatTestDead_LR = lr.predict_proba(xTestDeadLR)
    yHatTestAlive_LR = lr.predict_proba(xTestAliveLR)
   
    summaryTestDeadDf = testDeadDf[['id', 'start']].copy(deep=True)
    summaryTestDeadDf['ID'] = testDeadDf.index
    summaryTestDeadDf['H'] = yHatTestDead_LR[:, 1]
    
    a = getSurvivalDTPO(summaryTestDeadDf, 'ID', 'H', 'start')




    # columnsTS = dummyEncoder.columnsDummy

    # trainHugeDf, newColumns = makeHugeDTPO(trainDummyDf, columnsX, columnsTS)


    # S_TestDead, H_TestDead = getSurvivalHazard(ctv, xTestDead, columnId='ID')
    # S_TestAlive, H_TestAlive = getSurvivalHazard(ctv, xTestAlive, columnId='ID')
    
    # S = S_TestDead
    # a = getMeanSurvival(S_TestDead)
    
    # S_TestDeadDf = S_TestDead.to_pandas()
    
    # meanSurvivalTestDead = getMeanSurvival(S_TestDead)
    # meanSurvivalTestAlive = getMeanSurvival(S_TestAlive)
    
    # testDeadD_Dict = makeExtended_D_Dict(testDeadDf, 'id', 'D', stateLast.D_TS_Dict)
    # testAliveD_Dict = makeExtended_D_Dict(testAliveDf, 'id', 'D', stateLast.D_TS_Dict)

    # medianSurvivalTestDead, pAliveTestDead = getMedianSurvival(S_TestDead, D_Dict=testDeadD_Dict)
    # medianSurvivalTestAlive, pAliveTestAlive = getMedianSurvival(S_TestAlive, D_Dict=testAliveD_Dict)

    # make index as in old version


# def getSurvivalHazard2(fitter, data):
#     h0 = fitter.baseline_cumulative_hazard_ # for each time step
#     h0 = h0['baseline hazard'].to_dict() # key is stop, value - baseline hazard rate for this step
#     # s0 = fitter.baseline_survival_ # for each time step
#     h = fitter.predict_partial_hazard(data)
#     # logH = ctv.predict_log_partial_hazard(xTestDead)

#     rows = [x for x in range(min(h0.keys()), max(h0.keys()) + 1, 1)]    
#     columns = sorted(list(data['id'].unique()))
    
#     clientsList = list(data['id'].values)
#     stopList = list(data['stop'].values)
    
#     nanArray = np.empty(shape=(len(rows), len(columns)), dtype=float)
#     nanArray[:] = np.nan
#     H = FrameArray(rowNames=rows, columnNames=columns, df=nanArray, dtype=float)
#     S = FrameArray(rowNames=rows, columnNames=columns, df=nanArray, dtype=float)

#     for i in range(0, len(clientsList), 1):
#         client = clientsList[i]
#         stop = stopList[i]
#         baseH = h0.get(stop)
#         if baseH is None:
#             # missing baseline hazard rate
#             continue
#         partialH = h[i]
#         hazard = float(baseH * partialH)
#         H.put(stop, client, hazard)
#         survival = float(exp(-hazard))
#         S.put(stop, client, survival)        
    
#     return H, S
 
    
 
    
 
    # H_TestDead, S_TestDead = getSurvivalHazard2(ctv, xTestDead)
    # H_TestAlive, S_TestAlive = getSurvivalHazard2(ctv, xTestAlive)
 
    
 
    # log8 = sum(logH[:8])
    # partial8 = exp(log8)
    
    
    # exp(-2.74594*0.00458885)
    # exp(-2.23828*0.102694)
    
    h0 = ctv.baseline_cumulative_hazard_ # for each time step
    h0 = h0['baseline hazard'].to_dict() # key is stop, value - baseline hazard rate for this step
    s0 = ctv.baseline_survival_ # for each time step
    # F0 = 1 - s0
    # h = fitter.predict_partial_hazard(data)
    # logH = ctv.predict_log_partial_hazard(xTestDead)
    
    
    # # F = 1 - S
    
    # from scipy.stats import expon
    
    # x = np.linspace(expon.ppf(0.01),
    #             expon.ppf(0.99), 100)
    
    # x = np.linspace(0.01, 5, 500)
    # y = expon.pdf(x) # known
    
    # fig, ax = plt.subplots(1, 1)
    
    # ax.plot(x, y, 'r-', lw=2, alpha=0.6, label='expon pdf')
    
    # r = expon.rvs(size=1000)
    
    
    
    # expon.fit(r)
    
    
    
    # exp(-0.001)
    
    # from scipy.optimize import curve_fit
    
    
    # # def func(x, a, b, c):
    # #     return 1 - a * np.exp(-b * x) + c
    
    # def func(x, a, b, c):
    #     return a * np.exp(-b * x) + c
    
    # x = np.array(s0.index)
    # y = np.array(s0['baseline survival'].values)
    
    # popt, pcov = curve_fit(func, x, y)
    
    # x2 = np.arange(min(x), max(x)*2)
    # y2 = func(x2, popt[0], popt[1], popt[2])

    # plt.plot(x, y)
    # plt.plot(x2, y2)




    