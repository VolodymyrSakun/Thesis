investigate

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

#import numpy as np
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
    
def fitAFT(fitterAFT, trainDf, duration_col='T', event_col='E', columnsXY=None, 
    ancillary=True, fit_left_censoring=False):
    
    if columnsXY is None:
        columnsXY = list(trainDf.columns)
        
    if isinstance(fitterAFT, lifelines.AalenAdditiveFitter) or \
        isinstance(fitterAFT, lifelines.fitters.coxph_fitter.CoxPHFitter):
        fitterAFT.fit(trainDf[columnsXY], duration_col=duration_col, 
                                 event_col=event_col)
    elif isinstance(fitterAFT, lifelines.fitters.generalized_gamma_regression_fitter.GeneralizedGammaRegressionFitter):        
        if fit_left_censoring:
            fitterAFT.fit_left_censoring(trainDf[columnsXY], duration_col=duration_col, 
                             event_col=event_col)
        else:
            fitterAFT.fit(trainDf[columnsXY], duration_col=duration_col, 
                                     event_col=event_col)
    else:
        if fit_left_censoring:
            fitterAFT.fit_left_censoring(trainDf[columnsXY], duration_col=duration_col, 
                                     event_col=event_col, ancillary=ancillary)
        else:
            fitterAFT.fit(trainDf[columnsXY], duration_col=duration_col, 
                                     event_col=event_col, ancillary=ancillary)        

    return fitterAFT

def predictAFT(fitterAFT, testDf, columnsXY=None):
    if columnsXY is None:
        columnsXY = list(testDf.columns)
    
    if isinstance(fitterAFT, lifelines.fitters.coxph_fitter.CoxPHFitter):
        aic = fitterAFT.AIC_partial_
    elif isinstance(fitterAFT, lifelines.AalenAdditiveFitter):
        aic = np.nan
    else:
        aic = fitterAFT.AIC_                        
    
    y_pred_exp = fitterAFT.predict_expectation(testDf[columnsXY])
    y_pred_exp.fillna(y_pred_exp.mean(), inplace=True)
                    
    y_pred_median = fitterAFT.predict_median(testDf[columnsXY])
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

def predictSurvivalTime(estimator, xTest, probability=0.5):
    s = estimator.predict_survival_function(xTest)    
    yPred = np.zeros(shape=(s.shape[0]), dtype=int)
    for i in range(0, s.shape[0], 1):
        s1 = s[i]
        timeline1 = s1.x
        pAlive = s1.y
        idx = (np.abs(pAlive - probability)).argmin()
        yPred[i] = timeline1[idx]
    return yPred

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

    # columnId = 'iId'
    # columnEvent = 'ds'
    # columnSale = 'Sale'
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, 'CDNOW')
    
    resultsDir = os.path.join(workDir, 'results')
        
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])

    dStateList = sorted(list(statesDict.keys()))
    dFirst = dStateList[0]
    dLast = dStateList[-1]
    
    lastState = statesDict[dLast]
    firstState = statesDict[dFirst]
    
    E_Dict = lastState.lostDict
    lastEventDict = lastState.lastEventDict

#   1. death time == last event
#   2. death time == last event + q05
    mapClientToLastStateDate = {}
    dropList = [] # died before start RFM == first state
    mapClientToLastStateExpectedDate = {}
    dropListExpected = [] # died before start RFM == first state
    for client, status in E_Dict.items():
        if status == 0:
            mapClientToLastStateDate[client] = dLast
            mapClientToLastStateExpectedDate[client] = dLast            
        else:
            lastEvent = lastState.lastEventDict.get(client)
            if lastEvent < dFirst:
                dropList.append(client)
            dState = find(lastEvent, dStateList)
            mapClientToLastStateDate[client] = dState
            
            q05 = lastState.q05Dict.get(client)
            lastEventExpected = lastEvent + timedelta(days=q05)
            if lastEventExpected < dFirst:
                dropListExpected.append(client)
            dStateExpected = find(lastEventExpected, dStateList)
            mapClientToLastStateExpectedDate[client] = dStateExpected
            
            
    dropSet = set(dropListExpected)
    mapClientToStateDate = mapClientToLastStateExpectedDate
    clientFeaturesDict = {}
    for client, E in E_Dict.items():
        if client in dropSet:
            continue
        
        dDate = mapClientToStateDate.get(client)
        if dDate is None:
            continue
        
        clientFeatures1 = statesDict[dDate].getFeatures(client)
        if clientFeatures1 is None:
            continue
        
        clientFeatures1['E'] = E
        clientFeaturesDict[client] = clientFeatures1
        
        
    clientsList = sorted(list(clientFeaturesDict.keys()))    
    for client, clientFeatures1 in clientFeaturesDict.items():
        break    
    featuresList = sorted(list(clientFeatures1.keys()))

    featuresFrame = TrendFrame(clientsList, featuresList, dtype=float)

    for client, clientFeatures1 in clientFeaturesDict.items():
        featuresFrame.fillRow(client, clientFeatures1, toFloat=True)

    featuresDf = featuresFrame.to_pandas()

###############################################################################


    deadDf = featuresDf[featuresDf['E'] == 1].copy(deep=True)
    testDeadDf = deadDf.sample(frac=TEST_DEAD_FRACTION)  
    
    aliveDf = featuresDf[featuresDf['E'] == 0].copy(deep=True)
    testAliveDf = aliveDf.sample(frac=TEST_ALIVE_FRACTION) 
    
    trainDf = featuresDf[~featuresDf.index.isin(testDeadDf.index)].copy(deep=True)
    trainDf = trainDf[~trainDf.index.isin(testAliveDf.index)]
    

    columnsTrainXY = list(trainDf.columns)
    columnsTrainXY.remove('recency')
    columnsTrainXY.remove('loyalty')   
    columnsTrainXY.remove('q05')
    columnsTrainXY.remove('q09')       
    columnsTrainXY.remove('timeTo05')
    columnsTrainXY.remove('timeTo09')
    
    columnsTestXY = columnsTrainXY.copy()
    columnsTestXY.remove('E')
    columnsTestXY.remove('T')
    
    # trainDf, scaler = scaleDf(trainDf, columns=columnsTestXY, scaler=None)
    # testDeadDf, _ = scaleDf(testDeadDf, columns=columnsTestXY, scaler=scaler)

    
    smoothing_penalizer = 0
    penalizer = 0.1
    ancillary = False
    maeExpectedList = []
    maeMedianList = []
    binScoreExpectedList = []
    binScoreMedianList = []
    for aftModelName in ['WeibullAFT', 'LogNormalAFT', 
        'LogLogisticAFT', 'AalenAdditiveFitter', 'CoxPHFitter']:
#        aftModelName = 'LogNormalAFT'
        fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
        
        fitterAFT = fitAFT(fitterAFT, trainDf, duration_col='T', event_col='E', 
            columnsXY=columnsTrainXY, ancillary=ancillary, fit_left_censoring=False)

        testDeadDict = predictAFT(fitterAFT, testDeadDf, columnsXY=columnsTestXY)
        testAliveDict = predictAFT(fitterAFT, testAliveDf, columnsXY=columnsTestXY)

        if np.isinf(testDeadDict['yExpected']).sum() == 0:
            maeExpectedList.append(mean_absolute_error(testDeadDf['T'].values, testDeadDict['yExpected']))
        else:
            maeExpectedList.append(np.inf)
        if np.isinf(testDeadDict['yMedian']).sum() == 0:
            maeMedianList.append(mean_absolute_error(testDeadDf['T'].values, testDeadDict['yMedian']))
        else:
            maeMedianList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yExpected']))
        binScoreMedianList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yMedian']))


    print('Mean:', maeExpectedList)
    print('Median:', maeMedianList)

    print('Alive Mean concordance index:', binScoreExpectedList)
    print('Alive Median concordance index:', binScoreMedianList)
 


    # fitterAFT.print_summary()
    # fitterAFT.plot()


###############################################################################

    deadDf = featuresDf[featuresDf['E'] == 1].copy(deep=True)
    testDeadDf = deadDf.sample(frac=TEST_DEAD_FRACTION)  
    
    aliveDf = featuresDf[featuresDf['E'] == 0].copy(deep=True)
    testAliveDf = aliveDf.sample(frac=TEST_ALIVE_FRACTION) 
    
    trainDf = featuresDf[~featuresDf.index.isin(testDeadDf.index)].copy(deep=True)
    trainDf = trainDf[~trainDf.index.isin(testAliveDf.index)]

    columnsTrainXY = list(trainDf.columns)
    columnsTrainXY.remove('recency')
    columnsTrainXY.remove('loyalty')   
    columnsTrainXY.remove('q05')
    columnsTrainXY.remove('q09')       
    columnsTrainXY.remove('timeTo05')
    columnsTrainXY.remove('timeTo09')

    columnsTestXY = columnsTrainXY.copy()
    columnsTestXY.remove('E')
    columnsTestXY.remove('T')
        
    aftModelName = 'CoxPHFitter'
    fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
            penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)

    smoothing_penalizer = 0
    penalizer = 0.1
    ancillary = False
    maeExpectedList = []
    maeMedianList = []
    binScoreExpectedList = []
    binScoreMedianList = []
    featuresOutList = []
    featuresRemainingList = []
    featureOut = ''
    while len(columnsTrainXY) > 3:
        featuresRemainingList.append(columnsTestXY)
        featuresOutList.append(featureOut)
        print(featureOut)
        
        fitterAFT = fitAFT(fitterAFT, trainDf, duration_col='T', event_col='E', 
            columnsXY=columnsTrainXY, ancillary=ancillary, fit_left_censoring=False)
        
        testDeadDict = predictAFT(fitterAFT, testDeadDf, columnsXY=columnsTestXY)
        testAliveDict = predictAFT(fitterAFT, testAliveDf, columnsXY=columnsTestXY)
        
        if np.isinf(testDeadDict['yExpected']).sum() == 0:
            mae = mean_absolute_error(testDeadDf['T'].values, testDeadDict['yExpected'])
        else:
            mae = np.inf
        maeExpectedList.append(mae)
        if np.isinf(testDeadDict['yMedian']).sum() == 0:
            maeMedianList.append(mean_absolute_error(testDeadDf['T'].values, testDeadDict['yMedian']))
        else:
            maeMedianList.append(np.inf)

        binScoreExpectedList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yExpected']))
        binScoreMedianList.append(getConcordanceMetricAlive(testAliveDf['T'].values, testAliveDict['yMedian']))
        
        featureOut = findFeatureOut(fitterAFT)            
        columnsTrainXY.remove(featureOut)
        columnsTestXY = columnsTrainXY.copy()
        columnsTestXY.remove('E')
        columnsTestXY.remove('T')

        print(mae)

###############################################################################

    deadDf = featuresDf[featuresDf['E'] == 1].copy(deep=True)
    testDeadDf = deadDf.sample(frac=TEST_DEAD_FRACTION)  
    
    aliveDf = featuresDf[featuresDf['E'] == 0].copy(deep=True)
    testAliveDf = aliveDf.sample(frac=TEST_ALIVE_FRACTION) 
    
    trainDf = featuresDf[~featuresDf.index.isin(testDeadDf.index)].copy(deep=True)
    trainDf = trainDf[~trainDf.index.isin(testAliveDf.index)]

    columnsTrainXY = list(trainDf.columns)
    columnsTrainXY.remove('recency')
    columnsTrainXY.remove('loyalty')   
    columnsTrainXY.remove('q05')
    columnsTrainXY.remove('q09')       
    columnsTrainXY.remove('timeTo05')
    columnsTrainXY.remove('timeTo09')

    columnsTestXY = columnsTrainXY.copy()
    columnsTestXY.remove('E')
    columnsTestXY.remove('T')
    
    max_features='auto'
    n_estimators = 300
    max_depth = 5
    min_samples_split = 2
    min_samples_leaf = 1
    n_jobs = 6
    random_state = 101
    
    # for n_estimators in [100, 150, 200, 250, 300, 400, 500]:
    # for modelName in ['RandomSurvivalForest', 'ExtraSurvivalTrees', 
    #     'GradientBoostingSurvivalAnalysis', 'ComponentwiseGradientBoostingSurvivalAnalysis']:
    if True:
        modelName = 'GradientBoostingSurvivalAnalysis'
        
        model = makeModelSksurv(modelName=modelName, n_estimators=n_estimators, 
            max_depth=max_depth, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, max_features=max_features, verbose=True, 
            n_jobs=n_jobs, random_state=random_state) 
                        
        xTrain, yTrain = get_x_y(trainDf[columnsTrainXY], attr_labels=['E', 'T'], pos_label=1, survival=True)
        xTestDead, _ = get_x_y(testDeadDf[columnsTrainXY], attr_labels=['E', 'T'], pos_label=1, survival=True)
        xTestAlive, _ = get_x_y(testAliveDf[columnsTrainXY], attr_labels=['E', 'T'], pos_label=1, survival=True)

        model.fit(xTrain, yTrain)
        yPredDead = predictSurvivalTime(model, xTestDead, probability=0.5)
        mae = mean_absolute_error(testDeadDf['T'].values, yPredDead)

        yPredAlive = predictSurvivalTime(model, xTestAlive, probability=0.5)
        binScore = getConcordanceMetricAlive(testAliveDf['T'].values, yPredAlive)





