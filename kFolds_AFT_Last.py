# survival refgression 
# takes feature from kFolds_binary
# plots




import os
import pandas as pd
import numpy as np
from datetime import timedelta
import lib2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from sksurv.metrics import cumulative_dynamic_auc
from FrameList import FrameList
import lifelines
import libChurn2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
import lib3
from State import TS_STEP
from States import loadCDNOW
import State
from States import States
from datetime import date
from sklearn.metrics import mean_absolute_error
from Distributions import Statistics

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

    bGRU_only = False
    bUseGRU_predictions = True
    bUseLongTrends = True
    bUseShortTrends = True
    
    MIN_FREQUENCY = State.MIN_FREQUENCY
    RFM_HOLDOUT = 90

    random_state = 101
    nFolds = 3
    lag = 0

    sData = 'Simulated5' # CDNOW casa_2 plex_45 Retail Simulated5
    dEndTrain = None # real data
    dEndTrain = date(2004, 1, 1) # simulated data; additional MAE 

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)    
    lib2.checkDir(resultsSubDir)
        
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))
    
    yHatFiles = lib3.findFiles(dataSubDir, 'yHat.dat', exact=False)
    if len(yHatFiles) == 0:
        raise RuntimeError('GRU first')
        
    # clients and states to fit / predict
    # last clients state only if censored or death state if dead        
    yHat = lib2.loadObject(yHatFiles[0])
    yHat['iD'] = yHat['user'] + ' | ' + yHat['date'].astype(str)
    yHat.set_index('iD', drop=True, inplace=True)
    clientsAllList = list(yHat['user'].unique())
    dStateList = states.getExistingStates()
    nStates = len(dStateList)
    dStateReverseList = sorted(dStateList, reverse=True)
    stateLast = states.loadState(dStateList[-1], error='raise')

    foldsDict = libChurn2.splitList(nFolds, clientsAllList)

    columnsSpecial = ['client', 'date', 'duration', 'E', 'yHat']
    
    features = ['D_TS', 'status', 'frequency', 'pPoisson',
        'C_Orig', 'moneySum', 'moneyMedian', 'moneyDailyStep',
        'r10_R', 'r10_F', 'r10_L', 'r10_M', 'r10_C', 
        'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']
    

    
    columnsFrame = columnsSpecial + features
        
    # make frameList and fill with nan
    frame = FrameList(rowNames=list(yHat.index), columnNames=columnsFrame, df=None, dtype=None)
    for column in frame.columnNames:
        frame.fillColumn(column, np.nan)

#   Fill FrameArray with data from states
#   loop on states
    for i in tqdm(range(1, len(dStateList), 1)):
        dState = dStateList[i]
        yHat1 = yHat[yHat['date'] == dState]
        if len(yHat) == 0:
            continue    
        # load state
        state = states.loadState(dState, error='raise')
        # loop on users in state
        for i, row in yHat1.iterrows():
            client = row['user']
            clientObj = state.Clients.get(client)
            if client is None:
                raise ValueError('State: {} has no client: {}'.format(dState, client)) 
            iD = '{} | {}'.format(client, dState)
            # data from yHat to frame
            frame.put(iD, 'yHat', row['yHat'])
            frame.put(iD, 'client', client)
            frame.put(iD, 'date', row['date'])

            # loop on features for user in state
            for column in features:               
                value = clientObj.descriptors.get(column)
                if value is None:
                    raise ValueError('Client: {} has no feature: {}'.format(client, column))            
                frame.put(iD, column, value)
                    
#   prepare data for AFT
#   frame to pandas
    df = frame.to_pandas()
    df['E'] = df['status'].map(states.mapStatusToResponse)
    df['duration'] = df['D_TS']
    # make duratin 1 for clients with duration zero
    df['duration'] = np.where(df['duration'] == 0, 1, df['duration']) 
    df = df[df['duration'] > 0]
    df = df[columnsFrame] # arange columns
    del(df['D_TS'])
    df.sort_index(inplace=True)
    features.remove('D_TS')
    features.remove('status')
    
#!!! feature selection    
    if bUseGRU_predictions:
#   Use probability of churn obtained from ANN as input feature for survival        
        features.append('yHat')
    
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
        features = ['yHat']
        
# fit survival; one observation - one client
    predictionsList = []
    fitResultsFoldDict = {} # AIC, CI for each model for each fold
    sDict = {} # survival matrix
    hDict = {} # cumulative hazard matrix
    
    for fold, fold1_Dict in foldsDict.items():
        # break
        print('Fold: {} out of {}'.format(fold+1, nFolds))
    #   split train test    
        trainDf = df[df['client'].isin(fold1_Dict['clientsTrain'])].copy(deep=True)
        testDf = df[df['client'].isin(fold1_Dict['clientsTest'])].copy(deep=True)
        trainDf.sort_index(inplace=True)
        testDf.sort_index(inplace=True)            
    
    #   Scale features    
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
        predictionsDf = testDf[columnsSpecial + ['status']].copy(deep=True)    
        modelsAFT = ['WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT', 'CoxPHFitter'] #, 'GeneralizedGammaRegressionFitter']
        sY_List = [] # names for predicting columns
        for aftModelName in tqdm(modelsAFT):
        # aftModelName = modelsAFT[0]
        
            fitterAFT = libChurn2.makeAFT_Model(aftModelName, 
                penalizer=penalizer, smoothing_penalizer=smoothing_penalizer)
                
            fitterAFT = fitAFT(fitterAFT, xTrainScaled, duration_col='duration', event_col='E', 
                ancillary=ancillary, fit_left_censoring=False, timeline=timeLine)
            
            fitResults = predictAFT(fitterAFT, xTestScaled)
            fitResults = fixFloat(fitResults)
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
                                                                   
        predictionsList.append(predictionsDf)
                
    predictionsDf = pd.concat(predictionsList, axis=0)
    predictionsDf.sort_index(inplace=True)
    
    
###############################################################################
#   fit everything to get coefficients and AIC 
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
#   No more fit / predict, just collect results from folds
    ibsDict = {}    
    aucTS_Dict = {}
    aucDict = {}
    bsDict = {}
    for model in modelsAFT:

        sList = sDict[model]        
        s = pd.concat(sList, axis=1)
        columns = sorted(list(s.columns)) # sort columns
        s = s[columns]  
        hList = hDict[model]
        h = pd.concat(hList, axis=1)
        columns = sorted(list(h.columns)) # sort columns
        h = h[columns]
                             
        survival_train = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
        survival_test = makeStructuredArray(xScaled['duration'].values, xScaled['E'].values)
        times = sorted(list(xScaled['duration'].unique()))
        _ = times.pop(-1)
        estimate = s[s.index.isin(times)]
        estimate = estimate.values.T
        estimateH = h[h.index.isin(times)]
        estimateH = estimateH.values.T  
        
#   Integrated Brier score   
        ibs = integrated_brier_score(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimate, times=times)        
        ibsDict[model] = ibs

# dynamic AUC for right censored data:
        aucTS, aucMean = cumulative_dynamic_auc(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimateH, times=times)
        aucTS_Dict[model] = aucTS
        aucDict[model] = aucMean

#   Brier score for plot
        bs = brier_score(survival_train=survival_train, survival_test=survival_test,
                               estimate=estimate, times=times) 
        bsDict[model] = pd.Series(data=bs[1], index=bs[0])
            

# plot 4 diagrams on one pic
    plotBrierScore(bsDict, ibsDict)
    plot_cumulative_dynamic_auc(times, aucTS_Dict, aucDict)

#   Save coefficients to excel
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'Coefficients.xlsx')) as writer:
        for aftModelName, coef in coefDict.items():
            coef.to_excel(writer, sheet_name='{}'.format(aftModelName))
                
        
    with pd.ExcelWriter(os.path.join(resultsSubDir, 'LifeTime.xlsx')) as writer:
        predictionsDf.to_excel(writer)
        
        
    yamlDict = {'Features': features, 'Concordance index': fixFloat(ciDict), 'Integrated Brier score': fixFloat(ibsDict),
        'Dynamic AUC': fixFloat(aucDict), 'fitResultsFold': fixFloat(fitResultsFoldDict)}

    lib2.saveYaml(os.path.join(resultsSubDir, 'Metrics.yaml'), yamlDict, sort_keys=False)


    if dEndTrain is not None:
        transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(sData)), 
            mergeTransactionsPeriod=TS_STEP, minFrequency=MIN_FREQUENCY)
        
        mapClientToBirth = {}
        mapClientToBirthState = {}
        for client, clientObj in stateLast.Clients.items():
            status = clientObj.descriptors['status']
            if status == 4:
                # censored            
                mapClientToBirth[client] = clientObj.descriptors['dThirdEvent']
                mapClientToBirthState[client] = clientObj.descriptors['dBirthState']
    
        mapClientToFirstBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            eventType='first')
     
        mapClientToLastBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            eventType='last')
        
        mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            minFrequency=2)
    
        mapClientToMeanPeriodTS = {x : np.mean(y) / TS_STEP for x, y in mapClientToPeriods.items()}
        
        # from third buy to last buy + mean interevent time
        mapClientToDurationTS = {}
        for client, dBirth in mapClientToBirth.items():
            mapClientToDurationTS[client] = float((mapClientToLastBuy[client] - mapClientToBirth[client]).days / TS_STEP + mapClientToMeanPeriodTS[client])
    
        # only just died
        deadDf = predictionsDf[predictionsDf['status'] == 2].copy(deep=True)
        deadDf['dBirth'] = deadDf['client'].map(mapClientToFirstBuy)
        deadDf['meanPeriod'] = deadDf['client'].map(mapClientToMeanPeriodTS)
        deadDf['lifetime'] = (deadDf['date'] - deadDf['dBirth']).dt.days
        deadDf['lifetimeTS'] = deadDf['lifetime'].add(deadDf['meanPeriod']).div(TS_STEP)
        meanDead = deadDf['lifetimeTS'].mean()
        
        # only censored
        censoredDf = predictionsDf[predictionsDf['E'] == 0].copy(deep=True)
        censoredDf['D_TS_True'] = censoredDf['client'].map(mapClientToDurationTS)
        # MAE for censored clients only
        yamlDict['Mean TS lifetime'] = censoredDf['D_TS_True'].mean()
        censoredDf['mean'] = meanDead # baseline prediction; any censored lives mean time of dead
        sY_List.append('mean')
        for column in sY_List:
            # column = sY_List[0]
            # fit predictions to distribution
            stats1 = Statistics(censoredDf[column], testStats='ks', criterion='pvalue')
            # restrict outliers by 95 quantile, few crazy numbers crush all metrics
            maxLifetime = stats1.bestModel.ppf(0.95)
            censoredDf[column] = np.where(censoredDf[column] > maxLifetime, maxLifetime, censoredDf[column])
            
            mae = mean_absolute_error(censoredDf['D_TS_True'].values, censoredDf[column].values)
            yamlDict['MAE {}'.format(column)] = float(mae)
        
        lib2.saveYaml(os.path.join(resultsSubDir, 'Metrics.yaml'), yamlDict, sort_keys=False)
    
        del(censoredDf['mean'])
        with pd.ExcelWriter(os.path.join(resultsSubDir, 'LifeTimeCensored.xlsx')) as writer:
            censoredDf.to_excel(writer)    
        
    
    # client = '00040'
    # clientObj = stateLast.Clients[client]
    # clientObj.show()
    
    






