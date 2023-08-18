# train regression on dead clients
# predict survival time for alive


import os
import pandas as pd
import numpy as np
import lib2
from datetime import date
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import numpy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from libRecommender import METRICS_BINARY
from tensorflow.keras.callbacks import EarlyStopping
# import libChurn2
from random import sample
from FrameArray import FrameArray
from tqdm import tqdm
import lib4
import libRecommender
from tensorflow.keras.models import load_model
from random import random
from random import sample
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP
import lib3
from tensorflow.keras.utils import plot_model

def makeModelGRU(nTimeSteps, nFeatures, metrics):
    input1 = Input(shape=(nTimeSteps, nFeatures))
    gru1 = GRU(128, return_sequences=True)(input1)
    drop1 = Dropout(0.3)(gru1)
    gru2 = GRU(128)(drop1) # The requirements to use the cuDNN implementation : activation == tanh
    drop2 = Dropout(0.3)(gru2)
    dense2 = Dense(128, kernel_initializer='normal', activation='relu')(drop2)
    drop3 = Dropout(0.3)(dense2)
    dense3 = Dense(128, kernel_initializer='normal', activation='relu')(drop3)
    drop4 = Dropout(0.3)(dense3)
    dense4 = Dense(128, kernel_initializer='normal', activation='relu')(drop4)
    drop5 = Dropout(0.3)(dense4)    
    outputs = Dense(1, kernel_initializer='normal')(drop5)
    model = Model(inputs=input1, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
    return model

# def makeModelGRU(nTimeSteps, nFeatures, metrics):
#     input1 = Input(shape=(nTimeSteps, nFeatures))
#     gru1 = GRU(128, return_sequences=True)(input1)
#     drop1 = Dropout(0.3)(gru1)
#     gru2 = GRU(128)(drop1) # The requirements to use the cuDNN implementation : activation == tanh
#     drop2 = Dropout(0.3)(gru2)
#     dense2 = Dense(128, kernel_initializer='normal', activation='relu')(drop2)
#     drop3 = Dropout(0.3)(dense2)
#     outputs = Dense(1, kernel_initializer='normal')(drop3)
#     model = Model(inputs=input1, outputs=outputs)
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
#     return model

def scale(x1, x2, x3=None):
    scaler = StandardScaler()                 
    x1Scaled = scaler.fit_transform(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape)
    x2Scaled = scaler.transform(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape)
    if x3 is not None:
        x3Scaled = scaler.transform(x3.reshape(-1, x3.shape[-1])).reshape(x3.shape)
        return x1Scaled, x2Scaled, x3Scaled
    return x1Scaled, x2Scaled


METRICS_REGRESSION = [RootMeanSquaredError(name='RMSE'),
    MeanAbsoluteError(name='MAE'),
    MeanAbsolutePercentageError(name='MAPE')]

###############################################################################    
if __name__ == "__main__":
    
    p = 0.2 # take a fraction of train points to reduce memory usage
    
    kwargs = {'nTimeSteps': 50, 
            'validFrac': 0.2,
            'batch_size': 128, 
            'epochs': 100,
            'patience': 5,
            'maxNumTrain': 200000,
            'frac': 1,
            'bLoadModel': False} # for large data use fraction of training data to avoid memory problem
    kwargs['miniBatchPredict'] = kwargs['batch_size'] * 20 # number of observations to predict at once

    sData = 'Simulated1' # 'SimulatedShort001', 'Simulated1'
    mergeTransactionsPeriod = 1
    
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
    dEndTrain = args.get('dEnd') # None for real data
        
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    # dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)

    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)
    lib2.checkDir(resultsSubDir)
    
    # loweres mean predicted time
    # featuresList = ['r10_R', 'r10_F', 'r10_L', 'r10_M', 'r10_C', 'r10_C_Orig',
    #                 # 'pPoisson', 
    #                 # 'pSurvival', 'trend_pSurvival', 'trend_short_pSurvival',
    #                 # 'pChurn','trend_pChurn', 'trend_short_pChurn',
    #     'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
    #     'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
    #     'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
    #     'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
    #     'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']
    
    featuresList = ['C', 'trend_C', 'trend_short_C', 
        'C_Orig', 'trend_C_Orig', 'trend_short_C_Orig', 
        # 'pPoisson', 'trend_pPoisson', 'trend_short_pPoisson',
        # 'ratePoisson', 'trend_ratePoisson', 'trend_short_ratePoisson',        
        # 'r10_R', # never include it
        'r10_F', 
        'r10_M',
        # 'r10_L',
        'moneySum', 
        'moneyMedian', 
        'moneyDailyStep',        
        'r10_RF', 'trend_r10_RF', 'trend_short_r10_RF', 
        'r10_FM', 'trend_r10_FM', 'trend_short_r10_FM', 
        'r10_RFM', 'trend_r10_RFM', 'trend_short_r10_RFM',
        'r10_RFML', 'trend_r10_RFML', 'trend_short_r10_RFML']
        # 'tPoissonLifetime',
        # 'tPoissonDeath']
        # 'activity']

    # load parameters
    nTimeSteps = kwargs.get('nTimeSteps', 50)
    maxNumTrain = kwargs.get('maxNumTrain', 200000)
    frac = kwargs.get('frac', 1)
    features = kwargs.get('features', featuresList)
    trendLong = kwargs.get('trendLong', 50)
    trendShort = kwargs.get('trendShort', 10)    
    
    inputFilePath = os.path.join(dataSubDir, '{}.txt'.format(sData))
    transactionsDf = loadCDNOW(inputFilePath, 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)

    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat')) 
    states.getExistingStates()
    stateLast = states.loadLastState()
    
    clients = stateLast.getClients(statusOnly=[0, 1], mapClientToStatus=states.mapClientToStatus)
    
    # size = int(len(clients) / 2)
    # clients1 = sorted(sample(clients, size))
    # clients2 =sorted(list(set(clients).difference(set(clients1))))
    
    print('Load features from states')
    mapClientToFrame, mapIdToActivity, mapClientToFirstState = states.loadFeatures(features, nTimeSteps)
    
    # stateLast = states.loadLastState()
    
    print('Make train and predict arrays')
    trainStackList = []
    predictStackList = []
    responseTrainList = []
    
    userTrainList = []
    userPredictList = []
    dateTrainList = []    
    datePredictList = []
    for client in tqdm(stateLast.Clients.keys()):
        
        # break
        status = states.mapClientToStatus[client]
        # if client == '0006':
            # break
        
        if status == 3:
            # skip non-born
            continue

        frameClient = mapClientToFrame[client]
        # a = frameClient.to_pandas()
        # break

        if status == 0:
            # censored, predict last state
            dRightList = [frameClient.rowNames[-1]]
        elif status == 1:
            # dead
            # break
            dRightFirst = mapClientToFirstState.get(client)        
            dRightList = [x for x in frameClient.rowNames if x >= dRightFirst]
            # same lifetime for all dead observations from one client
            duration = float((stateLast.Clients[client].descriptors['dChurnState'] - stateLast.Clients[client].descriptors['dBirthState']).days / states.tsStep)
        
        for dRight in dRightList:    
            if status == 1:
                r = random()
                if r > p:
                    continue                
            # make 1 frame
            idxRight = frameClient.rowNames.index(dRight)
            idxLeft = idxRight - nTimeSteps
            rows = frameClient.rowNames[idxLeft+1: idxRight+1]
            # take window of nTimeSteps rows
            frameClient_i = frameClient.sliceFrame(rows=rows, columns=None)
            # b = frameClient_i.to_pandas()
            if status == 0:
                # censored
                predictStackList.append(frameClient_i.array.astype(np.float32))
                userPredictList.append(client)
                datePredictList.append(dRight)
            elif status == 1:
                # dead
                trainStackList.append(frameClient_i.array.astype(np.float32))
                responseTrainList.append(duration)
                userTrainList.append(client)
                dateTrainList.append(dRight)                                       
            else:
                # non-born
                pass
    
    del(mapClientToFrame)
    del(mapIdToActivity)
    del(mapClientToFirstState)
    
    # if maxNumTrain
    
    dfTrain = pd.DataFrame(columns=['user', 'date', 'y'])
    dfTrain['user'] = userTrainList
    dfTrain['date'] = dateTrainList
    dfTrain['y'] = responseTrainList
    
    dfPredict = pd.DataFrame(columns=['user', 'date'])
    dfPredict['user'] = userPredictList
    dfPredict['date'] = datePredictList
        
    xTrain = np.stack(trainStackList, axis=0)
    xPredict = np.stack(predictStackList, axis=0) 

    del(trainStackList)
    del(predictStackList)
    
    # reduce memory
    # xTrain = xTrain.astype(np.float32)
    # xPredict = xPredict.astype(np.float32)
    
    if np.isnan(xTrain).sum() > 0 or np.isnan(xPredict).sum() > 0:
        raise ValueError('Nan in ANN input array')
    
    if np.isinf(xTrain).sum() > 0 or np.isinf(xPredict).sum() > 0:
        raise ValueError('Inf in ANN input array')
    
###############################################################################

    validFrac = kwargs.get('validFrac', 0.15)
    batch_size = kwargs.get('batch_size', 128)
    epochs = kwargs.get('epochs', 5)
    patience = kwargs.get('patience', 5)
    bClassWeights = kwargs.get('bClassWeights', False)
    # monitor = kwargs.get('monitor', 'val_loss')
    monitor = kwargs.get('monitor', 'val_MAE')
    bLoadModel = kwargs.get('bLoadModel', True)
    miniBatchPredict = kwargs.get('miniBatchPredict', batch_size * 50)
    
    
    yTrain = dfTrain['y'].values
    xTrain, xPredict = scale(xTrain, xPredict)
    
        
    model = makeModelGRU(xTrain.shape[1], xTrain.shape[2], metrics=METRICS_REGRESSION)        
    plot_model(model, show_shapes=True, to_file=os.path.join(resultsSubDir, '{}.png'.format('GRU 128 128')))
    
    callbacks = EarlyStopping(monitor=monitor, patience=patience,\
        verbose=1, mode='min', restore_best_weights=True)        

    model.fit(xTrain, yTrain, 
        batch_size=batch_size, epochs=epochs,\
        validation_split=validFrac, 
        shuffle=True, callbacks=[callbacks])
                    

    yPredict = model.predict(xPredict)
    dfPredict['yHat'] = yPredict
    dfPredict['yHat'] = dfPredict['yHat'].astype(int)

    mapClientToDuration, mapClientToDurationTS = states.getTrueLifetime(transactionsDf)

    dfPredict['yTrue'] = dfPredict['user'].map(mapClientToDurationTS)
    dfPredict['yTrue'] = dfPredict['yTrue'].astype(int)

    
    survived = []
    for i, row in dfPredict.iterrows():
        user = row['user']
        dBirthState = stateLast.Clients[user].descriptors['dBirthState']
        dLastEventState = stateLast.Clients[user].descriptors['dLastEventState']
        duration = (dLastEventState - dBirthState).days / states.tsStep
        survived.append(duration)
        
    dfPredict['survived'] = survived
    dfPredict['remainingTrue'] = dfPredict['yTrue'] - dfPredict['survived']
    dfPredict['remainingPredicted'] = dfPredict['yHat'] - dfPredict['survived']
    dfPredict['remainingPredicted'] = dfPredict['remainingPredicted'].where(dfPredict['remainingPredicted'] >= 0, 0)
    
    dfPredict['errorAbsWeeks'] = (dfPredict['remainingTrue'] - dfPredict['remainingPredicted']).abs()
    meanError = dfPredict['errorAbsWeeks'].mean()
    medianError = dfPredict['errorAbsWeeks'].median()
    print('Mean error weeks: {}. Median error weeks: {}'.format(meanError, medianError))
    print('Mean remainingTrue weeks: {}'.format(dfPredict['remainingTrue'].mean()))
    print('Median remainingTrue weeks: {}'.format(dfPredict['remainingTrue'].median()))    
    print('Mean remainingPredicted weeks: {}'.format(dfPredict['remainingPredicted'].mean()))
    print('Median remainingPredicted weeks: {}'.format(dfPredict['remainingPredicted'].median()))


#   profit
    if mapClientToDurationTS is not None:
        if dEndTrain is not None:
            transactionsCutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=None, dEnd=dEndTrain,\
                bIncludeStart=True, bIncludeEnd=False, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
            transactionsHoldOutDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=dEndTrain, dEnd=None,\
                bIncludeStart=False, bIncludeEnd=True, minFrequency=MIN_FREQUENCY, dMaxAbsence=None) 
            mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsCutDf, 'iId', 'ds', 
                leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
                minFrequency=2)    
            mapClientToMeanPeriod = {x : round(np.mean(y), 2) for x, y in mapClientToPeriods.items()}   
            
    #   check with simulation
        # only censored clients
        transactionsHoldOutCensoredDf = transactionsHoldOutDf[transactionsHoldOutDf['iId'].isin(list(dfPredict['user']))].copy(deep=True)
        print('{} remaining censored'.format(transactionsHoldOutCensoredDf['iId'].nunique()))
        trueProfit = transactionsHoldOutCensoredDf['Sale'].sum()
        print('True spending:', trueProfit)

        dfPredict['meanInterPurchaseDays'] = dfPredict['user'].map(mapClientToMeanPeriod)

        dfPredict['dEndBuy'] = dfPredict['date'] + \
            pd.to_timedelta(dfPredict['remainingPredicted'].mul(TS_STEP).astype(int) \
                - dfPredict['meanInterPurchaseDays'].astype(int), unit='D')  

        dLastState = stateLast.dState
        moneyDailyDict = stateLast.getDescriptor('moneyDaily', activity=None, error='raise')

        dDate = dLastState
        datesList = []
        dLastBuy = dfPredict['dEndBuy'].max()

        while dDate < dLastBuy:
            dDate += timedelta(days=1)
            datesList.append(dDate)
                
        mapClientToChurnDate = dfPredict.set_index('user')[['dEndBuy']].squeeze().to_dict()

        profitList = []
        for dDate in tqdm(datesList):
            dailyProfit = 0
            for client, dEndBuy in mapClientToChurnDate.items():
                if dEndBuy > dDate:
                    dailyProfit += moneyDailyDict.get(client, 0)
            profitList.append(dailyProfit)
                    
        sum(profitList)

    yamlDict = {'Features': features}
    yamlDict['Mean error weeks'] = float(dfPredict['errorAbsWeeks'].mean())
    yamlDict['Median error weeks'] = float(dfPredict['errorAbsWeeks'].median())
    yamlDict['Population Mean remaining time True weeks'] = float(dfPredict['remainingTrue'].mean())
    yamlDict['Population Median remaining time True weeks'] = float(dfPredict['remainingTrue'].median())
    yamlDict['Population Mean remaining time Predicted weeks'] = float(dfPredict['remainingPredicted'].mean())
    yamlDict['Population Median remaining time Predicted weeks'] = float(dfPredict['remainingPredicted'].median())
    yamlDict['Profit censored True'] = float(trueProfit)
    yamlDict['Profit censored Predicted'] = float(sum(profitList))
    
    lib2.saveYaml(os.path.join(resultsSubDir, 'Scores Regression.yaml'), yamlDict, sort_keys=False)

    with pd.ExcelWriter(os.path.join(resultsSubDir, 'Censored Regression.xlsx')) as writer:
        dfPredict.to_excel(writer)

    lib2.saveObject(os.path.join(resultsSubDir, 'dfPredict.dat'), dfPredict)


