

# prepare data for LSTM binary classifier for churn

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
from sklearn.preprocessing import MinMaxScaler
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
from State import TS_STEP
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from libRecommender import METRICS_BINARY
from tensorflow.keras.callbacks import EarlyStopping
from collections.abc import Iterable
from random import choice

def findNearest(array, value):
    idx = lib2.argmin([abs(x - value) for x in array])
    return idx

def stackClients(clients):
    
    mapStatusToResponse = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}

    stackArrays = []
    statusList = []
    usersList = []
    datesList = []
    
    stackShortArrays = []
    statusShortList = []
    usersShortList = []
    datesShortList = []
    plotFrames = {}
    
    for client in tqdm(clients):
        dBirth = stateLast.Clients[client].descriptors['dBirthState']        
        datesClient = dStateList[findNearest(dStateList, dBirth) :]
            
        nTimeSteps = len(dStateList)
        timeSteps = list(range(0, nTimeSteps, 1))
        features_i_j = {}
        # for plot
        # frame_i = FrameArray(rowNames=datesClient, columnNames=descriptorsList, df=None, dtype=float)
        frame_i = FrameList(rowNames=datesClient, columnNames=descriptorsList, df=None, dtype=None)
        # from birth state to last state
        for dState_i in datesClient:
        # while dState_i <= dStateList[-1]:
            clientObj = statesDict[dState_i].Clients.get(client)
            # features_i_j = features1.copy()
            for feature in featuresList:
                value = clientObj.descriptors.get(feature, 0) # for zeroState there are no trends
                values = features_i_j.get(feature, [])
                values.append(value)
                features_i_j[feature] = values
                
            # lagre model; one customer makes many observations
            frame_i_j = FrameArray(rowNames=timeSteps, columnNames=featuresList, df=None, dtype=float)
            frame_i_j.padDict(features_i_j)
            stackArrays.append(frame_i_j.array.copy())
            usersList.append(clientObj.id)
            datesList.append(statesDict[dState_i].dState)
            status = clientObj.descriptors['status']
            statusList.append(status)
                
            # short model; one customer - one observation
            if (status == 2) or ((status == 4) and (dState_i == dStateList[-1])):
                # just died or censored (active) at last state
                stackShortArrays.append(frame_i_j.array.copy())
                usersShortList.append(clientObj.id)
                datesShortList.append(statesDict[dState_i].dState)
                statusShortList.append(status)                           
            
            # for plots
            for descriptor in descriptorsList:
                value = clientObj.descriptors.get(descriptor, 0)
                frame_i.put(dState_i, descriptor, value)
            
        plotFrames[client] = frame_i
            
    df = pd.DataFrame(columns=['user', 'date', 'status'])
    df['user'] = usersList
    df['date'] = datesList
    df['status'] = statusList
    df['response'] = df['status'].map(mapStatusToResponse)

    dfShort = pd.DataFrame(columns=['user', 'date', 'status'])
    dfShort['user'] = usersShortList
    dfShort['date'] = datesShortList
    dfShort['status'] = statusShortList    
    dfShort['response'] = dfShort['status'].map(mapStatusToResponse)
    
    return np.stack(stackArrays, axis=0), df, np.stack(stackShortArrays, axis=0), dfShort, plotFrames

def makeModelGRU(nTimeSteps, nFeatures, metrics):

    input1 = Input(shape=(nTimeSteps, nFeatures))
    lstm1 = GRU(128, return_sequences=True)(input1)
    drop1 = Dropout(0.3)(lstm1)
    dense1 = GRU(128, activation='relu')(drop1)       
    drop2 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation='relu')(drop2)
    drop3 = Dropout(0.3)(dense2)
    dense3 = Dense(128, activation='relu')(drop3)
    drop4 = Dropout(0.3)(dense3)
    dense4 = Dense(128, activation='relu')(drop4)
    drop5 = Dropout(0.3)(dense4)    
    outputs = Dense(1, activation='sigmoid')(drop5)
    model = Model(inputs=input1, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    return model

def scale(x1, x2, x3):
    scaler = StandardScaler()                 
    x1Scaled = scaler.fit_transform(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape)
    x2Scaled = scaler.transform(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape)
    x3Scaled = scaler.transform(x3.reshape(-1, x3.shape[-1])).reshape(x3.shape)
    return x1Scaled, x2Scaled, x3Scaled

def plotClient(client, prefix):    
    f = plotFramesTest[client]
    purchases = mapUserToPurchases.get(client)
    purchases = [x for x in purchases if x >= f.rowNames[0]]
    statusLast = f.get(f.rowNames[-1], 'status')
    if statusLast in [0, 4]:
        sStatus = 'Active'
    else:
        sStatus = 'Churn'
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19,15))

    # ax1.plot(f.rowNames, f.data['r10_FC'], 'm-', lw=1, alpha=0.6, label='r10_FC')
    ax1.plot(f.rowNames, f.data['r10_F'], 'g-', lw=1, alpha=0.6, label='r10_F')
    ax1.plot(f.rowNames, f.data['r10_FM'], 'b-', lw=1, alpha=0.6, label='r10_FM')
    ax1.plot(f.rowNames, f.data['r10_M'], 'r-', lw=1, alpha=0.6, label='r10_M')
    # ax1.plot(f.rowNames, f.data['r10_LFM'], 'k-', lw=1, alpha=0.6, label='r10_LFM')
    # ax1.plot(f.rowNames, f.data['r10_L'], 'y-', lw=1, alpha=0.6, label='r10_L')


    ax2.plot(f.rowNames, f.data['yHat'], 'r-', lw=1, alpha=0.6, label='yHat')
    ax2.plot(f.rowNames, f.data['pPoisson'], 'b-', lw=1, alpha=0.6, label='pPoisson')
    
    for purchase in purchases:
        ax2.axvline(purchase, linestyle='-.', label='purchase')

    ax1.legend()
    ax2.legend()
    
    ax1.set_ylabel('Ranking [1..10]')
    ax1.title.set_text('Ranking features')

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Probability')
    ax2.title.set_text('Probability')
    
    title = '{} - {}'.format(prefix, client)
    fig.suptitle(title, fontsize=14)
    
    fileName = '{}{}'.format('{} {}'.format(prefix, client), '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return

###############################################################################    
if __name__ == "__main__":


    random_state = 101
    maxTimeSteps = 20
    sData = 'CDNOW' # CDNOW casa_2 plex_45

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
    clientsAllList = stateLast.clientsAll


    # clientsList = sample(clientsAllList, 100)
    clientsList = clientsAllList

    # only int and float variables
    for client, clientObj in stateLast.Clients.items():
        break
    # for frameArray
    # descriptorsList = [x for x, y in clientObj.descriptors.items() if type(y) in [int, float, np.float64]]
    # for frameList
    descriptorsList = [x for x, y in clientObj.descriptors.items() if not isinstance(y, Iterable)]
    


    
    # featuresList = ['r10_R', 'r10_F', 'r10_L', 'r10_C', 'r10_C_Orig', 'r10_M',
    #     'r10_RF', 'r10_RFM', 'r10_RFMC', 'r10_FM', 'r10_FC', 'r10_RFC',
    #     'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC', 'trend_r10_FM', 'trend_r10_RFC',
    #     'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC', 'trend_short_r10_FM', 'trend_short_r10_RFC']

# 'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFMC'

    # featuresList = ['r10_F', 'r10_C', 'r10_M', 'r10_FM', 'r10_FC', 'r10_L']
    featuresList = ['r10_F', 'r10_L', 'r10_M', 'r10_FM', 'r10_C',
        'r10_FM', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFMC',
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        'trend_short_r10_LFMC']



#   Split clients
    VALID_FRAC = 0.2
    TEST_FRAC = 0.2
    
    clientsTest = sample(clientsAllList, int(len(clientsAllList) * TEST_FRAC))
    clientsTrainValid = list(set(clientsAllList).difference(set(clientsTest)))
    clientsValid = sample(clientsTrainValid, int(len(clientsTrainValid) * VALID_FRAC))
    clientsTrain = list(set(clientsTrainValid).difference(set(clientsValid)))
    del(clientsTrainValid)
    
# make 3D arrays and response
    xTrain, dfTrain , xTrainShort, dfTrainShort, plotFramesTrain = stackClients(clientsTrain)
    xValid, dfValid, xValidShort, dfValidShort, plotFramesValid = stackClients(clientsValid)
    xTest, dfTest, xTestShort, dfTestShort, plotFramesTest = stackClients(clientsTest)


    modelType = 'lagre'
#   Scale 3D arrays
    if modelType == 'lagre':
        xTrainScaled, xValidScaled, xTestScaled = scale(xTrain, xValid, xTest)
        yTrainDf = dfTrain.copy(deep=True)
        yValidDf = dfValid.copy(deep=True)
        yTestDf = dfTest.copy(deep=True)
    else:
        xTrainScaled, xValidScaled, xTestScaled = scale(xTrainShort, xValidShort, xTestShort)
        yTrainDf = dfTrainShort.copy(deep=True)
        yValidDf = dfValidShort.copy(deep=True)
        yTestDf = dfTestShort.copy(deep=True)

    model = makeModelGRU(xTrainScaled.shape[1], xTrainScaled.shape[2], metrics=METRICS_BINARY)

    # class_weight = lib4.getClassWeights(yTrainDf['response'].values)
    class_weight = None

    batch_size = 256
    epochs = 6
    patience = 10

    callbacks = EarlyStopping(monitor='val_auc', patience=patience,\
        verbose=1, mode='max', restore_best_weights=True)        

    history = model.fit(xTrainScaled, yTrainDf['response'].values, 
        batch_size=batch_size, epochs=epochs,\
        validation_data=(xValidScaled, yValidDf['response'].values), 
        shuffle=True, callbacks=[callbacks], class_weight=class_weight)
                 
   
    yProba = model.predict(xTestScaled)
    yTestDf['yHat'] = yProba

    lib2.saveObject(os.path.join(resultsSubDir, 'yTestDf.dat'), yTestDf, protocol=4)

#   add column
    for client in plotFramesTest.keys():
        plotFramesTest[client].addColumn('yHat', np.nan)

#   fill probability
    for i, row in yTestDf.iterrows():
        client = row['user']
        dDate = row['date']
        value = row['yHat']
        plotFramesTest[client].put(dDate, 'yHat', value)
                
    mapUserToPurchases = stateLast.mapUserToPurchases
    
#   random choice of tree types of clients
    dead = []
    dormant = []
    active = []
    for client in clientsTest:
        f = plotFramesTest[client]
        statusLast = f.get(f.rowNames[-1], 'status')
        if statusLast == 3:
            # skip not born
            continue
        if statusLast != 4:
            dead.append(client)
            continue
        foundDormant = False
        for status in f.data['status']:
            if status in [1, 2]:
                dormant.append(client)
                foundDormant = True
                break
        if foundDormant:
            continue
        active.append(client)

    maxPlots = 100
    deadSample = sample(dead, maxPlots)
    dormantSample = sample(dormant, maxPlots)
    activeSample = sample(active, maxPlots)

    # plot
    
    for client in deadSample:
        plotClient(client, 'Dead')
    for client in dormantSample:
        plotClient(client, 'Dormant')
    for client in activeSample:
        plotClient(client, 'Active')


# client = '05515'
# mapUserToPurchases[client]
# f = plotFramesTest[client]

