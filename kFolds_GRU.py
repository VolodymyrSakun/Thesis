

# prepare data for LSTM binary classifier for churn

import os
# import libStent
# from datetime import datetime
import pandas as pd
# import libPlot
import numpy as np
# import lib3
# from datetime import timedelta
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
# from random import sample
# import libChurn2
# from matplotlib import pyplot as plt
# from scipy.stats import norm
# from Encoder import DummyEncoderRobust
# from sklearn.linear_model import LogisticRegression
import lib4
# from xgboost import XGBClassifier
# from lifelines import CoxTimeVaryingFitter
# from sklearn.preprocessing import minmax_scale
# from pandas import Series
# from StentBase1 import StentBase1
# from sklearn.ensemble import RandomForestClassifier
from FrameList import FrameList
from FrameArray import FrameArray
# from State import State
# from State import TS_STEP
import numpy
# from sksurv.datasets import load_gbsg2
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.metrics import integrated_brier_score
# from sksurv.preprocessing import OneHotEncoder
# from sksurv.metrics import concordance_index_censored
# from lifelines.utils import concordance_index
# from copy import deepcopy
# from datetime import datetime
# from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from libRecommender import METRICS_BINARY
from tensorflow.keras.callbacks import EarlyStopping
# from collections.abc import Iterable
# from random import choice
from random import shuffle
# from numba import cuda
import tensorflow as tf
import libChurn2

def findNearest(array, value):
    idx = lib2.argmin([abs(x - value) for x in array])
    return idx

def stackClients(clients, descriptorsList=None):
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))

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
        if descriptorsList is not None:
            frame_i = FrameList(rowNames=datesClient, columnNames=descriptorsList, df=None, dtype=None)
        else:
            frame_i = None
            
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
            if descriptorsList is not None:
                for descriptor in descriptorsList:
                    value = clientObj.descriptors.get(descriptor, 0)
                    frame_i.put(dState_i, descriptor, value)
            
        plotFrames[client] = frame_i
            
    del(statesDict)
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
    
    if descriptorsList is not None:
        return np.stack(stackArrays, axis=0), df, np.stack(stackShortArrays, axis=0), dfShort, plotFrames
    
    return np.stack(stackArrays, axis=0), df, np.stack(stackShortArrays, axis=0), dfShort

def makeModelGRU(nTimeSteps, nFeatures, metrics):

    input1 = Input(shape=(nTimeSteps, nFeatures))
    lstm1 = GRU(128, return_sequences=True)(input1)
    drop1 = Dropout(0.3)(lstm1)
    dense1 = GRU(128)(drop1)       
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

def scale(x1, x2):
    scaler = StandardScaler()                 
    x1Scaled = scaler.fit_transform(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape)
    x2Scaled = scaler.transform(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape)
    return x1Scaled, x2Scaled



###############################################################################    
if __name__ == "__main__":


    random_state = 101
    maxTimeSteps = 30
    nFolds = 3
    VALID_FRAC = 0.1
    modelType = 'small' # small lagre
    batch_size = 256
    epochs = 5
    patience = 10
        
    featuresList = ['r10_F', 'r10_L', 'r10_M', 'r10_C', 
        'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC'
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        'trend_short_r10_LFMC']
    
    sData = 'Simulated1' # CDNOW casa_2 plex_45 Retail Simulated1

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)    
    lib2.checkDir(resultsSubDir)
    
    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    nStates = len(statesDict)
    # transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])
    
    dStateList = sorted(list(statesDict.keys()))
    dStateLast = dStateList[-1]
    # stateZero = statesDict[dStateList[0]]
    stateLast = statesDict[dStateLast]
    del(statesDict) # free memory
    
    # dStateReverseList = sorted(dStateList, reverse=True)
    clientsAllList = stateLast.clientsAll

    foldsDict = libChurn2.splitList(nFolds, clientsAllList)

#   make predictions for all clients by nFolds
    for fold in foldsDict.keys():
        clientsTrain = foldsDict[fold]['clientsTrain']
        clientsTest = foldsDict[fold]['clientsTest']
           
# make 3D arrays and response
        xTrain, dfTrain , xTrainShort, dfTrainShort = stackClients(clientsTrain)    
        xTest, dfTest, xTestShort, dfTestShort = stackClients(clientsTest)
    
    #   Scale 3D arrays
        print('Scale')
        if modelType == 'lagre':
            del(xTrainShort)
            del(dfTrainShort)
            xTrain, xTest = scale(xTrain, xTest)
            yTrain = dfTrain['response'].values
            yTestDf = dfTest.copy(deep=True)
            del(dfTrain)
        else:
            del(xTrain)
            del(dfTrain)
            xTrain, xTest = scale(xTrainShort, xTestShort)
            yTrain = dfTrainShort['response'].values
            yTestDf = dfTestShort.copy(deep=True)
            del(dfTrainShort)
    
        print('Fit')
        # class_weight = lib4.getClassWeights(yTrainDf['response'].values)
        class_weight = None
        
        model = makeModelGRU(xTrain.shape[1], xTrain.shape[2], metrics=METRICS_BINARY)
        
        callbacks = EarlyStopping(monitor='val_auc', patience=patience,\
            verbose=1, mode='max', restore_best_weights=True)        
    
        history = model.fit(xTrain, yTrain, 
            batch_size=batch_size, epochs=epochs,\
            validation_split=VALID_FRAC, 
            shuffle=True, callbacks=[callbacks], class_weight=class_weight)
                            
        yProba = model.predict(xTest)
        yTestDf['yHat'] = yProba
        
        del(xTrain)
        del(xTest)

        lib2.saveObject(os.path.join(dataSubDir, '{} fold yHat.dat'.format(fold)), yTestDf, protocol=4)






