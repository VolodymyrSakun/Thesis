

# prepare data for RNN binary classifier for churn
# fit RNN with 3 folds, save predictions

import os
import pandas as pd
import numpy as np
import lib2
# from datetime import date
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
import libChurn2




def findNearest(array, value):
    idx = lib2.argmin([abs(x - value) for x in array])
    return idx


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

def scale(x1, x2):
    scaler = StandardScaler()                 
    x1Scaled = scaler.fit_transform(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape)
    x2Scaled = scaler.transform(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape)
    return x1Scaled, x2Scaled

def stack(clients, features_i, mapUserToResponse):
    xList = []
    responseList = []
    for client in clients:
        x1 = features_i[client]
        xList.append(x1)
        responseList.append(mapUserToResponse[client])        
    x = np.stack(xList, axis=0)
    y = np.array(responseList)
    return x, y


###############################################################################    
if __name__ == "__main__":


    random_state = 101
    RFM_HOLDOUT = 90
    nTimeSteps = 50
    nFolds = 3
    VALID_FRAC = 0.15
    batch_size = 256
    epochs = 100
    patience = 10

        
    featuresList = ['r10_R', 'r10_F', 'r10_L', 'r10_M', 'r10_C', 'pPoisson', 'tPoissonLifetime',
        'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']
    
    sData = 'Simulated5' # CDNOW casa_2 plex_45 Retail Simulated5

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)    
    lib2.checkDir(resultsSubDir)
    
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))
        
    print('Stack variables in 3D array')
    features_i, clientsDf = states.stackClients(featuresList, nTimeSteps=nTimeSteps)

    clientsAllList = sorted(list(clientsDf['user'].unique()))
    foldsDict = libChurn2.splitList(nFolds, clientsAllList)
    tmp = clientsDf.set_index('user')
    mapUserToResponse = tmp['response'].to_dict()
    del(tmp)

#   use nFolds to fit / predict
    predictionsList = []
    model = None
    for fold in foldsDict.keys():
        clientsTrain = foldsDict[fold]['clientsTrain']
        clientsTest = foldsDict[fold]['clientsTest']
                        
        # stack x, get y
        xTrain, yTrain = stack(clientsTrain, features_i, mapUserToResponse)
        xTest, yTest = stack(clientsTest, features_i, mapUserToResponse)
        # scale
        xTrain, xTest = scale(xTrain, xTest)
    
        print('Fit')
        
        if model is None:
            # class_weight = lib4.getClassWeights(yTrain)
            class_weight = None            
            model = makeModelGRU(xTrain.shape[1], xTrain.shape[2], metrics=METRICS_BINARY)
        
        callbacks = EarlyStopping(monitor='val_auc', patience=patience,\
            verbose=1, mode='max', restore_best_weights=True)        
    
        history = model.fit(xTrain, yTrain,
            batch_size=batch_size, epochs=epochs,\
            validation_split=VALID_FRAC, 
            shuffle=True, callbacks=[callbacks], class_weight=class_weight)
                            
        yProba = model.predict(xTest)
        df1 = pd.DataFrame(columns=['user', 'yHat'])
        df1['user'] = clientsTest
        df1['yHat'] = yProba
        df1.set_index('user', inplace=True)
        predictionsList.append(df1)
        
#   collect predictions from folds
    df = pd.concat(predictionsList, axis=0)
    mapUserToYhat = df['yHat'].to_dict()
    clientsDf['yHat'] = clientsDf['user'].map(mapUserToYhat)

    lib2.saveObject(os.path.join(dataSubDir, 'yHat.dat'), clientsDf, protocol=4)


