
#!!! looks good on plot4.py
# prepare data for RNN binary classifier for churn
# fit RNN save predictions
# no folds
# response based on status

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


def makeModelGRU(nTimeSteps, nFeatures, metrics):
    input1 = Input(shape=(nTimeSteps, nFeatures))
    lstm1 = GRU(128, return_sequences=True)(input1)
    drop1 = Dropout(0.3)(lstm1)
    # dense1 = GRU(128, activation='relu')(drop1)
    dense1 = GRU(128)(drop1) # The requirements to use the cuDNN implementation : activation == tanh
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

def scale(x1, x2, x3=None):
    scaler = StandardScaler()                 
    x1Scaled = scaler.fit_transform(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape)
    x2Scaled = scaler.transform(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape)
    if x3 is not None:
        x3Scaled = scaler.transform(x3.reshape(-1, x3.shape[-1])).reshape(x3.shape)
        return x1Scaled, x2Scaled, x3Scaled
    return x1Scaled, x2Scaled

def fitGRU(xTrain, xPredict, xOther, dfTrain, dfPredict, dfOther, localDir, **kwargs):
    
    """
    xTrain = xTrain1
    xPredict = xPredict2
    xOther = xTrain2
    dfTrain = dfTrain1
    dfPredict = dfPredict2
    dfOther = dfTrain2
    """
    
    def predictMiniBatch(xPredict, miniBatchPredict):
        yPredictList = []
        idxStart = 0
        while idxStart < xPredict.shape[0]:
            x = xPredict[idxStart : idxStart + miniBatchPredict, :, :]        
            y = model.predict(x)
            yPredictList.append(y.reshape(-1))
            idxStart += miniBatchPredict    
        yProbaPredict = np.concatenate(yPredictList, axis=0)
        return yProbaPredict
    
    
    validFrac = kwargs.get('validFrac', 0.15)
    batch_size = kwargs.get('batch_size', 256)
    epochs = kwargs.get('epochs', 5)
    patience = kwargs.get('patience', 10)
    bClassWeights = kwargs.get('bClassWeights', False)
    monitor = kwargs.get('monitor', 'val_auc')
    bLoadModel = kwargs.get('bLoadModel', True)
    miniBatchPredict = kwargs.get('miniBatchPredict', batch_size * 50)
    
    
    yTrain = dfTrain['y'].values
    xTrain, xPredict, xOther = scale(xTrain, xPredict, xOther)
    
    print('Fit')
    class_weight = lib4.getClassWeights(yTrain)
    if not bClassWeights:
        print('Set class_weight to None')
        class_weight = None

    if bLoadModel:
        try:
            model = load_model(os.path.join(localDir, 'GRU'), custom_objects={'fbeta': libRecommender.fbeta,
                'METRICS_BINARY': METRICS_BINARY})
            print('GRU model loaded')
        except:
            print('Make new GRU model')
            model = makeModelGRU(xTrain.shape[1], xTrain.shape[2], metrics=METRICS_BINARY)
    else:
        print('Make new GRU model')
        model = makeModelGRU(xTrain.shape[1], xTrain.shape[2], metrics=METRICS_BINARY)        
    
    callbacks = EarlyStopping(monitor=monitor, patience=patience,\
        verbose=1, mode='max', restore_best_weights=True)        

    model.fit(xTrain, yTrain, 
        batch_size=batch_size, epochs=epochs,\
        validation_split=validFrac, 
        shuffle=True, callbacks=[callbacks], class_weight=class_weight)
                      
    # release GPU memory by saving and loading model
    model.save(os.path.join(localDir, 'GRU'))
    model = load_model(os.path.join(localDir, 'GRU'), custom_objects={'fbeta': libRecommender.fbeta,
        'METRICS_BINARY': METRICS_BINARY})
        
    # predict; use mini batches to avoid memory crush
    print('Make predictions')
    yProbaPredict = predictMiniBatch(xPredict, miniBatchPredict)
    # yProbaPredict = model.predict(xPredict)
    dfPredict['yHat'] = yProbaPredict

    yProbaOther = predictMiniBatch(xOther, miniBatchPredict)
    # yProbaOther = model.predict(xOther)
    dfOther['yHat'] = yProbaOther

    return dfPredict, dfOther


###############################################################################    
 
def main(localDir, **kwargs):
    """
    localDir = dataSubDir
    kwargs = args
    
    Parameters
    ----------
    localDir : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        frac : int, fraction of training date to create

    Returns
    -------
    None.

    """
    
    featuresList = ['r10_R', 'r10_F', 'r10_L', 'r10_M', 'r10_C', 'r10_C_Orig', 'pPoisson', 
        'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
        'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
        'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']

    # load parameters
    nTimeSteps = kwargs.get('nTimeSteps', 50)
    maxNumTrain = kwargs.get('maxNumTrain', 200000)
    frac = kwargs.get('frac', 1)
    features = kwargs.get('features', featuresList)
    trendLong = kwargs.get('trendLong', 50)
    trendShort = kwargs.get('trendShort', 10)    
        
    states = lib2.loadObject(os.path.join(localDir, 'States.dat')) 
    states.getExistingStates()
    stateLast = states.loadLastState()
    
    clients = stateLast.getClients(statusOnly=[0, 1], mapClientToStatus=states.mapClientToStatus)
    
    size = int(len(clients) / 2)
    clients1 = sorted(sample(clients, size))
    clients2 =sorted(list(set(clients).difference(set(clients1))))
    
    print('Load features from states')
    mapClientToFrame, mapIdToActivity, mapClientToFirstState = states.loadFeatures(features, nTimeSteps)
    
    print('Stack features part 1')
    xTrain1, xPredict1, dfTrain1, dfPredict1 = states.stackAugmented2(clients1, \
        nTimeSteps, mapClientToFrame, mapIdToActivity, mapClientToFirstState, \
        maxNumTrain=maxNumTrain, frac=frac)
    
    print('Stack features part 2')
    xTrain2, xPredict2, dfTrain2, dfPredict2 = states.stackAugmented2(clients2, \
        nTimeSteps, mapClientToFrame, mapIdToActivity, mapClientToFirstState, \
        maxNumTrain=maxNumTrain, frac=frac)
               
    # train on first half of clients, predict second half of clients
    dfPredict2, dfTrain2 = fitGRU(xTrain1, xPredict2, xTrain2, dfTrain1, dfPredict2, dfTrain2, \
        localDir, **kwargs)

    dfPredict1, dfTrain1 = fitGRU(xTrain2, xPredict1, xTrain1, dfTrain2, dfPredict1, dfTrain1, \
        localDir, **kwargs)

    outList = [dfTrain1[dfPredict1.columns], dfTrain2[dfPredict1.columns], dfPredict1, dfPredict2]
    outDf = pd.concat(outList, axis=0)
    outDf.sort_values(['user', 'date'], inplace=True)
    outDf.reset_index(drop=True, inplace=True)
    lib2.saveObject(os.path.join(localDir, 'yHatExtended.dat'), outDf, protocol=4)
        
    print('Update pChurn in all states')
    dStateList = states.dStateExistList
    for dState in tqdm(dStateList):
        state = states.loadState(dState)
        out1Df = outDf[outDf['date'] == dState].copy(deep=True)
        if len(out1Df) == 0:
            continue
        for i, row in out1Df.iterrows():
            user = row['user']
            pChurn = row['yHat']
            clientObj = state.Clients.get(user)
            if clientObj is None:
                continue
            state.Clients[user].descriptors['pChurn'] = pChurn
            
        state.save(states.subDir)
        
    print('Update pChurn trends')
    states.updateTrends(['pChurn'], short=trendShort, long=trendLong, forceAll=True)

    return

if __name__ == "__main__":
    
    args = {'nTimeSteps': 50, 
            'validFrac': 0.2,
            'batch_size': 128, 
            'epochs': 100,
            'patience': 5,
            'maxNumTrain': 200000,
            'frac': 1,
            'bLoadModel': False} # for large data use fraction of training data to avoid memory problem
    args['miniBatchPredict'] = args['batch_size'] * 20 # number of observations to predict at once

    sData = 'SimulatedShort001' # 'SimulatedShort001'
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)
    
    main(dataSubDir, **args)
    




