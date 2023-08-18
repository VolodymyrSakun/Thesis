#   plot sample of features and trends for framework section



import os
# import pandas as pd
# import numpy as np
import lib2
# from datetime import date
# from datetime import timedelta
# from sklearn.preprocessing import StandardScaler
# import numpy
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import GRU
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense
# from libRecommender import METRICS_BINARY
# from tensorflow.keras.callbacks import EarlyStopping
# import libChurn2
from random import sample
# from FrameArray import FrameArray
# from tqdm import tqdm
# import lib4
# import libRecommender
# from tensorflow.keras.models import load_model
# from random import random
# from random import sample
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.metrics import MeanAbsoluteError
# from tensorflow.keras.metrics import MeanAbsolutePercentageError
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

def plotClient(client):
    f = mapClientToFrame[client]
    purchases = states.mapUserToPurchases.get(client)
    purchases = [x for x in purchases if x >= f.rowNames[0]]
    statusLast = states.mapClientToStatus.get(client)
    if statusLast in [0, 4]:
        sStatus = 'Active'
    else:
        sStatus = 'Churn'
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19,15))

    for dPurchase in purchases:
        ax1.axvline(dPurchase, linestyle='-.')
        
    feature1 = 'r10_RFM'
    feature2 = 'C'
    feature3 = 'pPoisson'
    
    y1 = minmax_scale(f.array[:, f.getIdxColumn(feature1)], feature_range=(0, 1))
    
    ax1.plot(f.rowNames, y1, 'g', 
             linestyle='-', lw=1, alpha=0.6, label='{} scaled to [0..1]'.format(feature1))
    ax1.plot(f.rowNames, f.array[:, f.getIdxColumn(feature2)], 'b', 
             linestyle='-', lw=1, alpha=0.6, label=feature2)
    ax1.plot(f.rowNames, f.array[:, f.getIdxColumn(feature3)], 'r', 
             linestyle='-', lw=1, alpha=0.6, label=feature3)

    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_{}'.format(feature1))], 
        'g', linestyle='-', lw=1, alpha=0.6, label='trend_{}'.format(feature1))
    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_short_{}'.format(feature1))], 
        'g', linestyle='-.', lw=1, alpha=0.6, label='trend_short_{}'.format(feature1))

    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_{}'.format(feature2))], 
        'b', linestyle='-', lw=1, alpha=0.6, label='trend_{}'.format(feature2))
    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_short_{}'.format(feature2))], 
        'b', linestyle='-.', lw=1, alpha=0.6, label='trend_short_{}'.format(feature2))    
    
    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_{}'.format(feature3))], 
        'r', linestyle='-', lw=1, alpha=0.6, label='trend_{}'.format(feature3))
    ax2.plot(f.rowNames, f.array[:, f.getIdxColumn('trend_short_{}'.format(feature3))], 
        'r', linestyle='-.', lw=1, alpha=0.6, label='trend_short_{}'.format(feature3))    

    ax1.legend()
    ax2.legend()
    
    ax1.set_ylabel('Value')
    ax1.title.set_text('Time-dependent features')

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.title.set_text('Short and long trends')
    
    title = '{} - {}'.format(sStatus, client)
    fig.suptitle(title, fontsize=14)
    
    fileName = '{}{}'.format('{}'.format(title), '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return





if __name__ == "__main__":
    
    # featuresList = ['r10_R', 'r10_F', 'r10_L', 'r10_M', 'r10_C', 'r10_C_Orig', 'activity',
    #                 'dBirthState', 'dChurn', 'dDeath'
    #                 # 'pSurvival', 'trend_pSurvival', 'trend_short_pSurvival',
    #                 # 'pChurn','trend_pChurn', 'trend_short_pChurn',
    #     'r10_MC', 'r10_FM', 'r10_FC', 'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RF', 'r10_RFM', 'r10_RFMC',
    #     'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
    #     'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC',
    #     'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
    #     'trend_short_r10_LFMC', 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']
    
    featuresList = ['r10_RFM', 'C', 'pPoisson',                    
        'trend_r10_RFM', 'trend_short_r10_RFM',
        'trend_C', 'trend_short_C',
        'trend_pPoisson', 'trend_short_pPoisson']
   
    
    sData = 'CDNOW'
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)

    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)
    lib2.checkDir(resultsSubDir)
    
    transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(sData)), 
        mergeTransactionsPeriod=TS_STEP, minFrequency=MIN_FREQUENCY)
    
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat')) 
    states.getExistingStates()
    stateLast = states.loadLastState()
    states.setStatus(transactionsDf)
    
    clientsActive = stateLast.getClients(statusOnly=[0], mapClientToStatus=states.mapClientToStatus)
    clientsDead = stateLast.getClients(statusOnly=[1], mapClientToStatus=states.mapClientToStatus)
    
    features = featuresList
    nTimeSteps = 66
    mapClientToFrame, mapIdToActivity, mapClientToFirstState = states.loadFeatures(features, nTimeSteps=0)

    nPlots = 20
    clientsActiveSample = sample(clientsActive, nPlots)
    clientsDeadSample = sample(clientsDead, nPlots)
    
    for client in clientsActiveSample:
        plotClient(client)  
    for client in clientsDeadSample:
        plotClient(client)  
        
        


    
    
    