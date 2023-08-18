# plot ANN predictions


import os
# import pandas as pd
import numpy as np
import lib2
from matplotlib import pyplot as plt

# from tensorflow.keras.layers import LSTM

# import libChurn2
from random import sample
from Distributions import Statistics
from datetime import date




def plotActivity(client):
    
    # client = '0028'
    # periods = periodsDict[client]
    status = mapClientToStatus[client]
    frame = mapClientToFrame[client]
    dFirstState = mapClientToFirstState[client]
    dPurchaseList = mapUserToPurchases[client]
    dChurn = stateLast.Clients[client].descriptors.get('dChurn')
    dDeath = stateLast.Clients[client].descriptors.get('dDeath')
    
    datesList = []
    pChurnList = []
    activityList = []
    poissonList = []
    for i in range(0, len(frame.rowNames), 1):
        row = frame.rowNames[i]
        if row < dFirstState:
            continue
        datesList.append(row)
        pChurnList.append(frame.get(row, 'pChurn'))
        activity = frame.get(row, 'activity')
        activityList.append(mapActivityToPlot[activity])
        poissonList.append(frame.get(row, 'pPoisson'))
        
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19,15))

    ax1.plot(datesList, activityList, 'g', lw=1, alpha=0.6, label='r10_F')
    # ax1.plot(f.rowNames, f.data['r10_FM'], 'b-', lw=1, alpha=0.6, label='r10_FM')
    # ax1.plot(f.rowNames, f.data['r10_M'], 'r-', lw=1, alpha=0.6, label='r10_M')
    # ax1.plot(f.rowNames, f.data['r10_LFM'], 'k-', lw=1, alpha=0.6, label='r10_LFM')
    # ax1.plot(f.rowNames, f.data['r10_L'], 'y-', lw=1, alpha=0.6, label='r10_L')

    for dPurchase in dPurchaseList:
        ax1.axvline(dPurchase, linestyle='-.')

    ax1.axvline(dDeath, c='k', linestyle='-.', label='dDeath')
    ax1.axvline(dEndTrain, c='y', linestyle='--', label='dEndTrain')


    ax2.plot(datesList, pChurnList, c='r', lw=1, alpha=0.6, label='pChurn')
    ax2.plot(datesList, poissonList, c='b', lw=1, alpha=0.6, label='pPoisson')
    
    ax2.axvline(dPurchaseList[0], c='g', linestyle='-.', label='dFirstBuy')
    ax2.axvline(dPurchaseList[0], c='g', linestyle='-.', label='dLastBuy')

    ax2.axvline(dChurn, c='k', linestyle=':', label='dChurn')
    ax2.axvline(dDeath, c='k', linestyle='-.', label='dDeath')
    ax2.axvline(dEndTrain, c='y', linestyle='--', label='dEndTrain')
    

    ax1.legend()
    ax2.legend()
    
    ax1.set_ylabel('Activity')
    ax1.title.set_text('Activity')

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Probability')
    ax2.title.set_text('Probability')
    
    if status == 0:
        suffix = 'censored'
    else:
        suffix = 'churn'
        
    title = '{} {}'.format(client, suffix)
    fig.suptitle(title, fontsize=14)
    
    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return


if __name__ == "__main__":
       
    sData = 'SimulatedShort001'
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)
    resultsDir = os.path.join(workDir, 'results')
    lib2.checkDir(resultsDir)
    resultsSubDir = os.path.join(resultsDir, sData)
    lib2.checkDir(resultsSubDir)
    dEndTrain = date(2004, 7, 1)

    features = ['pChurn', 'activity', 'pPoisson']
    mapActivityToPlot = {1: 0, 2: 0, 3: 0, 14: 0, 4: 1, 5: 2, 6: 2, 9: 3, 10: 3, 7: 4, 8: 4, 11: 5, 12: 5, 13: 6}
    
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat')) 
    states.getExistingStates()
    stateLast = states.loadLastState()
    
    mapClientToStatus = states.mapClientToStatus
    mapUserToPurchases = states.mapUserToPurchases
    periodsDict = states.periodsDict
    
    mapClientToFrame, mapIdToActivity, mapClientToFirstState = states.loadFeatures(features, nTimeSteps=0)
    

    censored = [x for x, y in mapClientToStatus.items() if y == 0]
    churn = [x for x, y in mapClientToStatus.items() if y == 1]
    
    nSamples = 10
    censoredRandom = sample(censored, nSamples)
    churnRandom = sample(churn, nSamples)
    
    # client = '0029'
    for client in censoredRandom:
        plotActivity(client)
    for client in churnRandom:
        plotActivity(client)
          
  
    from matplotlib import lines
    l = lines.lineStyles.keys()
    print(l)        
        
        
        
        

#     from State import P_DEATH
#     from State import P_CHURN

#     stats = Statistics(periods, models=['Lognormal', 'Exponential'], testStats='ks', criterion='pvalue')
#     qChurn = round(stats.bestModel.ppf(P_CHURN)) # 38
#     qDeath = round(stats.bestModel.ppf(P_DEATH)) # 268

        
#     stats2 = stateLast.Clients[client].stats
#     qChurn2 = round(stats2.bestModel.ppf(P_CHURN)) # 20
#     qDeath2 = round(stats2.bestModel.ppf(P_DEATH)) # 59

#     stats3 = Statistics(periods, models=['Lognormal', 'Exponential'])

    
#     np.mean(periods)


#     stats4 = Statistics(periods, models='all', testStats='ks', criterion='pvalue')


#     stateLast.setStats()



#             clientObj = stateLast.Clients.get(client)
#             if clientObj is None:
#                 raise RuntimeError('customer {} has no instance'.format(client))
#             periods = clientObj.descriptors.get('periods')
#             if periods is None:
#                 raise RuntimeError('customer {} has no inter-purchase intervals'.format(client)) 
#             stats = Statistics(periods, models=['Lognormal', 'Exponential'])
#             clientObj.stats = stats
#             stateLast.Clients[client] = clientObj   


# a = Statistics(periods, models=['Lognormal', 'Exponential'])

