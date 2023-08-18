# Discrete Time Proportional Odds Model (DTPO)
# model survival curves for customers
# update states with new feature pSurvivalLR

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import lib2
from datetime import date
import libSurvival
from Encoder import DummyEncoderRobust
from sklearn.linear_model import LogisticRegression
import lib4
from FrameArray import FrameArray

def buildPersonPeriodFrameLR(features, mapClientToFrame):
    
    lastStateDf = states.getLastUserState()
    dStateList = states.dStateExistList
    
    dataList = []
    print('Build person-period data')
    for idx, row in tqdm(lastStateDf.iterrows()):
        client = row['user']
        dBirthState = row['dBirthState']
        dLastState = row['dLastState']
        statusLast = row['status']                     
        df1 = build_TV_data1a(client, dBirthState, dLastState, statusLast, dStateList, mapClientToFrame, features)
            
        if df1 is not None:
            dataList.append(df1)   
        
    df = pd.concat(dataList, axis=0)
    return df
    
def build_TV_data1a(client, dBirthState, dLastState, statusLast, dStateList, mapClientToFrame, features):
    """
    NEW
    One client DTPO
    keep rows after death

    client = '0001'
    Parameters
    ----------
    client : TYPE
        DESCRIPTION.
    dStateBirth : TYPE
        DESCRIPTION.
    dStateDeath : TYPE
        DESCRIPTION.
    lastStatus : TYPE
        DESCRIPTION.
    dStateList : TYPE
        DESCRIPTION.
    mapClientToFrame : TYPE
        DESCRIPTION.
    mapIdToStatus : TYPE
        DESCRIPTION.
    yHatDict : TYPE
        DESCRIPTION.
    features : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    rows = sorted([x for x in dStateList if x > dBirthState])
    if len(rows) == 0:
        return None
    
    stepsList = [x for x in range(1, len(rows)+1)]
    # rows starting from next after birth state 
    frame = mapClientToFrame[client].sliceFrame(rows=rows, columns=features)
        
    if statusLast == 0:
        # alive
        statusList = [0 for x in rows]
    elif statusLast == 1:
        # dead, 1 for all before dLastState, than 1
        statusList = []
        for dDate in rows:
            if dDate < dLastState:
                statusList.append(0)
            else:
                statusList.append(1)        
        
    # to pandas
    df = frame.to_pandas()
    df['E'] = statusList    
    df['date'] = df.index
    df['user'] = client
    df['step'] = stepsList
    df['iD'] = df['user'] + ' | ' + df['date'].astype(str)
    df.set_index('iD', drop=True, inplace=True)
    columns = ['user', 'date', 'E', 'step'] + features
    df = df[columns]

    return df



def makeDataLR():
    mapClientToFrame, mapIdToStatus, mapClientToFirstState = states.loadFeatures(features, nTimeSteps=0)

    # some clients will be missing (who has only one birth state)
    ppfDf = buildPersonPeriodFrameLR(features, mapClientToFrame)
    xPPF = libSurvival.scaleSurvivalData(ppfDf, featuresScale)

#   make data for LR    
    dummyEncoder = DummyEncoderRobust(unknownValue=0, prefix_sep='_')
    prefix = 'ts'
    xPPF_dummy = dummyEncoder.fit_transform(xPPF, 'step', prefix=prefix, drop=False)
    columnsDummy = list(dummyEncoder.columnsDummy)
            
    xPPF_huge, columnsDummyTiveVarying = libSurvival.makeHugeDTPO(xPPF_dummy, features, columnsDummy)
        
    # xPPF_dummy[columnsDummy].values : first part of equation p. 176 05_survival.pdf
    # xPPF_huge[columnsDummyTiveVarying].values : second part
    xTrain = np.concatenate((xPPF_dummy[columnsDummy].values, xPPF_huge[columnsDummyTiveVarying].values), axis=1)
    yTrain = xPPF['E'].values
    return xTrain, yTrain, xPPF, mapClientToFirstState

###############################################################################    
if __name__ == "__main__":


    # SimulatedShort001
    # args = {'dStart': None, 'holdOut': None,
    #     'dZeroState': date(2000, 7, 1), 'dEnd': date(2004, 7, 1)}

    # CDNOW
    args = {'dStart': None, 'holdOut': None,
        'dZeroState': None, 'dEnd': None}
    
    # casa_2
    # args = {'dStart': None, 'holdOut': None,
    #     'dZeroState': date(2016, 1, 1), 'dEnd': date(2019, 1, 1)}
        
    features = ['pPoisson', 'pChurn','trend_pChurn', 'trend_short_pChurn',
        'C_Orig', 'moneySum', 
        'r10_FMC', 'r10_LFM', 'r10_LFMC', 'r10_RFM', 'r10_RFMC',
        'trend_r10_FMC', 'trend_r10_LFMC', 
        'trend_r10_RFM', 'trend_r10_RFMC',
        'trend_short_r10_FMC', 
        'trend_short_r10_LFMC', 'trend_short_r10_RFM', 'trend_short_r10_RFMC']

    #!!! tems
    # features = ['pPoisson', 'pChurn','trend_pChurn', 'trend_short_pChurn']
    # featuresScale = ['trend_pChurn', 'trend_short_pChurn']
    
    featuresKeepOrig = ['pPoisson', 'pChurn', 'C_Orig']
    featuresScale = [x for x in features if x not in featuresKeepOrig]
    
    # random_state = 101
    # lag = 0

    sData = 'CDNOW' # CDNOW casa_2 plex_45 Retail Simulated5 SimulatedShort001
    dEndTrain = args.get('dEnd') # None for real data
    dZeroState = args.get('dZeroState') # None for real data
    holdOut = args.get('holdOut') # None for real data

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    # dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)
        
    states = lib2.loadObject(os.path.join(dataSubDir, 'States.dat'))

    xTrain, yTrain, xPPF, mapClientToFirstState = makeDataLR()

    class_weight = None
    # class_weight = lib4.getClassWeights(yTrain)

    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
        fit_intercept=False, intercept_scaling=1, class_weight=class_weight, 
        random_state=None, solver='newton-cg', max_iter=500, 
        multi_class='auto', verbose=True, warm_start=False, n_jobs=1, l1_ratio=None)

    lr.fit(xTrain, yTrain)
        
    # get Survival functions from LR predictions
    p = lr.predict_proba(xTrain)
    p = p[:, 1]
    rowNames = np.arange(0, xPPF['step'].max()+1).astype(int)
    columnNames = sorted(list(xPPF['user'].unique()))
    # first row corresponts to dBirthState for each user
    S = FrameArray(rowNames=rowNames, columnNames=columnNames, df=None, dtype=float)
    S.array[:, :] = np.nan
    
    data = xPPF[['user', 'step']].copy(deep=True)
    data['hazard'] = p
    
    for i, row in tqdm(data.iterrows()):
        user = row['user']
        step = row['step']
        h_t = row['hazard']
        if step == 1:
            sPrev = 1
            S.put(0, user, 1)
        s_t = sPrev * (1 - h_t)
        S.put(step, user, s_t)
        sPrev = s_t

#   Prepate Frame for updating states
    states.getExistingStates()
    stateLast = states.loadLastState()
    clients = sorted(list(stateLast.Clients.keys()))
    S_States = FrameArray(rowNames=states.dStateExistList, columnNames=clients, df=None, dtype=float)
    S_States.array[:, :] = 1
    
    for client in tqdm(clients):
        dBirthState = mapClientToFirstState[client]
        idxColumn = S.mapColumnNameToIndex.get(client)
        if idxColumn is None:
            continue        
        s = S.array[:, idxColumn]
        dStateList = [x for x in states.dStateExistList if x >= dBirthState]
        for i in range(0, len(dStateList), 1):
            dState = dStateList[i]
            pSurvival = s[i]
            S_States.put(dState, client, pSurvival)
            
    print('Update pSurvival in all states')
    dStateList = states.dStateExistList
    for dState in tqdm(dStateList):
        state = states.loadState(dState)
        for client in state.Clients.keys():
            pSurvival = S_States.get(dState, client)
            state.Clients[client].descriptors['pSurvival'] = pSurvival
        state.save(states.subDir)
        
    print('Update pChurn trends')
    trendShort = 10
    trendLong = 50
    states.updateTrends(['pSurvival'], short=trendShort, long=trendLong, forceAll=True)

#!!! temp
    # states.updateTrends(['C_Orig'], short=trendShort, long=trendLong, forceAll=True)




# #   get last hazard for clients
#     mapClientToHazard = {}
#     for client in H.columnNames:
#         idxColumn = H.mapColumnNameToIndex.get(client)
#         hazards = H.array[:, idxColumn]
#         vPrev = 0
#         for v in hazards:
#             if lib2.isNaN(v):
#                 break
#             vPrev = v
#         mapClientToHazard[client] = vPrev

#     data = xScaled[['user', 'duration', 'E']].copy(deep=True)
    
#     data['hazard'] = data['user'].map(mapClientToHazard)
#     data.dropna(inplace=True)
    
#     ci_LR = concordance_index(data['duration'], data['hazard'], data['E'])

    


