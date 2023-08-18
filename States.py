


import os
import libStent
from datetime import datetime
import pandas as pd
# import libPlot
import numpy as np
import lib3
from datetime import timedelta
import lib2
# from time import time
# import warnings
# from math import log
# from math import exp
# from Rank import Rank
# from datetime import date
from State import State
from tqdm import tqdm
from FrameArray import FrameArray
from lib2 import InterpolateTimeSeries
from datetime import date
from pathlib import Path
from random import sample
from FrameList import FrameList
from random import random
from Distributions import Statistics
from copy import deepcopy
from State import MIN_FREQUENCY
import libSurvival
from State import TS_STEP

# from State import TS_STEP
# from State import INCLUDE_START
# from State import INCLUDE_END
# from State import P_CHURN # quantile from CDF of best distribution to consider customers illness; 
# lifetime is time from birth to last event +  PPF(P_CHURN)
from State import P_DEATH # quantile from CDF of best distribution to consider customers death
activityANN_0 = [1, 2, 3, 14] # ANN response == 0
activityANN_1 = [6, 8, 10, 12] # ANN response == 1
activityANN_Predict = [4, 5, 7, 9, 11, 13] # ANN predict where this activitys exist


def getUserToPurchases(transactionsDf, columnId, columnDate):
    """
    for each user, get sorted list of transactions he made

    Parameters
    ----------
    transactionsDf : DataFrame
        transactins
    columnId : str
        columnId
    columnDate : str
        columnDate

    Returns
    -------
    mapUserToPurchases : dict
        user -> sorted list of transactions user made

    """
    g10 = transactionsDf.groupby([columnId])[columnDate].apply(list)
    d = g10.to_dict()
    mapUserToPurchases = {}
    for client, l in d.items():
        lSorted = sorted(list(set(l)))
        mapUserToPurchases[client] = lSorted            
    return mapUserToPurchases


def cdNowRawToDf(rawData):
    idList = []
    dateList = []
    # nPurchasesList = []
    moneyList = []
    for s in rawData:
        s1 = s.strip()
        l1 = s1.split()
        idList.append(l1[0])
        date_object = datetime.strptime(l1[1], '%Y%m%d').date()
        dateList.append(date_object)
        moneyList.append(float(l1[3]))
    outDf = pd.DataFrame(columns=['iId', 'ds', 'Sale'])
    outDf['iId'] = idList
    outDf['ds'] = dateList
    outDf['Sale'] = moneyList
    return outDf

def loadCDNOW(filePath, mergeTransactionsPeriod=7, minFrequency = 3):
    rawData = libStent.loadTextFile(filePath)
    dataDf = cdNowRawToDf(rawData)
    dataDf = lib3.groupTransactions2(dataDf, 'iId', 'ds', 'Sale',
        freq=mergeTransactionsPeriod)     
    g1 = dataDf.groupby('iId').agg({'ds': 'count'})
    g2 = g1[g1['ds'] >= minFrequency]
    clientsList = sorted(list(g2.index))
    # get rid of random clients; keep minFrequency+ only
    dataDf = dataDf[dataDf['iId'].isin(clientsList)]
    dataDf.sort_values(['ds', 'iId'], inplace=True)
    dataDf.reset_index(drop=True, inplace=True)
    return dataDf

def makeTsDates(transactionsDf, tsStep=7, dZeroState=None, holdOut=None):
    """
    Start from last transaction. Sort at the end -> from old to recent

    Parameters
    ----------
    transactionsDf : TYPE
        DESCRIPTION.

    Returns
    -------
    datesList : TYPE
        DESCRIPTION.

    """
                
    dLast = transactionsDf['ds'].max()
    dFirst = transactionsDf['ds'].min()    
    if dZeroState is None and holdOut is None:
        dZeroState = dFirst
    elif dZeroState is None and holdOut is not None:
        dZeroState = dFirst + timedelta(days=holdOut)
        
    dState = dZeroState
    datesList = []
    while dState <= dLast:
        datesList.append(dState)
        dState += timedelta(days=tsStep)
    return sorted(datesList)

class TrendFrame(FrameArray):
    
    def __init__(self, rowNames=None, columnNames=None, df=None, dtype=float):
        super().__init__(rowNames=rowNames, columnNames=columnNames, df=df, dtype=dtype)  
        return
    
    def calculateTrends(self):
        mapClientToTrend = {}
        regTime = InterpolateTimeSeries(timeSeries=True)
        for i in range(0, self.array.shape[0], 1):
            regTime.fit(self.columnNames, self.array[i, :])
            mapClientToTrend[self.rowNames[i]] = float(regTime.lrCoef[0])
        return mapClientToTrend
    
class States(object):
    DIR_NAME = 'States'
    MAX_LEN_LONG_TREND = 300
    # mapStatusToResponse = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}
    fileName = 'States.dat'

    def __init__(self, path, transactionsDf, tsStep=7, dZeroState=None, holdOut=None):
        if not Path(path).exists():
            raise RuntimeError('Folder {} does not exist'.format(path)) 
        self.dStateExistList = None # nothing for now
        self.statusAll = [0, 1, 3]
        self.Dir = path
        self.subDir = os.path.join(path, self.DIR_NAME)
        lib2.checkDir(self.subDir) # make dir is does not exist
        self.tsStep = tsStep        
        self.dStateList =  makeTsDates(transactionsDf, tsStep=tsStep, dZeroState=dZeroState, holdOut=holdOut)
        self.dZeroState = self.dStateList[0]
        self.clients = sorted(list(transactionsDf['iId'].unique()))
        self.dFirstTransaction = transactionsDf['ds'].min()
        self.dLastTransaction = transactionsDf['ds'].max()
        self.setStatus(transactionsDf)
        self.mapActivityToResponse = {x: 0 for x in activityANN_0}
        for activity1 in activityANN_1:
            self.mapActivityToResponse[activity1] = 1
        
        
        return        

    def setStatus(self, transactionsDf):
        """
        status : global determinator of clients end of activity
            corresponds to last available transaction date
            0 : censored; death is not observed within transactions
            1 : death event is observed (dLastPurchase + qDeath <= dLastTransaction) 
            3 : not born, does not have transactions after zero state       
            
        Parameters
        ----------
        transactionsDf : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # update values
        self.dFirstTransaction = transactionsDf['ds'].min()
        self.dLastTransaction = transactionsDf['ds'].max()        
        self.mapUserToPurchases = getUserToPurchases(transactionsDf, 'iId', 'ds')
        self.periodsDict = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, 
            bIncludeStart=True, bIncludeEnd=True, minFrequency=MIN_FREQUENCY)
        
        self.mapClientToStatus = {}
        self.mapClientToStats = {}
        self.mapClientTo_qDeath = {}
        print('Set status')
        for client in tqdm(self.clients):
            periods = self.periodsDict.get(client)
            if periods is None:
                # less than 3 purchases; discard client
                continue                      
            # distributions    
            stats = Statistics(periods, models=['Lognormal', 'Exponential'], testStats='ks', criterion='pvalue')
            self.mapClientToStats[client] = deepcopy(stats)
            # qChurn = round(stats.bestModel.ppf(P_CHURN)) # 0.5 from last event                            
            qDeath = round(stats.bestModel.ppf(P_DEATH)) # 0.98 from last event
            self.mapClientTo_qDeath[client] = qDeath
            dPurchaseList = self.mapUserToPurchases.get(client)
            if isinstance(dPurchaseList, list):
                dPurchaseList = sorted(dPurchaseList)
                dLastPurchase = dPurchaseList[-1]
                if dLastPurchase < self.dZeroState:
                    self.mapClientToStatus[client] = 3 # not born, does not have transactions after zero state           
                elif dLastPurchase + timedelta(days=qDeath) <= self.dLastTransaction:
                    self.mapClientToStatus[client] = 1 # dead
                else:
                    self.mapClientToStatus[client] = 0 # censored
            else:
                raise ValueError('Client: {}, dPurchaseList: {} crap'.format(client, dPurchaseList))
                
        return

    def getExistingStates(self):
        """
        load list of dates of states that exist already        
        update self.dStateExistList

        Returns
        -------
        None.

        """
        pathList = lib3.findFiles(self.subDir, '*.dat', exact=True)
        dStateExistList = []
        for path in pathList:
            fileName = os.path.split(path)[-1]
            sDate = fileName.split('.')[0]
            dDate = datetime.strptime(sDate, '%Y-%m-%d').date()
            dStateExistList.append(dDate)
        self.dStateExistList = sorted(dStateExistList)
        return 
        
    def update(self, transactionsDf, forceAll=False):
    #   load list of dates of states that exist already
        if forceAll:
            dStateExistList = []
        else:
            self.getExistingStates()
            dStateExistList = self.dStateExistList
        dStateExistSet = set(dStateExistList)
        
        # get list of missing states
        dStateMissList = sorted([x for x in self.dStateList if x not in dStateExistSet])
        
    #   Create states ; save in binary
        prevState = None
        for i in tqdm(range(0, len(dStateMissList), 1)):        
            dState = dStateMissList[i]
            state = State(dState, self.dZeroState, transactionsDf, self.mapClientToStatus, prevState=prevState)
            state.save(self.subDir) # save state to binary file  
            prevState = state
        return

    def updateTrends(self, features, short=None, long=None, forceAll=False):
        
        # find oldest state without trends
        self.getExistingStates() 
        dStateExistList = self.dStateExistList 
        if len(dStateExistList) < 2:
            print('Nothing to update. Number of states: {}'.len(dStateExistList))
            return
        
        if forceAll:
            dLast = dStateExistList[1]
        else:
            dLast = None
            for dState in dStateExistList:
                if dState == self.dZeroState:
                    # zero state has no trends
                    continue
                # fileName = '{}.dat'.format(dState)
                state = self.loadState(dState)
                # lib2.loadObject(os.path.join(self.subDir, fileName))
                # if variable state.bTrends is not defined, state does not have trends
                if 'bTrends' not in dir(state):
                    # not defined\
                    print(dState, self.dZeroState)
                    print()
                    dLast = dState                
                    break
                if not state.bTrends:
                    # defined but false
                    dLast = dState
                    break                
            if dLast is None:
                print('All trends are good')
                return
            if dLast == self.dZeroState:
                # go to first after zero state
                dLast = dStateExistList[1] 

        shortTrendLen = short
        if long is None:
            longTrendLen = self.MAX_LEN_LONG_TREND
        else:
            longTrendLen = long
                
#     empty frames to be filled by feature values for trends
        stateLast = self.loadLastState()
        clients = sorted(list(stateLast.Clients.keys()))
        framesDict = {x: TrendFrame(rowNames=clients, 
            columnNames=dStateExistList, df=None, dtype=float) for x in features}
        del(stateLast)
        
#   load state one by one and get descriptors        
# and extract features for trends separately
        print('Load data from states')
        for i in tqdm(range(0, len(dStateExistList), 1)): 
            dState = dStateExistList[i]
            # fileName = '{}.dat'.format(dState)
            state = self.loadState(dState)
            # lib2.loadObject(os.path.join(self.subDir, fileName))
            
            for trendDescriptor in features:
                featureDict = state.getDescriptor(trendDescriptor, activity=None, error='raise')
                framesDict[trendDescriptor].fillColumn(dState, featureDict)
        
    #   load state on by one and calculate trents, than save  
        print('Update trends')
        for i in tqdm(range(dStateExistList.index(dLast), len(dStateExistList), 1)):
            dState = dStateExistList[i]
            state = self.loadState(dState)            
            clients = sorted(list(state.Clients.keys()))
    
            # select dates for slicing
            if longTrendLen > i:
                datesLongSlope = dStateExistList[0 : i+1]
            else:
                datesLongSlope = dStateExistList[i-longTrendLen+1 : i+1]  
                
            if shortTrendLen > i:
                datesShortSlope = dStateExistList[0 : i+1]
            else:
                datesShortSlope = dStateExistList[i-shortTrendLen+1 : i+1]  
                
            for trendDescriptor in features:
                # trendDescriptor = features[0]
                # make slice for selected clients that belong to current state
                # long trend : dates are from somewhere to state.dDate with length longTrendLen
                # short trend : from state shortTrendLen steps before current state to state.dDate
                trendLongFrame = framesDict[trendDescriptor].sliceFrame(rows=clients, columns=datesLongSlope)
                trendShortFrame = framesDict[trendDescriptor].sliceFrame(rows=clients, columns=datesShortSlope)
                # calculate trends
                trendLong = trendLongFrame.calculateTrends()
                trendShort = trendShortFrame.calculateTrends()
                # put them to state object inside Clients
                state.putDescriptor('trend_{}'.format(trendDescriptor), trendLong)
                state.putDescriptor('trend_short_{}'.format(trendDescriptor), trendShort)
                                
            state.bTrends = True
            state.longTrendLen = len(datesLongSlope)
            state.shortTrendLen = len(datesShortSlope)
    
            state.save(self.subDir) # save state to binary file
        
        return

#     def stackClientsLast(self, featuresList, nTimeSteps=50):
#         """
#         OLD
#         Prepare data for ANN 
#         One client -> one observation

#         Parameters
#         ----------
#         featuresList : TYPE
#             DESCRIPTION.
#         nTimeSteps : TYPE, optional
#             DESCRIPTION. The default is 50.

#         Returns
#         -------
#         None.

#         """
#         dummyDates = []
#         dDummy = date(1900, 1, 1)
#         for i in range(0, nTimeSteps, 1):
#             dummyDates.append(dDummy)
#             dDummy += timedelta(days=1)
                        
#         dStateList = self.getExistingStates()
#         dStateReverseList = sorted(dStateList, reverse=True)
#         # dZeroState = dStateList[0]
#         dLastState = dStateReverseList[0]
#         stateLast = self.loadState(dLastState, error='raise')
#         # fileName = '{}.dat'.format(dLastState)
#         # stateLast = lib2.loadObject(os.path.join(self.subDir, fileName))
    
#         # return stateLast

# #   Find dates for each client
# # For censored from some state to last state; sequence length = nTimeSteps
# # For dead : from some state to death state; max sequence length = nTimeSteps
#         # mapClientToDates = {}
#         features_i = {}
#         usersList = []
#         datesList = []
#         statusList = []
#         for client, clientObj in stateLast.Clients.items():
#             status = clientObj.descriptors['status']
#             if status == 3:
#                 # not born
#                 continue
#             usersList.append(client)
#             statusList.append(status)
#             if status in [0, 4]:
#                 # censored, take last nTimeSteps points
#                 dDatesList = dStateReverseList[0 : nTimeSteps]
#                 datesList.append(dLastState)
#             elif status in [1, 2]:
#                 # dead
#                 dDeathState = clientObj.descriptors['dDeathState']                
#                 if dDeathState is None:
#                     raise ValueError('Client: {}. Status: {}. State : {}. dDeathState is None'.format(client, status, dLastState))
#                 datesList.append(dDeathState)
#                 idx = dStateReverseList.index(dDeathState)
#                 dDatesList = dStateReverseList[idx : idx + nTimeSteps]
#             else:
#                 raise ValueError('Client: {}. Status: {}. At last state'.format(client, status))
                
#             # make dummy dates for zero padding
#             if len(dDatesList) < nTimeSteps:
#                 nDummy = nTimeSteps - len(dDatesList)
#                 dummyDatesSample = sample(dummyDates, nDummy)
#                 dDatesList = dDatesList + dummyDatesSample
                
#             dDatesList = sorted(dDatesList)
#             frame_i = FrameArray(rowNames=dDatesList, columnNames=featuresList, df=None, dtype=float)
#             features_i[client] = frame_i
            
#         # some clients are not born; must be excluded
#         usersSet = set(usersList)
# #   Fill FrameArray with data from states
#         for i in tqdm(range(1, len(dStateList), 1)):
#             dState = dStateList[i]
#             state = self.loadState(dState, error='raise')
            
#             # fileName = '{}.dat'.format(dState)
#             # state = lib2.loadObject(os.path.join(self.subDir, fileName))            
#             # state = lib2.loadObject(os.path.join(stateSubDir, fileName))            
# # state = stateLast
#             for client, clientObj in state.Clients.items():
#                 if client not in usersSet:
#                     # some clients are not born; must be excluded                    
#                     continue
#                 if dState in features_i[client].rowNames:
#                     for descriptor in featuresList:
#                         value = clientObj.descriptors.get(descriptor, 0)
#                         features_i[client].put(dState, descriptor, value)                    
                
# #   get arrays from frames                        
#         for client in usersList:
#             data = features_i[client].array
#             features_i[client] = data
            
#         df = pd.DataFrame(columns=['user', 'date', 'status'])
#         df['user'] = usersList
#         df['date'] = datesList
#         df['status'] = statusList
#         df['response'] = df['status'].map(self.mapStatusToResponse)                     
            
#         return features_i, df
            
    def save(self):
        lib2.saveObject(os.path.join(self.Dir, self.fileName), self, protocol=4)
        return            

    def loadState(self, dDate, error='ignore'):
        files = lib3.findFiles(self.subDir, '{}.dat'.format(dDate), exact=True)
        if len(files) == 0:
            if error == 'ignore':
                return None            
            raise RuntimeError('At {} state {} not found'.format(self.subDir, dDate))        
        return lib2.loadObject(files[0])
        
    def loadLastState(self, error='ignore'):
        return self.loadState(self.dStateList[-1], error=error)
        
    def load_yHat(self):
        yHatFiles = lib3.findFiles(self.Dir, 'yHatExtended.dat', exact=False)
        if len(yHatFiles) == 0:
            raise RuntimeError('GRU first')
            
        # clients and states to fit / predict
        # last clients state only if censored or death state if dead        
        yHat = lib2.loadObject(yHatFiles[0])
        yHat['iD'] = yHat['user'] + ' | ' + yHat['date'].astype(str)
        yHat.set_index('iD', drop=True, inplace=True)        
        return yHat
    
    def makeSurvivalData(self, features):
        columnsSpecial = ['user', 'dBirthState', 'dLastState', 'duration', 'E']
        self.getExistingStates()
        dStateList = self.dStateExistList
        # for each user existing in last state get state of churn if dead or last state if censored        
        lastStateDf = self.getLastUserState()
    
        columnsFrame = columnsSpecial + features
            
        # make frameList and fill with nan
        frame = FrameList(rowNames=list(lastStateDf.index), columnNames=columnsFrame, df=None, dtype=None)
        for column in frame.columnNames:
            frame.fillColumn(column, np.nan)

    #   Fill FrameArray with data from states
    #   loop on states
        for i in tqdm(range(1, len(dStateList), 1)):
            dState = dStateList[i]
            lastState1Df = lastStateDf[lastStateDf['dLastState'] == dState]
            if len(lastState1Df) == 0:
                continue    
            # load state
            state = self.loadState(dState, error='raise')
            # loop on users in state
            for i, row in lastState1Df.iterrows():
                # break
                client = row['user']
                status = self.mapClientToStatus.get(client)
                if status is None:
                    raise ValueError('Client: {} has no status'.format(client))  
                if status not in [0, 1]:
                    raise ValueError('Client: {} has status {}. Must be 0 or 1'.format(client, status))
                clientObj = state.Clients.get(client)
                if client is None:
                    raise ValueError('State: {} has no client: {}'.format(dState, client)) 
                iD = '{} | {}'.format(client, dState)                    
                frame.put(iD, 'user', client)
                frame.put(iD, 'dBirthState', row['dBirthState'])
                frame.put(iD, 'dLastState', row['dLastState'])
                duration = (row['dLastState'] - row['dBirthState']).days
                durationTS = int(duration / TS_STEP)
                # frame.put(iD, 'duration', clientObj.descriptors.get(sDuration))
                frame.put(iD, 'duration', durationTS) # more certain
                frame.put(iD, 'E', status)
                
                # loop on features for user in state
                for column in features:               
                    value = clientObj.descriptors.get(column)
                    if value is None:
                        raise ValueError('Client: {} has no feature: {}'.format(client, column))            
                    frame.put(iD, column, value)
                        
    #   prepare data for AFT
    #   frame to pandas
        df = frame.to_pandas()
        # make duratin 1 for clients with duration zero
        df['duration'] = np.where(df['duration'] == 0, 1, df['duration']) 
        df = df[df['duration'] > 0]
        df = df[columnsFrame] # arange columns
        df.sort_index(inplace=True)
        
        return df
    
    def getChurnDate(self, transactionsDf):
        """
        Churn date : date of last purchase + mean inter-purchase period

        Returns
        -------
        mapClientToChurnDate : TYPE
            DESCRIPTION.

        """
        
        mapClientToLastBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            eventType='last')
        
        mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            minFrequency=MIN_FREQUENCY)
    
        mapClientToMeanPeriod = {x : np.mean(y) for x, y in mapClientToPeriods.items()}

        # date of last purchase + mean inter-purchase period
        mapClientToChurnDate = {}
        for client, dLastPurchase in mapClientToLastBuy.items():
            mapClientToChurnDate[client] = dLastPurchase + timedelta(days=mapClientToMeanPeriod[client])
                      
        return mapClientToChurnDate      
        
        
    def getTrueLifetime(self, transactionsDf):
        """
        For simulated data only
        For censored clients only
        Lifetime == birthState (third event) to last buy date + mean interevent time

        Returns
        -------
        mapClientToDurationTS : TYPE
            DESCRIPTION.

        """
        
        stateLast = self.loadLastState()
        
        mapClientToBirth = {}
        mapClientToBirthState = {}
        for client, clientObj in stateLast.Clients.items():
            status = self.mapClientToStatus.get(client)
            if status is None:
                raise ValueError('Client {} has no status in states.mapClientToStatus'.format(client))
            if status in [0, 4]:
                # censored            
                mapClientToBirth[client] = clientObj.descriptors['dThirdEvent']
                mapClientToBirthState[client] = clientObj.descriptors['dBirthState']
     
        mapClientToLastBuy = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            eventType='last')
        
        mapClientToPeriods = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, 
            minFrequency=2)
    
        mapClientToMeanPeriod = {x : np.mean(y) for x, y in mapClientToPeriods.items()}
    
        # from birth state to last buy date + mean interevent time
        mapClientToDurationTS = {}
        mapClientToDuration = {}
        for client, dBirth in mapClientToBirth.items():
            duration = float((mapClientToLastBuy[client] - \
                mapClientToBirthState[client]).days + mapClientToMeanPeriod[client])
            mapClientToDuration[client] = duration
            mapClientToDurationTS[client] = duration / self.tsStep

        return mapClientToDuration, mapClientToDurationTS

    def loadFeatures(self, features, nTimeSteps=50):   
        """
        Prepare data for augmented ANN
        One client -> many observations
        Pad zeros
        Followed by stack

        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        nTimeSteps : TYPE, optional
            DESCRIPTION. The default is 50.

        Returns
        -------
        mapClientToFrame : TYPE
            DESCRIPTION.
        mapIdToStatus : TYPE
            DESCRIPTION.
        mapClientToFirstState : TYPE
            DESCRIPTION.

        """
        self.getExistingStates()
        dStateList = self.dStateExistList

        dummyDates = []
        dDummy = date(1900, 1, 1)
        for i in range(0, nTimeSteps, 1):
            dummyDates.append(dDummy)
            dDummy += timedelta(days=1)
            
        print('load features from states to frames')
        mapClientToFrame = {} # client -> frame
        mapIdToActivity = {}
        mapClientToFirstState = {}
        for i in tqdm(range(0, len(dStateList), 1)):
            dState = dStateList[i]
            state = self.loadState(dState)
            for client, clientObj in state.Clients.items():
                frameClient = mapClientToFrame.get(client)
                iD = '{} | {}'.format(client, dState)
                activity = clientObj.descriptors.get('activity')
                if activity is None:
                    raise ValueError('Client: {} has no activity at {}'.format(client, dState))
                mapIdToActivity[iD] = activity
                if frameClient is None:
                    mapClientToFirstState[client] = dState
                    # new client, make new  frame
                    # dates that exist in states before dState (including dState)
                    dLeftList = [x for x in dStateList if x <= dState]
                    
                    # make dummy dates for zero padding
                    if len(dLeftList) < nTimeSteps:
                        nDummy = nTimeSteps - len(dLeftList)
                        # insert bummy dates for zero padding
                        dummyDatesSample = sample(dummyDates, nDummy)
                    else:
                        dummyDatesSample = []
    
                    dRightList = [x for x in dStateList if x > dState]
                    # all dates for client
                    dFrameList = sorted(dummyDatesSample + dLeftList + dRightList)
                    # make frame; rows == dates, columns == features
                    frameClient = FrameArray(rowNames=dFrameList, columnNames=features, df=None, dtype=float)
                    
                # put all features from one state to one client 
                for descriptor in features:
                    value = clientObj.descriptors.get(descriptor, 0)
                    frameClient.put(dState, descriptor, value)                 
                
                mapClientToFrame[client] = frameClient
    
        return mapClientToFrame, mapIdToActivity, mapClientToFirstState


    def stackAugmented(self, clients, nTimeSteps, mapClientToFrame, mapIdToActivity, 
            mapClientToFirstState, maxNumTrain=200000, frac=1):
        
        """
        CRAP results
        Prepare data for augmented RNN
        use outputs from self.loadFeatures
        use activity
        standalone (no folds)
    
        Parameters
        ----------
        states : TYPE
            DESCRIPTION.
        nTimeSteps : TYPE
            DESCRIPTION.
        mapClientToFrame : TYPE
            DESCRIPTION.
        mapIdToStatus : TYPE
            DESCRIPTION.
        mapClientToFirstState : TYPE
            DESCRIPTION.
        maxNumTrain : int, optional
            Limit observations of transactions. The default is 200000.
        frac : float, optional
            Fraction of training data to use. Works faster if assigned than maxNumTrain alone. The default is 1.
    
        Returns
        -------
        xTrain : TYPE
            DESCRIPTION.
        xPredict : TYPE
            DESCRIPTION.
        dfTrain : TYPE
            DESCRIPTION.
        dfPredict : TYPE
            DESCRIPTION.
    
        """
        stateLast = self.loadLastState()
        
        print('Make train and predict arrays')
        trainStackList = []
        predictStackList = []
        activityTrainList = []
        activityPredictList = []
        
        userTrainList = []
        userPredictList = []
        dateTrainList = []    
        datePredictList = []
        for client in tqdm(clients):
            status = self.mapClientToStatus.get(client)
            if status is None:
                raise ValueError('Client: {} has no status'.format(client))     
            
            if status == 3:
                # skip non-born
                continue
            dRightFirst = mapClientToFirstState.get(client)
            frameClient = mapClientToFrame[client]
            # break
            dRightList = [x for x in frameClient.rowNames if x >= dRightFirst]
            for dRight in dRightList:
                iD = '{} | {}'.format(client, dRight)
                activity = mapIdToActivity.get(iD)
                if activity is None:
                    raise ValueError('Client: {} has no activity at {}'.format(client, dRight))                                    
                
                # make 1 frame
                idxRight = frameClient.rowNames.index(dRight)
                idxLeft = idxRight - nTimeSteps
                rows = frameClient.rowNames[idxLeft+1: idxRight+1]
                # take window of nTimeSteps rows
                frameClient_i = frameClient.sliceFrame(rows=rows, columns=None)
                    
                if status == 0:
                    # censored
                    dLastEventState = stateLast.Clients[client].descriptors.get('dLastEventState')
                    if dLastEventState is None:
                        raise ValueError('Client: {} has no dLastEventState at last state'.format(client))                        
                    if rows[-1] >= dLastEventState:
                    # to predict, if frame corresponds to state >= last purchase
                        predictStackList.append(frameClient_i.array)
                        activityPredictList.append(activity)
                        userPredictList.append(client)
                        datePredictList.append(dRight)                 
                        continue
                    
                if activity in activityANN_Predict:
                    # predict
                    predictStackList.append(frameClient_i.array)
                    activityPredictList.append(activity)
                    userPredictList.append(client)
                    datePredictList.append(dRight)
                elif (activity in activityANN_0) or (activity in activityANN_1):
                    # train
                    trainStackList.append(frameClient_i.array)
                    activityTrainList.append(activity)
                    userTrainList.append(client)
                    dateTrainList.append(dRight)                    
                else:
                    # non-born
                    pass                    
        
        dfTrain = pd.DataFrame(columns=['user', 'date', 'activity', 'y'])
        dfTrain['user'] = userTrainList
        dfTrain['date'] = dateTrainList
        dfTrain['activity'] = activityTrainList
        dfTrain['y'] = dfTrain['activity'].map(self.mapActivityToResponse)
        
        dfPredict = pd.DataFrame(columns=['user', 'date', 'activity'])
        dfPredict['user'] = userPredictList
        dfPredict['date'] = datePredictList
        dfPredict['activity'] = activityPredictList
            
        xTrain = np.stack(trainStackList, axis=0)
        xPredict = np.stack(predictStackList, axis=0) 
        
        # just in case
        xTrain = xTrain.astype(np.float32)
        xPredict = xPredict.astype(np.float32)
        
        if np.isnan(xTrain).sum() > 0 or np.isnan(xPredict).sum() > 0:
            raise ValueError('Nan in ANN input array')
        
        if np.isinf(xTrain).sum() > 0 or np.isinf(xPredict).sum() > 0:
            raise ValueError('Inf in ANN input array')
        
        return xTrain, xPredict, dfTrain, dfPredict

    def stackAugmented2(self, clients, nTimeSteps, mapClientToFrame, mapIdToActivity, 
            mapClientToFirstState, maxNumTrain=200000, frac=1):
        
        """
        Prepare data for augmented RNN
        use outputs from self.loadFeatures
        response from status
        designed for 2 folds
    
        Parameters
        ----------
        states : TYPE
            DESCRIPTION.
        nTimeSteps : TYPE
            DESCRIPTION.
        mapClientToFrame : TYPE
            DESCRIPTION.
        mapIdToStatus : TYPE
            DESCRIPTION.
        mapClientToFirstState : TYPE
            DESCRIPTION.
        maxNumTrain : int, optional
            Limit observations of transactions. The default is 200000.
        frac : float, optional
            Fraction of training data to use. Works faster if assigned than maxNumTrain alone. The default is 1.
    
        Returns
        -------
        xTrain : TYPE
            DESCRIPTION.
        xPredict : TYPE
            DESCRIPTION.
        dfTrain : TYPE
            DESCRIPTION.
        dfPredict : TYPE
            DESCRIPTION.
    
        """
        stateLast = self.loadLastState()
        
        print('Make train and predict arrays')
        trainStackList = []
        predictStackList = []
        responseTrainList = []
        
        userTrainList = []
        userPredictList = []
        dateTrainList = []    
        datePredictList = []
        for client in tqdm(clients):
            status = self.mapClientToStatus.get(client)
            if status is None:
                raise ValueError('Client: {} has no status'.format(client))     
            
            if status == 3:
                # skip non-born
                continue
            dRightFirst = mapClientToFirstState.get(client)
            frameClient = mapClientToFrame[client]
            # break
            dRightList = [x for x in frameClient.rowNames if x >= dRightFirst]
            for dRight in dRightList:
                iD = '{} | {}'.format(client, dRight)
                activity = mapIdToActivity.get(iD)
                if activity is None:
                    raise ValueError('Client: {} has no activity at {}'.format(client, dRight))                                    
                
                # make 1 frame
                idxRight = frameClient.rowNames.index(dRight)
                idxLeft = idxRight - nTimeSteps
                rows = frameClient.rowNames[idxLeft+1: idxRight+1]
                # take window of nTimeSteps rows
                frameClient_i = frameClient.sliceFrame(rows=rows, columns=None)
                    
                if status == 0:
                    # censored
                    dLastEventState = stateLast.Clients[client].descriptors.get('dLastEventState')
                    if dLastEventState is None:
                        raise ValueError('Client: {} has no dLastEventState at last state'.format(client))                        
                    if rows[-1] > dLastEventState:
                    # to predict, if frame corresponds to state > last purchase
                        predictStackList.append(frameClient_i.array)
                        userPredictList.append(client)
                        datePredictList.append(dRight)                 
                        continue
                    else:
                        # to train
                        trainStackList.append(frameClient_i.array)
                        responseTrainList.append(0)
                        userTrainList.append(client)
                        dateTrainList.append(dRight)
                elif status == 1:
                    trainStackList.append(frameClient_i.array)
                    responseTrainList.append(1)
                    userTrainList.append(client)
                    dateTrainList.append(dRight)                                       
                else:
                    # non-born
                    pass                    
        
        dfTrain = pd.DataFrame(columns=['user', 'date', 'y'])
        dfTrain['user'] = userTrainList
        dfTrain['date'] = dateTrainList
        dfTrain['y'] = responseTrainList
        
        dfPredict = pd.DataFrame(columns=['user', 'date'])
        dfPredict['user'] = userPredictList
        dfPredict['date'] = datePredictList
            
        xTrain = np.stack(trainStackList, axis=0)
        xPredict = np.stack(predictStackList, axis=0) 
        
        # reduce memory
        xTrain = xTrain.astype(np.float32)
        xPredict = xPredict.astype(np.float32)
        
        if np.isnan(xTrain).sum() > 0 or np.isnan(xPredict).sum() > 0:
            raise ValueError('Nan in ANN input array')
        
        if np.isinf(xTrain).sum() > 0 or np.isinf(xPredict).sum() > 0:
            raise ValueError('Inf in ANN input array')
        
        return xTrain, xPredict, dfTrain, dfPredict
    
    def getLastUserState(self):
        """
        Retreive last users state: churn state if dead
        last state for censoring clients is the one when last purchase was made
        
        Skip not born clients
    
        Parameters
        ----------
        stateLast : TYPE
            DESCRIPTION.
    
        Raises
        ------
        ValueError
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        usersList = []
        dBirthStateList = []
        dLastStateList = []
        statusList = []
        stateLast = self.loadLastState()
        dLastState = stateLast.dState
        for client, clientObj in stateLast.Clients.items():
            status = self.mapClientToStatus.get(client)
            if status is None:
                raise ValueError('Client: {} has no status in states.mapClientToStatus'.format(client))        
            if status == 3:
                # not born
                continue
            
            dBirthState = clientObj.descriptors.get('dBirthState')            
            dBirthStateList.append(dBirthState)            
            usersList.append(client)
            statusList.append(status)
            if status == 0:
                # censored
                # last state for censoring clients is the one when last purchase was made
                dLastEventState = clientObj.descriptors['dLastEventState']
                dLastStateList.append(dLastEventState)                   
            elif status == 1:
                # dead
                # last state for dead clients is the one when 0.5 quantile is after last purchase (dChurnState)
                dChurnState = clientObj.descriptors['dChurnState']                
                if dChurnState is None:
                    raise ValueError('Client: {}. Status: {}. State : {}. dChurnState is None'.format(client, status, dLastState))
                dLastStateList.append(dChurnState)
            else:
                raise ValueError('Client: {}. Status: {}. At last {}'.format(client, status, dLastState))

        df = pd.DataFrame(columns=['iD', 'user', 'dBirthState', 'dLastState', 'status'])
        df['user'] = usersList
        df['dBirthState'] = dBirthStateList
        df['dLastState'] = dLastStateList
        df['status'] = statusList
        df['iD'] = df['user'] + ' | ' + df['dLastState'].astype(str)
        df.set_index('iD', drop=True, inplace=True)    
        return df

    def buildPersonPeriodFrame(self, features, mapClientToFrame):
        """
        For COX TV

        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        mapClientToFrame : TYPE
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
        lastStateDf = self.getLastUserState()
        dStateList = self.dStateExistList
        
        dataList = []
        print('Build person-period data')
        for idx, row in tqdm(lastStateDf.iterrows()):
            client = row['user']
            dBirthState = row['dBirthState']
            dLastState = row['dLastState']
            statusLast = row['status']                     
            df1 = libSurvival.build_TV_data1(client, dBirthState, dLastState, statusLast, dStateList, mapClientToFrame, features)
                
            if df1 is not None:
                dataList.append(df1)   
            # else:
            #     raise RuntimeError('Cannot create person-period frame for client {}'.format(client))
            
        df = pd.concat(dataList, axis=0)
        return df
        
        
   
    



