# dState: date of state
# birth: time at 3rd event
# birthState: birth shifted towards the date of more recent state
# death: time equal to 90% quantile of all time intervals between events * 1.5
# deathState: death shifted towards the date of more recent state
# firstEvent: date of first event
# lastEvent: date of last event
# lastEventState: date of last event shifted towards the date of more recent state
# thirdEvent: date of third event

# loyalty: duration between first and last events
# loyaltyState: duration between birth state and lastEventState
# loyaltyTS: == loyaltyState / TS_STEP
# recency: duration between lastEvent and dState
# recencyState: duration between lastEventAdj and dState
# recencyTS: recencyState / TS_STEP
# T: time since first event to dState; == recency + loyalty; ignores death event
# T_State: since birth to dState
# T_TS: since birth to dState in state units; if time between states == TS_STEP (week), T_TS = int(T_State) / TS_STEP
# D: time interval between first event and death if dead or between first event and dState if alive
# D_State: time interval between birth state and churn state if dead or between birth and dState if alive
# D_TS: int(D_State) / TS_STEP
# C_Dict : clumpiness from first to last purchase (last restrictd by state)
# C_OrigDict : clumpiness from first purchase to state
# statsDict : client -> Statistics
# 'ratePoisson', 
# 'pPoisson', 
# 'tPoissonLifetime', 
# 'tPoissonDeath'


# periods: chronological list of durations between i and i+1 consecutive events i == [1..N]




import os
import numpy as np
import lib3
from datetime import timedelta
import lib2
from time import time
# import warnings
from math import log
from math import exp
from Rank import Rank
from Distributions import Statistics
# from Distributions import LogNormal
# from Distributions import GeneralizedGamma
# from Distributions import Gamma
# from Distributions import Normal
# from Distributions import Weibull
# from Distributions import Exponential
from collections.abc import Iterable
# from tqdm import tqdm
# from datetime import date
from ClientTS import ClientTS as Client
from pathlib import Path
# import States
from copy import deepcopy
#%%
MIN_FREQUENCY = 3
TS_STEP = 7
INCLUDE_START = True
INCLUDE_END = True
P_CHURN = 0.5 # quantile from CDF of best distribution to consider customers illness; 
# lifetime is time from birth to last event +  PPF(P_CHURN)
P_DEATH = 0.98 # quantile from CDF of best distribution to consider customers death

#%%
def calcC(durations, span=None):
    """
    June 2022
    Modified from article: Predicting Customer Value Using Clumpiness: From RFM to RFMC
    if span is None, Interval between last event end end of period is not considered
    if span is provived, lastInterval == span - np.sum(durations) will be added (according to article)
        
    
    Parameters
    ----------
    durations : list
        inter-event times

    span : numeric or None
        see func. description
        
    Returns
    -------
    Clumpiness

    """
    durationsList = list(durations)
    if len(durationsList) < 2:
        raise ValueError('2 is minimum length of span requered')
    
    if span is not None:
        # span += 1
        lastInterval = span - np.sum(durationsList)
        if lastInterval < 0:
            raise ValueError('span: {} shorter that sum of intervals: {}'.format(span, np.sum(durationsList)))
        if lastInterval > 0:
            durationsList.append(lastInterval)
    else:
        span = np.sum(durationsList)

    n = len(durationsList)
    durationsScaled = [x / span for x in durationsList]
    return float(1 + np.sum(np.log(durationsScaled) * durationsScaled) / np.log(n))

#%%
    
class State(object):
    """
    
    
    """
    activityRFM = [1, 2, 3, 4, 5, 6, 7, 9, 11, 14] # fit RFM clusters on those activity
    activityAll = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # activityANN_0 = [1, 2, 3, 14] # ANN response == 0
    # activityANN_1 = [8, 12] # ANN response == 1
    # activityANN_Predict = [4, 5, 6, 7, 9, 10, 11] # ANN predict where this activitys exist

    def __init__(self, dDate, dZeroState, transactionsDf, mapClientToStatus, prevState=None):
    
        # state variables        
        # self.mapClientToStatus = mapClientToStatus
        self.bTrends = False # when set trends, make it true
        self.longTrendLen = None
        self.shortTrendLen = None
        self.stateId = None # startes from 0; assigned by makeTS_Dates
        self.dState = dDate
        self.dZeroState = dZeroState
        self.firstTransaction = transactionsDf['ds'].min()
        self.lastTransaction = transactionsDf['ds'].max()        
        # self.transactionsCut = None
        
        self.Clients = {} # clientId -> Client(object)
        
        # from all (non-cut) transactions; info will not be in Clients
        # self.mapUserToPurchases = None # 
        
        # clients
        # self.clientsAll = None
        # self.clientsActive = None # active that have next purchase within short period
        # self.clientsOldDead = None # died before
        # self.clientsJustLost = None # set of clients that died at this state
        # self.clientsNotBorn = None # nonBorn, died before zero state
        # self.clientsCensored = None # active that have last purchase close to ent of transactions        
      
        self.makeTS_Dates() # self.datesList: dates of all previos states + current        
        start_time = time()

# not cut by state : required for death events
        # self.mapUserToPurchases = getUserToPurchases(transactionsDf, 'iId', 'ds')                    
        
        transactionsCut = lib3.cutTransactions(transactionsDf, 'iId', 'ds', 
            dStart=None, dEnd=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY, dMaxAbsence=None)

        if len(transactionsCut) == 0:
            raise RuntimeError('No transactions')
            
        # print(len(self.transactionsCut))
        
        firstEventDict = lib3.getMarginalEvents(transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='first')  
        self.putDescriptor('dFirstEvent', firstEventDict)
            
        lastEventDict = lib3.getMarginalEvents(transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='last')
        self.putDescriptor('dLastEvent', lastEventDict)

        thirdEventDict = lib3.getNthEvent(transactionsCut, 'iId', 'ds', 3)
        self.putDescriptor('dThirdEvent', thirdEventDict)
                
        periodsDict = lib3.getTimeBetweenEvents(transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, 
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY)
        self.putDescriptor('periods', periodsDict)

        frequencyDict = lib3.getFrequency(transactionsCut, 'iId', 'ds',\
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START,\
            bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY, exceptFirst=False)
        self.putDescriptor('frequency', frequencyDict)
                    
        
        self.setStats() # self.statsDict
        self.setRecency() # self.recencyDict, self.recencyStateDict, self.recencyTS_Dict                         
        self.setBirth() # self.birthStateDict
        self.setLoyalty() # loyaltyDict, loyaltyStateDict, loyaltyTS_Dict
        self.setT() # T_Dict, T_StateDict, T_TS_Dict
        self.getC() # C_Dict, C_OrigDict        
        self.setActivity(mapClientToStatus, prevState) # see method
        # self.setClients() # lists of clients according to status        
        self.setD() # D_Dict, D_StateDict, D_TS_Dict
        self.getRatePoisson() # Poisson features, see method for details
        # total spending            
        moneySumDict = lib3.getValue(transactionsCut, 'iId', 'ds', 'Sale',
            value='sum', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)
        self.putDescriptor('moneySum', moneySumDict)
        
        moneyMedianDict = lib3.getValue(transactionsCut, 'iId', 'ds', 'Sale',
            value='median', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)
        self.putDescriptor('moneyMedian', moneyMedianDict)

        self.setMoneyDaily() # daily spending   'moneyDaily', 'moneyDailyStep'      

        rank10 = Rank(method='Quantiles', nBins=10, dropZero=False, \
            returnZeroClass=True, includeMargin='right', zeroCenter=False)

#!!! Rank only alive customers, even if dead are just dormant
                    
        # Rank recency
        recencyActiveDict = self.getDescriptor('recency', activity=self.activityRFM, error='raise')
        rank10.fit(recencyActiveDict)
        recencyDict = self.getDescriptor('recency', activity=None, error='raise')                
        r10_R = rank10.getClusters(recencyDict, reverse=True)
        self.putDescriptor('r10_R', r10_R)
        
        # Rank frequency 
        frequencyActiveDict = self.getDescriptor('frequency', activity=self.activityRFM, error='raise')
        rank10.fit(frequencyActiveDict)
        frequencyDict = self.getDescriptor('frequency', activity=None, error='raise')                
        r10_F = rank10.getClusters(frequencyDict, reverse=False)
        self.putDescriptor('r10_F', r10_F)
        
        # Rank daily monetary value moneyDailyStep or moneyDaily
        moneyDailyActiveDict = self.getDescriptor('moneyDailyStep', activity=self.activityRFM, error='raise')
        rank10.fit(moneyDailyActiveDict)
        moneyDailyDict = self.getDescriptor('moneyDailyStep', activity=None, error='raise')                
        r10_M = rank10.getClusters(moneyDailyDict, reverse=False)
        self.putDescriptor('r10_M', r10_M)
        
        # Rank loyalty
        loyaltyActiveDict = self.getDescriptor('loyaltyState', activity=self.activityRFM, error='raise')
        rank10.fit(loyaltyActiveDict)
        loyaltyDict = self.getDescriptor('loyaltyState', activity=None, error='raise')                
        r10_L = rank10.getClusters(loyaltyDict, reverse=False)
        self.putDescriptor('r10_L', r10_L)
        
        # Rank climpiness; step, fit on active and dead
        # cDict = self.getDescriptor('C', activity=None, error='raise')
        # rank10.fit(cDict)
        # r10_C = rank10.getClusters(cDict, reverse=True)
        # self.putDescriptor('r10_C', r10_C)        
        
        # Rank climpiness original, fit on active and dead
        # cOrigDict = self.getDescriptor('C_Orig', activity=None, error='raise')
        # rank10.fit(cOrigDict)
        # r10_C_Orig = rank10.getClusters(cOrigDict, reverse=True)
        # self.putDescriptor('r10_C_Orig', r10_C_Orig)         
                
        # M * C
        # r10_MC = State.makeComplexRank([r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        # self.putDescriptor('r10_MC', r10_MC)

        # F * C
        # r10_FC = State.makeComplexRank([r10_F, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        # self.putDescriptor('r10_FC', r10_FC)

        # R * F * C
        # r10_FMC = State.makeComplexRank([r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        # self.putDescriptor('r10_FMC', r10_FMC)        

        # L * F * M * C
        # r10_LFMC = State.makeComplexRank([r10_L, r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        # self.putDescriptor('r10_LFMC', r10_LFMC)
        
        # R * F
        r10_RF = State.makeComplexRank([r10_R, r10_F], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RF', r10_RF)

        # F * M
        r10_FM = State.makeComplexRank([r10_F, r10_M], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_FM', r10_FM)
        
        # R * F * M
        r10_RFM = State.makeComplexRank([r10_R, r10_F, r10_M], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RFM', r10_RFM)

        # R * F * M * L
        r10_RFML = State.makeComplexRank([r10_R, r10_F, r10_M, r10_L], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RFML', r10_RFML)
    
        # R * F * M * C        
        # r10_RFMC = State.makeComplexRank([r10_R, r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        # self.putDescriptor('r10_RFMC', r10_RFMC)
 
        end_time = time()
        self.initTime = end_time - start_time        
        # self.printTime()
        return

#%%
    @staticmethod
    def roundToNext(dDate, dList):
        for d in sorted(dList):
            if d >= dDate:
                return d
        return None
    
#%%    
    @staticmethod
    def checkDictForNaN(d):
        """
        
        Parameters
        ----------
        d : dict
            any dictionary.

        Returns
        -------
        bool
            True if there is no None or NaN.

        """
        for key, value in d.items():
            if lib2.isNaN(value):
                return False
        return True
    
#%%    
    @staticmethod
    def makeComplexRank(ranks, model, weights=None, nBins=10):
        """
        Fixed Jul 2022
        Parameters
        ----------
        ranks : list of dict
            all dict have same keys
        model : str, optional
            'additive' or 'multiplicative' 
        weights : None of list of weights with length equal to len(ranks)
            default is None; all weights are 1
        Returns
        -------
        scaledDict

        """
        if weights is not None:
            w = list(weights)
            if len(w) != len(ranks):
                raise ValueError('len of weights = {} mismatches len of ranks = {}'.format(weights, ranks))
        else:
            w = [1 for x in range(0, len(ranks), 1)]
            
        comboDict = {}
        # minX = np.inf
        # maxX = -np.inf
        for client in ranks[0].keys():
            xList = []
            for dict1 in ranks:
                value = dict1.get(client)
                if value is None:
                    raise ValueError('client {} does not have value in dictionary'.format(client))
                xList.append(value) # [0..1]
            # multiply each value by weight
            xW = [i * j for i, j in zip(xList, w)]
            if model == 'additive':
                X = sum(xW)
            elif model == 'multiplicative':
                X = np.prod(xW)
            else:
                raise ValueError('Wrong model in makeComplexRank')
            comboDict[client] = X
            # minX = min(minX, X)
            # maxX = max(maxX, X)

        rank = Rank(method='Quantiles', nBins=nBins , dropZero=False, \
            returnZeroClass=True, includeMargin='right', zeroCenter=False)
        scaledDict = rank.fitGetClusters(comboDict, reverse=False)

        return scaledDict        
        
#%%
    def putDescriptor(self, sDescriptor, clientDict):
        for client, value in clientDict.items():
            clientObject = self.Clients.get(client) 
            if clientObject is None:
                clientObject = Client(client)# creare instance
            clientObject.descriptors[sDescriptor] = value
            self.Clients[client] = clientObject
        return

#%%
    def getDescriptor(self, sDescriptor, activity=None, error='ignore'):
        """
        make dict : client -> value
        get values from self.Clients that correspond to descriptor sDescriptor

        Parameters
        ----------
        sDescriptor : str
            feature name from Clients object.
        activity : int, optional
            filter clients according to activity. One value or list. The default is None: all clients. 
        error : str, optional
            if 'raise' : raises error if activity or feature are not found in data for any client. The default is 'ignore'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        oneDescriptorDict : dict
            client -> value

        """
        if activity is not None:
            if isinstance(activity, Iterable):
                mask = list(activity)
            else:
                mask = [activity]
    
        oneDescriptorDict = {}
        for client, clientObj in self.Clients.items():
            value = clientObj.descriptors.get(sDescriptor)
            if error == 'raise' and value is None:
                raise ValueError('Customer {} has no descriptor {} at {}'.format(client, sDescriptor, self.dState))
            if activity is not None:
                a = clientObj.descriptors.get('activity')
                if error == 'raise' and a is None:
                    raise ValueError('Customer {} has no activity at {}'.format(client, self.dState))
                if a in mask:
                    oneDescriptorDict[client] = value
            else:
                oneDescriptorDict[client] = value
        return oneDescriptorDict

#%%        
    def setMoneyDaily(self):
        """
        Set : 'moneyDaily', 'moneyDailyStep'

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))
            moneySum = clientObj.descriptors.get('moneySum')
            if moneySum is None:
                raise RuntimeError('customer {} has no moneySum'.format(client))            
            loyalty = clientObj.descriptors.get('loyalty')
            if loyalty is None:
                raise RuntimeError('customer {} has no loyalty'.format(client))          
            loyalty = max(1, loyalty) # avoid division by 0
            T = clientObj.descriptors.get('T')
            if T is None:
                raise RuntimeError('customer {} has no T'.format(client))          
            T = max(1, T) # avoid division by 0
            clientObj.descriptors['moneyDailyStep'] = moneySum / loyalty
            clientObj.descriptors['moneyDaily'] = moneySum / T
            self.Clients[client] = clientObj
        return

#%%
    def printTime(self):
        d, h, m, s = lib2.formatTime(self.initTime)
        print('Time elapsed:', d, 'days', h, 'hours', m, 'min', s, 'sec')
        return

#%%
    def setStats(self):
        """
        Fit set of distributions to sequence of inter-purchase interval for each client
        see details of Statistics in Distributions.py

        Returns
        -------
        None.

        """            
            
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))
            periods = clientObj.descriptors.get('periods')
            if periods is None:
                raise RuntimeError('customer {} has no inter-purchase intervals'.format(client)) 
            stats = Statistics(periods, models=['Lognormal', 'Exponential'])
            clientObj.stats = deepcopy(stats)
            self.Clients[client] = clientObj
        return

#%%
#!!!
    def setActivity(self, mapClientToStatus, prevState):
        """
        Check customers activity, assigne dead or alive
        If dead, get death and observed death dates

        Set : 'activity' , 'dChurn', 'dChurnState', 'dDeath', 'dDeathState', 'qChurn', 'qDeath'
            
        activity :           ANN response
            1 : active         0
            2 : awaken         0
            3 : reincarnated   0
            4 : recent         ? question ? or 0
            5 : late           ?
            6 : churn          1
            7 : coma           ?
            8 : dead           1
            9 : hibernating    ?
            10 : old churn     1
            11 : old coma      ?
            12 : corpse        1
            13 : not born      ?
            14 : new           0

            
        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.dZeroState
        
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                
            stats = clientObj.stats
            if stats is None:
                raise RuntimeError('customer {} has no stats instance'.format(client))                 
            dLastEvent = clientObj.descriptors.get('dLastEvent')
            # from states point of view
            if dLastEvent is None:
                raise RuntimeError('customer {} has no dLastEvent'.format(client)) 
            # dEventsList = self.mapUserToPurchases.get(client) # all clients purchases GLOBAL
            # if dEventsList is None:
                # raise RuntimeError('State Error for customer {}. No purchases in self.mapUserToPurchases'.format(client))            
            qChurn = round(stats.bestModel.ppf(P_CHURN)) # 0.5 from last event                            
            qDeath = round(stats.bestModel.ppf(P_DEATH)) # 0.98 from last event
            if lib2.isNaN(qDeath) or qDeath == np.inf:
                raise ValueError('Customer {} has qDeath = {}. Params: {}. Data: {}'.\
                    format(client, qDeath, stats.bestModel.params, stats.data))            

            dChurn = dLastEvent + timedelta(days=qChurn) # date of churn
            dChurnState = State.roundToNext(dChurn, self.datesList) # None or some state
            dDeath = dLastEvent + timedelta(days=qDeath) # date of death
            dDeathState = State.roundToNext(dDeath, self.datesList) # None or some state
            
            status = mapClientToStatus.get(client)
            if status is None:
                raise ValueError('Client: {} has no status in mapClientToStatus'.format(client))
                      
            activity = None
            if status == 3:
                # not born
                activity = 13
                    
            elif (prevState is None or self.dState == self.dZeroState) and status != 3:
                # at zero state; dead or alive, but will have transactions
                # skip all point activities
                if dLastEvent + timedelta(days=qChurn) >= self.dState:
                    activity = 4 #recent
                elif dLastEvent + timedelta(days=qDeath) > self.dState:
                    activity = 9 # hibernating
                else:
                    activity = 11 # old coma
                
            else:
                # all states except zero state
                if (0 <= (self.dState - dLastEvent).days) and ((self.dState - dLastEvent).days < TS_STEP):
                    # active, awaken, reincarnated
                    prevClientObj = prevState.Clients.get(client)
                    if prevClientObj is None:
                        # new born
                        activity = 14 # new
                    else:
                        prevActivity = prevClientObj.descriptors.get('activity')
                        if prevActivity is None:
                            raise ValueError('Client: {} has no activity at dState: {}'.format(client, prevState.dState))
                        if prevActivity == 9: # hibernating
                            activity = 2 # awaken
                        elif prevActivity in [7, 11]: # coma or old coma
                            activity = 3 # reincarnated
                        else:
                            activity = 1 # active
        
                elif (0 <= (self.dState - (dLastEvent + timedelta(days=qChurn))).days) and \
                    ((self.dState - (dLastEvent + timedelta(days=qChurn))).days < TS_STEP):
                    # late, churn
                    if status == 0: # censored
                        activity = 5 # late
                    elif status == 1: # dead
                        activity = 6 # churn
                    else:
                        raise RuntimeError('This must not has happened 1')
                        
                elif (0 <= (self.dState - (dLastEvent + timedelta(days=qDeath))).days) and \
                    ((self.dState - (dLastEvent + timedelta(days=qDeath))).days < TS_STEP):
                    # coma, dead
                    if status == 0: # censored
                        activity = 7 # coma
                    elif status == 1: # dead
                        activity = 8 # dead
                    else:
                        raise RuntimeError('This must not has happened 2')
            
                elif self.dState < (dLastEvent + timedelta(days=qChurn)):
                    activity = 4 # recent
                    
                elif ((dLastEvent + timedelta(days=qChurn)) < self.dState) and \
                    (self.dState < (dLastEvent + timedelta(days=qDeath))):
                    # hibernating, old churn
                    if status == 0: # censored
                        activity = 9 # hibernating
                    elif status == 1: # dead
                        activity = 10 # old churn
                    else:
                        raise RuntimeError('This must not has happened 3')            
            
                elif (dLastEvent + timedelta(days=qDeath)) < self.dState:
                    # old coma, corpse
                    if status == 0: # censored
                        activity = 11 # old coma
                    elif status == 1: # dead
                        activity = 12 # corpse
                    else:
                        raise RuntimeError('This must not has happened 4')
            
                else:                    
                    print('all variables out, error here')
                    raise RuntimeError('This must not has happened 5')
                    
            if activity is None:
                raise RuntimeError('This must not has happened 6')

            clientObj.descriptors['activity'] = activity
            clientObj.descriptors['dChurn'] = dChurn
            clientObj.descriptors['dChurnState'] = dChurnState
            clientObj.descriptors['dDeath'] = dDeath
            clientObj.descriptors['dDeathState'] = dDeathState
            clientObj.descriptors['qChurn'] = qChurn
            clientObj.descriptors['qDeath'] = qDeath            
                            
            self.Clients[client] = clientObj
                        
        return
    
#%%    
    def getC(self):    
        """
        Calculete clumpiness from transactions data. Source:
        Predicting Customer Value Using Clumpiness: From RFM to RFMC 
            
        """        

        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                                
            T = clientObj.descriptors.get('T')
            if T is None:
                raise RuntimeError('customer {} has no T'.format(client)) 
            periods = clientObj.descriptors.get('periods')
            if periods is None:
                raise RuntimeError('customer {} has no periods'.format(client))                 
            c = calcC(periods, span=None)
            cOrig = calcC(periods, span=T)
            clientObj.descriptors['C'] = c
            clientObj.descriptors['C_Orig'] = cOrig            
            self.Clients[client] = clientObj

        return        
    
#%%
    def getRatePoisson(self):
        """
        Calculate Poisson rate and probability of event at current state for customers
        Set : 'ratePoisson', 'pPoisson', 'tPoissonLifetime', 'tPoissonDeath'
        
        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                 
            periods = clientObj.descriptors.get('periods')
            if periods is None:
                raise RuntimeError('customer {} has no periods'.format(client)) 
            if sum(periods) <= 0:
                raise RuntimeError('customer {} has crap periods: {}'.format(client, periods)) 
            recency = clientObj.descriptors.get('recency')
            if recency is None:
                raise RuntimeError('customer {} has no recency'.format(client))                 
            ratePoisson = len(periods) / sum(periods)
            pPoisson = 1 - exp(-ratePoisson * recency)                                                
            clientObj.descriptors['ratePoisson'] = ratePoisson
            clientObj.descriptors['pPoisson'] = pPoisson   
            clientObj.descriptors['tPoissonLifetime'] = log(1-P_CHURN) / (-ratePoisson)
            clientObj.descriptors['tPoissonDeath'] = log(1-P_DEATH) / (-ratePoisson)             
            self.Clients[client] = clientObj
             
        return

#%%
    # def setClients(self):
    #     """
    #     set : 'clientsActive', 'clientsOldDead', 'clientsJustLost', 'clientsNotBorn', 'clientsCensored', 'clientsAll'

    #     Raises
    #     ------
    #     RuntimeError
    #         DESCRIPTION.
    #     ValueError
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     clientsActive = []
    #     clientsOldDead = []
    #     clientsJustLost = []
    #     clientsNotBorn = []
    #     clientsCensored = []
        
    #     for client in self.Clients.keys():
    #         clientObj = self.Clients.get(client)
    #         if clientObj is None:
    #             raise RuntimeError('customer {} has no instance'.format(client))                  
    #         status = clientObj.descriptors.get('status')
    #         if status is None:
    #             raise RuntimeError('customer {} has no status'.format(client))  
            
    #         if status == 0:
    #             clientsActive.append(client)
    #         elif status == 1:
    #             clientsOldDead.append(client)
    #         elif status == 2:
    #             clientsJustLost.append(client)
    #         elif status == 3:
    #             clientsNotBorn.append(client)
    #         elif status == 4:
    #             clientsCensored.append(client)
    #         else:
    #             raise ValueError('Wrong status: {}'.format(status))
                
    #     self.clientsActive = sorted(clientsActive)
    #     self.clientsOldDead = sorted(clientsOldDead)
    #     self.clientsJustLost = sorted(clientsJustLost)
    #     self.clientsNotBorn = sorted(clientsNotBorn) 
    #     self.clientsCensored = sorted(clientsCensored) 
    #     self.clientsAll = sorted(list(self.Clients.keys()))                
    #     return
    
#%%    
    def makeTS_Dates(self):
        i = 0
        dState = self.dZeroState
        self.datesList = []
        while dState < self.dState:
            self.datesList.append(dState)
            dState += timedelta(days=TS_STEP)
            i += 1
        self.datesList.append(self.dState) # fixes frist state problem
        self.stateId = i
        return
    
#%%    
    def setBirth(self):
        """
        set 'dBirthState'

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
                
                
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                
            dThirdEvent = clientObj.descriptors.get('dThirdEvent')
            dEventAdj = State.roundToNext(dThirdEvent, self.datesList)            
            if dEventAdj is None:
                raise RuntimeError('customer {} has dThirdEvent at {} which lies beyound last state {}'.format(client, dThirdEvent, self.datesList))
            clientObj.descriptors['dBirthState'] = dEventAdj            
            self.Clients[client] = clientObj
                
        return
    
#%%    
    def setRecency(self):
        """
        calculate 'recency', 'lastEventState', 'recencyState', 'recencyTS'     

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
                
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))
            dLastEvent = clientObj.descriptors.get('dLastEvent')
            if dLastEvent is None:
                raise RuntimeError('customer {} has no dLastEvent'.format(client))             
            recency = (self.dState - dLastEvent).days
            dLastState = State.roundToNext(dLastEvent, self.datesList)            
            if dLastState is None:
                raise RuntimeError('customer {} has dLastState = {} which lies beyound last state {}'.format(client, dLastState, self.datesList[-1]))
            recencyState = (self.dState - dLastState).days
            clientObj.descriptors['recency'] = recency
            clientObj.descriptors['dLastEventState'] = dLastState
            clientObj.descriptors['recencyState'] = recencyState
            clientObj.descriptors['recencyTS'] = round(recencyState / TS_STEP)
            self.Clients[client] = clientObj                            
            
        return
    
#%%    
    def setLoyalty(self):
        """
        set 'loyalty', 'loyaltyState', 'loyaltyTS'

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                                                 
            dBirthState = clientObj.descriptors.get('dBirthState')
            if dBirthState is None:
                raise RuntimeError('customer {} has no dBirthState'.format(client))
            dLastEvent = clientObj.descriptors.get('dLastEvent')
            if dLastEvent is None:
                raise RuntimeError('customer {} has no dLastEvent'.format(client))            
            dFirstEvent = clientObj.descriptors.get('dFirstEvent')
            if dFirstEvent is None:
                raise RuntimeError('customer {} has no dFirstEvent'.format(client)) 
            dLastEventState = clientObj.descriptors.get('dLastEventState')
            if dLastEventState is None:
                raise RuntimeError('customer {} has no dLastEventState'.format(client))                 
            loyalty = (dLastEvent - dFirstEvent).days
            loyaltyState = (dLastEventState - dBirthState).days            
            clientObj.descriptors['loyalty'] = loyalty
            clientObj.descriptors['loyaltyState'] = loyaltyState
            clientObj.descriptors['loyaltyTS'] = round(loyaltyState / TS_STEP)                                  
            self.Clients[client] = clientObj
            
        return
        
#%%
    def setT(self):
        """
        set 'T', 'T_State', 'T_TS'

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                                                 
            recency = clientObj.descriptors.get('recency')
            if recency is None:
                raise RuntimeError('customer {} has no recency'.format(client))
            loyalty = clientObj.descriptors.get('loyalty')
            if loyalty is None:
                raise RuntimeError('customer {} has no loyalty'.format(client))            
            recencyState = clientObj.descriptors.get('recencyState')
            if recencyState is None:
                raise RuntimeError('customer {} has no recencyState'.format(client))                 
            loyaltyState = clientObj.descriptors.get('loyaltyState')
            if loyaltyState is None:
                raise RuntimeError('customer {} has no loyaltyState'.format(client))                  
            T_State = recencyState + loyaltyState
            clientObj.descriptors['T'] = recency + loyalty
            clientObj.descriptors['T_State'] = T_State
            clientObj.descriptors['T_TS'] = round(T_State / TS_STEP)                                             
            self.Clients[client] = clientObj
            
        return        
        
#%%    
    def setD(self):
        """
        SEt : 'D', 'D_State', 'D_TS'

        Raises
        ------
        RuntimeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client)) 
            dFirstEvent = clientObj.descriptors.get('dFirstEvent')
            if dFirstEvent is None:
                raise RuntimeError('customer {} has no dFirstEvent'.format(client))             
            dBirthState = clientObj.descriptors.get('dBirthState')
            if dBirthState is None:
                raise RuntimeError('customer {} has no dBirthState'.format(client))

            activity = clientObj.descriptors.get('activity')
            if activity is None:
                raise RuntimeError('customer {} has no activity'.format(client))
                          
            dChurn = clientObj.descriptors.get('dChurn')
            if dChurn is None:
                raise RuntimeError('customer {} has no dChurn'.format(client))                 
                    
            if activity in [1, 2, 3, 4, 5, 7, 9, 11, 14]:
                # 1 : active
                # 2 : awaken
                # 3 : reincarnated
                # 4 : recent
                # 5 : late
                # 7 : coma
                # 9 : hibernating
                # 11 : old coma     
                # 14 : new
                D = (self.dState - dFirstEvent).days
                D_State = (self.dState - dBirthState).days
            elif activity in [6, 8, 10, 12]:
                # 6 : churn
                # 8 : dead
                # 10 : old churn
                # 12 : corpse          
                dChurnState = clientObj.descriptors.get('dChurnState')                
                if dChurnState is None:
                    raise RuntimeError('customer {} has no dChurnState'.format(client))               
                D = (dChurn - dFirstEvent).days
                D_State = (dChurnState - dBirthState).days
            elif activity == 13:
                # not born
                D = (dChurn - dFirstEvent).days
                D_State = 0
            else:
                raise ValueError('Wrong activity: {}'.format(activity))
                
            clientObj.descriptors['D'] = D
            clientObj.descriptors['D_State'] = D_State
            clientObj.descriptors['D_TS'] = round(D_State / TS_STEP)            
            self.Clients[client] = clientObj
                            
        return            

#%%        
    def save(self, folder=None):
        fileName = '{}.dat'.format(self.dState)
        if folder is not None:            
            if not Path(folder).exists():
                raise RuntimeError('Folder {} does not exist'.format(folder))
            fileName = os.path.join(folder, fileName)
        lib2.saveObject(fileName, self, protocol=4)
        return
        
#%%        
    def getClients(self, **kwargs):
        """
        Filter existing clients according to activity and / or status

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        activityExclude = kwargs.get('activityExclude', [])
        activityOnly = kwargs.get('activityOnly', self.activityAll)
        feasibleOnly = kwargs.get('feasibleOnly', False)
        mapClientToStatus = kwargs.get('mapClientToStatus', None)
        statusOnly = kwargs.get('statusOnly', None)
        
        if len(kwargs) == 0:
            return sorted(list(self.Clients.keys()))
        
        activityList = self.activityAll.copy()
        activityList = [x for x in activityList if x not in activityExclude] # exclude
        activityList = [x for x in activityList if x in activityOnly] # activityOnly
        if feasibleOnly:
            activityList = [x for x in activityList if x != 13] # exclude not born
        
        clients = []
        for client, clientObj in self.Clients.items():
            match = False
            activity = clientObj.descriptors.get('activity')
            if activity is None:
                raise ValueError('Client: {} has no activity at state: {}'.format(client, self.dState))
            if activity in activityList:
                match = True
            if isinstance(statusOnly, Iterable) and mapClientToStatus is not None:
                status = mapClientToStatus.get(client, -1)
                if status not in statusOnly:
                    match = False
            if match:
                clients.append(client)
                                            
        return sorted(clients)
        