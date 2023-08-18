# dState: date of state
# birth: time at 3rd event
# birthState: birth shifted towards the date of more recent state
# death: time equal to 90% quantile of all time intervals between events * 1.5
# deathState: death shifted towards the date of more recent state
# firstEvent: date of first event
# lastEvent: date of last event
# lastEventState: date of last event shifted towards the date of more recent state
# thirdEvent: date of third event
# lastEventAdj: date of last event shifted towards the date of more recent state

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
# D_State: time interval between birth state and death state if dead or between birth and dState if alive
# D_TS: int(D_State) / TS_STEP
# C_Dict : clumpiness from first to last purchase (last restrictd by state)
# C_OrigDict : clumpiness from first purchase to state
# statsDict : client -> Statistics

# lost: 1 if death occured before dState, 0 otherwise

# periods: chronological list of durations between i and i+1 consecutive events i == [1..N]
# q05: 50% quantile of periods
# q09: 90% quantile of periods

# pEvent: probability of event based on general logistic function fitted on periods
# timeTo05: time duration from dState to pEvent == 0.5
# timeTo09: time duration from dState to pEvent == 0.9


# import os
# import libStent
# from datetime import datetime
# import pandas as pd
# import libPlot
import numpy as np
import lib3
from datetime import timedelta
import lib2
from time import time
# import warnings
# from math import log
from math import exp
from Rank import Rank
from Distributions import Statistics
from Distributions import LogNormal
from Distributions import GeneralizedGamma
from Distributions import Gamma
from Distributions import Normal
from Distributions import Weibull
from Distributions import Exponential
from collections.abc import Iterable
from tqdm import tqdm
from datetime import date

#%%
MIN_FREQUENCY = 3
TS_STEP = 7
K = 1.5
INCLUDE_START = True
INCLUDE_END = True
SHORT_SLOPE_SPAN = 10 # SHORT_SLOPE_SPAN * TS_STEP days
P_LIFETIME = 0.5 # quantile from CDF of best distribution to consider customers illness; 
# lifetime is time from birth to last event +  PPF(P_LIFETIME)
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
    return 1 + np.sum(np.log(durationsScaled) * durationsScaled) / np.log(n)

#%%
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

#%%
class Statistic1(object):

    def __init__(self, data):
        self.data = data
        self.bestModel = Weibull()
        args = {'floc': 0}
        self.bestModel.fit(self.data, **args)
        return

#%%
class Statistic2(Statistics):
    
    def __init__(self, data, testStats='ks', criterion='pvalue'):
        self.modelsDict = {'Lognormal': {'model': LogNormal(), 'args': {'floc': 0}},
            'Exponential': {'model': Exponential(), 'args': {'floc': 0}}}        
        self.data = data
        self.bestModel = None
        for sModel, modelDict in self.modelsDict.items():
            _ = modelDict['model'].fit(self.data, **modelDict['args'])
        self.getBestModel(testStats=testStats, criterion=criterion)
        return
#%%
class Client(object):
    
    def __init__(self, clientId):
    
        self.id = clientId
        self.descriptors = {}
        

class State(object):
    """
    
    
    """
    
    def __init__(self, dDate, dZeroState, transactionsDf):
        
        self.stateId = None # startes from 0; assigned by makeTS_Dates
        self.dState = dDate
        self.dZeroState = dZeroState
        
        self.
        # from all (non-cut) transactions
        self.mapUserToPurchases = None
        self.firstTransaction = transactionsDf['ds'].min()
        self.lastTransaction = transactionsDf['ds'].max()        
        
        # set
        self.clientsAll = None
        self.clientsActive = None # active that have next purchase within short period
        self.clientsOldDead = None # died before
        self.clientsJustLost = None # set of clients that died at this state
        self.clientsNotBorn = None # nonBorn, died before zero state
        self.clientsCensored = None # active that have last purchase close to ent of transactions
        
        # dict
        self.firstEventDict = None
        self.lastEventDict = None
        self.lastEventStateDict = None
        self.thirdEventDict = None
        self.birthStateDict = None # 
        self.deathDict = None # date where client is considered to be dead if death event is observed from this state
        self.deathStateDict = None # state where client is considered to be dead if death event is observed from this state
        self.deathObservedDict = None # date where client is seen dead
        self.q05Dict = None
        self.q09Dict = None
        self.statusDict = None
        # self.lostDict = None # see setLost
        self.frequencyDict = None
        self.loyaltyDict = None
        self.loyaltyStateDict = None
        self.loyaltyTS_Dict = None
        self.recencyDict = None
        self.recencyStateDict = None
        self.recencyTS_Dict = None
        self.T_Dict = None # first event to state date
        self.T_StateDict = None # birth to state date
        self.T_TS_Dict = None # T_StateDict in TS units
        self.D_Dict = None # first event to state date if alive, to death if dead
        self.D_StateDict = None # birth to state date if alive, to death if dead
        self.D_TS_Dict = None # D_StateDict in TS units
        self.C_Dict = None # only inter-purchase intervals; step-like shape
        self.C_OrigDict = None # includes recency (interval from last event to dState); smooth shape
        self.statsDict = None
        self.tRemainingObservedDict = None # 95% from distribution
        self.tRemainingEstimateDict = None # 50% from distribution
            
        self.moneySumDict = None # total spending
        self.moneyMedianDict = None # median spending per transaction
        self.moneyDailyDict = None # daily average spending (moneySumDict / T)
        self.moneyDailyStepDict = None # moneySumDict / loyalty
        self.pEventDict = None
        # self.timeTo05Dict = None
        # self.timeTo09Dict = None
        self.ratePoissonDict = None
        self.periodsDict = None
        self.pEventPoissonDict = None
        self.q05PoissonDict = None
        self.q09PoissonDict = None
        self.timeTo05PoissonDict = None
        self.timeTo09PoissonDict = None
        
        # Ranks
        self.r10_Recency = None
        self.r10_Frequency = None          
        self.r10_Loyalty = None
        self.r10_Clumpiness = None
        self.r10_MoneySum = None
        self.r10_MoneyMedian = None
        self.r10_MoneyDaily = None
        
        # Complex Ranks
        self.r10_RFM = None
        self.r10_RF = None
        self.r10_RFMC = None
        self.r10_FM = None
        self.r10_MC = None
        self.r10_FC = None
        self.r10_FMC = None

#       Init trends
        self.trendShort_FM = {}
        self.trendShort_MC = {}
        self.trendShort_FC = {}
        self.trendShort_FMC = {}
        self.trend_FM = {}
        self.trend_MC = {}
        self.trend_FC = {}
        self.trend_FMC = {}
        
        # Trends
      
        self.makeTS_Dates() # self.datesList: dates of all previos states + current
        
        start_time = time()

# not cut by state : required for death events
        self.mapUserToPurchases = getUserToPurchases(transactionsDf, 'iId', 'ds')
        
        self.transactionsCut = lib3.cutTransactions(transactionsDf, 'iId', 'ds', 
            dStart=None, dEnd=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY, dMaxAbsence=None)

        if len(self.transactionsCut) == 0:
            raise RuntimeError('No transactions')
            
        # print(len(self.transactionsCut))
        
        self.firstEventDict = lib3.getMarginalEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='first')  
        
        self.lastEventDict = lib3.getMarginalEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='last')
                
        self.periodsDict = lib3.getTimeBetweenEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, 
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY)

        self.frequencyDict = lib3.getFrequency(self.transactionsCut, 'iId', 'ds',\
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START,\
            bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY, exceptFirst=False)
                    
        self.thirdEventDict = lib3.getNthEvent(self.transactionsCut, 'iId', 'ds', 3)
        
        # print('fit distributions')
        self.setStats() # self.statsDict
        # print('done fit distributions')
        self.setRecency() # self.recencyDict, self.recencyStateDict, self.recencyTS_Dict                         
        self.setBirth() # self.birthStateDict
        self.setLoyalty() # loyaltyDict, loyaltyStateDict, loyaltyTS_Dict
        self.setT() # T_Dict, T_StateDict, T_TS_Dict
        self.getC() # C_Dict, C_OrigDict        
        self.setStatus() # active, dead, just died, not born
        self.setClients() # lists of clients according to status        
        self.setD() # D_Dict, D_StateDict, D_TS_Dict        

        self.getRatePoisson() # make self.ratePoissonDict
        self.getProbabilityPoisson() # self.pEventPoissonDict; uses recency
        self.q05PoissonDict = lib3.getPeriodPoisson(self.periodsDict, p=0.5)
        self.q09PoissonDict = lib3.getPeriodPoisson(self.periodsDict, p=0.9)
        self.setTimeToPoisson() # self.timeTo05PoissonDict , self.timeTo09PoissonDict 
        
        # total spending            
        self.moneySumDict = lib3.getValue(self.transactionsCut, 'iId', 'ds', 'Sale',
            value='sum', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)
        
        self.moneyMedianDict = lib3.getValue(self.transactionsCut, 'iId', 'ds', 'Sale',
            value='median', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)

        self.setMoneyDaily()                

        rank10 = Rank(method='Quantiles', nBins=10, dropZero=False, \
            returnZeroClass=True, includeMargin='right', zeroCenter=False)

        # rank10.fit(self.filterDict(self.recencyStateDict, [0, 4]))
        # self.r10_Recency = rank10.getClusters(self.recencyStateDict, reverse=True)
            
        #!!! Survival features
        #!!! Rank only alive customers, even if dead are just dormant
        
        rank10.fit(self.filterDict(self.frequencyDict, [0, 4]))
        self.r10_Frequency = rank10.getClusters(self.frequencyDict, reverse=False)
        
        # rank10.fit(self.filterDict(self.loyaltyStateDict, [0, 4]))
        # self.r10_Loyalty = rank10.getClusters(self.loyaltyStateDict, reverse=False)        
        
        rank10.fit(self.filterDict(self.C_Dict, [0, 4]))
        self.r10_C = rank10.getClusters(self.C_Dict, reverse=True)

        rank10.fit(self.filterDict(self.moneySumDict, [0, 4]))
        self.r10_MoneySum = rank10.getClusters(self.moneySumDict, reverse=False)

        rank10.fit(self.filterDict(self.moneyMedianDict, [0, 4]))
        self.r10_MoneyMedian = rank10.getClusters(self.moneyMedianDict, reverse=False)
        
        rank10.fit(self.filterDict(self.moneyDailyStepDict, [0, 4]))
        self.r10_MoneyDailyStep = rank10.getClusters(self.moneyDailyStepDict, reverse=False)        
        
        # self.r10_RFM = State.makeComplexRank([self.r10_Recency, self.r10_Frequency, 
        #     self.r10_MoneySum], model='multiplicative')

        # self.r10_RF = State.makeComplexRank([self.r10_Recency, self.r10_Frequency], 
        #     model='multiplicative')

        # self.r10_RFMC = State.makeComplexRank([self.r10_Recency, self.r10_Frequency, 
        #     self.r10_MoneySum, self.r10_Clumpiness], model='multiplicative')
        
        self.r10_FM = State.makeComplexRank([self.r10_Frequency, 
            self.r10_MoneyDailyStep], model='multiplicative')
        
        self.r10_MC = State.makeComplexRank([self.r10_MoneyDailyStep, self.r10_C], model='multiplicative')
        
        self.r10_FC = State.makeComplexRank([self.r10_Frequency, self.r10_C], model='multiplicative')   
        
        self.r10_FMC = State.makeComplexRank([self.r10_C, 
            self.r10_Frequency, self.r10_MoneyDailyStep], model='multiplicative')   
        
        end_time = time()
        self.initTime = end_time - start_time
        
        self.printTime()
            
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
    def makeComplexRank(ranks, model, newRange=(1, 10)):
        """
        Fixed Jul 2022
        Parameters
        ----------
        ranks : list
            list of dict. Assuming all dict have same keys and ranking system is [1..10]
        model : str, optional
            'additive' or 'multiplicative' 

        Returns
        -------
        None.

        """
        comboDict = {}
        minX = np.inf
        maxX = -np.inf
        for client in ranks[0].keys():
            xList = []
            for dict1 in ranks:
                value = dict1.get(client)
                if value is None:
                    raise ValueError('client {} does not have value in dictionary'.format(client))
                xList.append(value) # [0..1]
            if model == 'additive':
                X = sum(xList)
            elif model == 'multiplicative':
                X = np.prod(xList)
            else:
                raise ValueError('Wrong model in makeComplexRank')
            comboDict[client] = X
            minX = min(minX, X)
            maxX = max(maxX, X)

        # rescale to [0..10]
        scaledDict = {}
        OldSpan = maxX - minX
        NewMin = newRange[0]
        NewMax = newRange[1]
        NewSpan = NewMax - NewMin        
        
        for client, x in comboDict.items():
            scaled = round(((x - minX) * NewSpan / OldSpan) + NewMin)                        
            scaledDict[client] = scaled

        return scaledDict        
        
#%%
    def filterDict(self, inputDict, status):
        """
        
        Parameters
        ----------
        status : int or list of int, 
            one or subset of possible status [0, 1, 2, 3, 4], see setStatus

        Returns
        -------
        None.

        """
        if isinstance(status, Iterable):
            mask = list(status)
        else:
            mask = [status]
        
        selectedDict = {}
        for client, value in inputDict.items():
            s = self.statusDict.get(client)
            if s is None:
                raise ValueError('client {} does not exist in statusDict'.format(client))
            if s in mask:
                selectedDict[client] = value
        return selectedDict

#%%        
    def setMoneyDaily(self):
        if self.moneySumDict is None or self.loyaltyDict is None or self.T_Dict is None:
            raise ValueError('self.moneySumDict is None or self.loyaltyDict is None or self.T_Dict is None')
            
        self.moneyDailyDict = {}
        self.moneyDailyStepDict = {}
        for client, moneySum in self.moneySumDict.items():
            loyalty = self.loyaltyDict.get(client, 0)
            T = self.T_Dict.get(client, 0)
            loyalty = max(1, loyalty) # avoid division by 0
            T = max(1, T) # avoid division by 0
            self.moneyDailyStepDict[client] = moneySum / loyalty
            self.moneyDailyDict[client] = moneySum / T
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
        if self.periodsDict is None:
            raise RuntimeError('setStats has None')
        
        self.statsDict = {}
        # for client, sequence in tqdm(self.periodsDict.items()):
        for client, sequence in self.periodsDict.items():
            stats = Statistic2(sequence)
            self.statsDict[client] = stats
        return

#%%
    def setStatus(self):
        """
        Check customers activity, assigne dead or alive
        If dead, get death and observed death dates

        each customer can have only one status at state
        statusDict values :
            0 : active, proved by analysis of next purchase time
            1 : dead for a while
            2 : just died at this state
            3 : not born, died before zero state
            4 : censored, (also active) last purchase is close to the end of transactions period
            
        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.statsDict is None or self.lastEventDict is None or self.mapUserToPurchases is None:
            raise RuntimeError('self.statsDict is None or self.lastEventDict is None')
        
        self.statusDict = {}
        self.deathDict = {}
        self.deathStateDict = {}
        self.deathObservedDict = {}
        self.tRemainingObservedDict = {}
        self.tRemainingEstimateDict = {}
        
        for client, stats in self.statsDict.items():
            tRemainingObserved = round(stats.bestModel.ppf(P_DEATH)) # from last event
            tRemainingEstimate = round(stats.bestModel.ppf(P_LIFETIME)) # from last event                
            
            self.tRemainingObservedDict[client] = tRemainingObserved
            self.tRemainingEstimateDict[client] = tRemainingEstimate
            
            if lib2.isNaN(tRemainingObserved) or tRemainingObserved == np.inf:
                print(tRemainingObserved, stats.bestModel.params, stats.data)
                raise ValueError('tRemainingObserved')
                
                        # norm.ppf(0.9, 20, 0.1)


            dLastPurchase = self.lastEventDict[client] # from states point of view
            dPurchaseList = self.mapUserToPurchases.get(client) # all clients purchases GLOBAL
            
            dNextPurchase = State.roundToNext(self.dState, dPurchaseList)
            if dNextPurchase is None:
                inactiveSpan = (self.lastTransaction - dLastPurchase).days
            else:
                inactiveSpan = (dNextPurchase - dLastPurchase).days
                
            self.statusDict[client] = 0 # until proved to be dead
            if inactiveSpan >= tRemainingObserved:
                # observed death within transactions length; dead or will be soon
                inactiveSeen = (self.dState - dLastPurchase).days
                if inactiveSeen >= tRemainingEstimate:
                    # dead
                    self.statusDict[client] = 1 # dead, but when is to be determined
                    dDeath = dLastPurchase + timedelta(days=tRemainingEstimate) # date of death
                    self.deathDict[client] = dDeath
                    dDeathState = State.roundToNext(dDeath, self.datesList)
                    if dDeathState is None:
                        raise ValueError('Crap 1 in setStatus')                        
                    self.deathStateDict[client] = dDeathState                
                    self.deathObservedDict[client] = dLastPurchase + timedelta(days=tRemainingObserved) # date of death
                    # check non-born
                    if dDeath <= self.dZeroState:
                        # died before zero state                
                        self.statusDict[client] = 3
                        continue
                    if self.dState == self.dZeroState:
                        # if zero state - no previos state than
                        continue
                    # find when died, at this state or before?
                    # from point of view of previos state
                    prevStateInactiveSeen = (self.dState - timedelta(days=TS_STEP) - dLastPurchase).days
                    if prevStateInactiveSeen >= tRemainingEstimate:
                        # was dead already at previos step
                        continue
                    else:
                        # dies at this state
                        self.statusDict[client] = 2
                        # check
                        if dDeathState != self.dState:
                            # they must be equal
                            print()
                            print('client', client, 
                                  'dLastPurchase', dLastPurchase,
                                  'dDeath', dDeath, 
                                  'dDeathState', dDeathState, 
                                  'self.dState', self.dState, 
                                  'inactiveSpan', inactiveSpan,
                                  'inactiveSeen', inactiveSeen,
                                  'prevStateInactiveSeen', prevStateInactiveSeen, 
                                  'tRemainingEstimate', tRemainingEstimate, 
                                  'tRemainingObserved', tRemainingObserved)
                            raise ValueError('Crap 2 in setStatus')
            

# client 03681 
# dLastPurchase 1997-03-23
# dDeath 1997-04-13 
# dDeathState 1997-04-13 
# self.dState 1997-04-20 
# inactiveSpan 154
# inactiveSeen 28 
# prevStateInactiveSeen 21 
# tRemainingEstimate 21.834136187638276 
# tRemainingObserved 123.22872467098657
            
# date(1997, 3, 23) + timedelta(days=21.834136187638276)

            elif dLastPurchase + timedelta(days=tRemainingObserved) > self.lastTransaction:
                # censored; observed death lies beyond transactions
                self.statusDict[client] = 4
        return
    
#%%    
    def getC(self):    
        """Calculete clumpiness from transactions data. Source:
            Predicting Customer Value Using Clumpiness: From RFM to RFMC 
            
        """
        if self.T_Dict is None or self.periodsDict is None:
            raise RuntimeError('getClumpiness has None')
                    
        self.C_Dict = {}
        self.C_OrigDict = {}
        for client, durations in self.periodsDict.items():
            c = calcC(durations, span=None)
            span = self.T_Dict.get(client)
            cOrig = calcC(durations, span=span)
            self.C_Dict[client] = c
            self.C_OrigDict[client] = cOrig

        return        
    
#%%
    def getRatePoisson(self):
        self.ratePoissonDict = {}
        for client, periods in self.periodsDict.items():
            if len(periods) == 0:
                continue
            self.ratePoissonDict[client] = len(periods) / sum(periods)
        return

#%%
    def getProbabilityPoisson(self):
        """
        Probability of one event occuring during [0..revency] period

        Parameters
        ----------

        Returns
        -------
        dict: client -> Poisson probability of one event

        """
        if self.periodsDict is None or self.ratePoissonDict is None or self.recencyDict is None:
            raise RuntimeError('self.periodsDict is None or self.ratePoissonDict is None or self.recencyDict is None')
        self.pEventPoissonDict = {}
        for client, periods in self.periodsDict.items():
            if len(periods) == 0:
                continue
            r = self.ratePoissonDict.get(client)
            if r is None:
                continue
            recency = self.recencyDict.get(client)
            if recency is None:
                continue
            self.pEventPoissonDict[client] = 1 - exp(-r * recency)             
        return 

#%%
    def setTimeToPoisson(self):
        self.timeTo05PoissonDict = {}
        self.timeTo09PoissonDict = {}
        for client, recency in self.recencyDict.items():            
            self.timeTo05PoissonDict[client] = self.q05PoissonDict.get(client) - recency
            self.timeTo09PoissonDict[client] = self.q09PoissonDict.get(client) - recency
        return

#%%
    def setClients(self):
        if self.statusDict is None:
            raise RuntimeError('self.statusDict is None')
        
        clientsActive = []
        clientsOldDead = []
        clientsJustLost = []
        clientsNotBorn = []
        clientsCensored = []
        
        for client, status in self.statusDict.items():
            if status == 0:
                clientsActive.append(client)
            elif status == 1:
                clientsOldDead.append(client)
            elif status == 2:
                clientsJustLost.append(client)
            elif status == 3:
                clientsNotBorn.append(client)
            elif status == 4:
                clientsCensored.append(client)
            else:
                raise ValueError('Wrong status: {}'.format(status))
                
        self.clientsActive = sorted(clientsActive)
        self.clientsOldDead = sorted(clientsOldDead)
        self.clientsJustLost = sorted(clientsJustLost)
        self.clientsNotBorn = sorted(clientsNotBorn) 
        self.clientsCensored = sorted(clientsCensored) 
        self.clientsAll = sorted(list(self.statusDict.keys()))                
        return
    
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
        if self.thirdEventDict is None or self.datesList is None:
            raise RuntimeError('self.lastEventDict is None or self.datesList is None')                        
        self.birthStateDict = {}
        for client, dEvent in self.thirdEventDict.items():
            dEventAdj = State.roundToNext(dEvent, self.datesList)
            if dEventAdj is not None:
                self.birthStateDict[client] = dEventAdj
            else:
                raise RuntimeError('Crap in setBirth. Client: {}, 3rd date: {}'.format(client, dEvent))
        return
    
#%%    
    def setRecency(self):
        if self.lastEventDict is None or self.datesList is None:
            raise RuntimeError('self.lastEventDict is None or self.datesList is None')
        self.recencyDict = {}
        self.recencyStateDict = {}
        self.recencyTS_Dict = {} 
        self.lastEventStateDict = {}
        for client, dLast in self.lastEventDict.items():
            self.recencyDict[client] = (self.dState - dLast).days
            dLastState = State.roundToNext(dLast, self.datesList)
            if dLastState is not None:
                self.lastEventStateDict[client] = dLastState
                recencyState = (self.dState - dLastState).days
                self.recencyStateDict[client] = recencyState
                self.recencyTS_Dict[client] = round(recencyState / TS_STEP)
            else:
                raise RuntimeError('Crap in setRecency. Client: {}, last date: {}, dates: {}'.format(client, dLast, self.datesList))
        return
    
#%%    
    def setLoyalty(self):
        if self.birthStateDict is None or self.lastEventStateDict is None:
            raise RuntimeError('self.birthStateDict is None or self.lastEventStateDict is None')
        self.loyaltyDict = {}
        self.loyaltyStateDict = {}
        self.loyaltyTS_Dict = {}        
        for client, dBirth in self.birthStateDict.items():
            self.loyaltyDict[client] = (self.lastEventDict.get(client) - self.firstEventDict.get(client)).days
            lastEventState = self.lastEventStateDict.get(client)
            loyaltyState = (lastEventState - dBirth).days
            self.loyaltyStateDict[client] = loyaltyState
            self.loyaltyTS_Dict[client] = round(loyaltyState / TS_STEP)
        return
        
#%%
    def setT(self):
        if self.recencyDict is None or self.loyaltyDict is None or \
            self.recencyStateDict is None or self.loyaltyStateDict is None:
            raise RuntimeError('setT: something is None') 
            
        self.T_Dict = {}
        for client, recency in self.recencyDict.items():
            loyalty = self.loyaltyDict.get(client)
            self.T_Dict[client] = recency + loyalty
            
        self.T_StateDict = {}
        self.T_TS_Dict = {}
        for client, recencyState in self.recencyStateDict.items():
            loyaltyState = self.loyaltyStateDict.get(client)
            T_State = recencyState + loyaltyState
            self.T_StateDict[client] = T_State
            self.T_TS_Dict[client] = round(T_State / TS_STEP)            
        return        
        
#%%    
    def setD(self):
        if self.birthStateDict is None or self.deathDict is None or \
            self.deathStateDict is None or self.statusDict is None or \
            self.firstEventDict is None:
            raise RuntimeError('self.birthStateDict is None or self.deathDict is None or self.deathStateDict is None')
        self.D_Dict = {}
        self.D_StateDict = {}
        self.D_TS_Dict = {}
        # self.clientsNotBorn = []
        for client, status in self.statusDict.items():
            dFirstEvent = self.firstEventDict.get(client)
            dBirthState = self.birthStateDict.get(client)
            dDeath = self.deathDict.get(client)
            dDeathState = self.deathStateDict.get(client)
                          
            if status in [0, 4]:
                # active
                D = (self.dState - dFirstEvent).days
                D_State = (self.dState - dBirthState).days
            elif status in [1, 2]:
                # old dead or just died
                D = (dDeath - dFirstEvent).days
                D_State = (dDeathState - dBirthState).days
            elif status == 3:
                # not born
                D = (dDeath - dFirstEvent).days
                D_State = 0
            else:
                raise ValueError('Wrong status: {}'.format(status))
                
            self.D_Dict[client] = D
            self.D_StateDict[client] = D_State
            self.D_TS_Dict[client] = round(D_State / TS_STEP)            
                            
        return            

#%%        
    def getFeatures(self, client):
        """
        """
        featuresDict = {}
        featuresDict['E'] = self.lostDict.get(client)
        featuresDict['T'] = self.T_Dict.get(client)
        featuresDict['T_State'] = self.T_StateDict.get(client)
        featuresDict['T_TS'] = self.T_TS_Dict.get(client)        
        featuresDict['D'] = self.D_Dict.get(client)
        featuresDict['D_State'] = self.D_StateDict.get(client)
        featuresDict['D_TS'] = self.D_TS_Dict.get(client) 
        featuresDict['clumpiness'] = self.clumpinessDict.get(client)
        featuresDict['frequency'] = self.frequencyDict.get(client)
        featuresDict['recency'] = self.recencyDict.get(client)
        featuresDict['recencyState'] = self.recencyStateDict.get(client)
        featuresDict['recencyTS'] = self.recencyTS_Dict.get(client)        
        featuresDict['loyalty'] = self.loyaltyDict.get(client)
        featuresDict['loyaltyState'] = self.loyaltyStateDict.get(client)
        featuresDict['loyaltyTS'] = self.loyaltyTS_Dict.get(client)
        featuresDict['moneyDaily'] = self.moneyDailyDict.get(client)
        featuresDict['moneyMedian'] = self.moneyMedianDict.get(client)
        featuresDict['moneySum'] = self.moneySumDict.get(client)
        featuresDict['pEvent'] = self.pEventDict.get(client)        
        featuresDict['q05'] = self.q05Dict.get(client)
        featuresDict['q09'] = self.q09Dict.get(client)
        featuresDict['timeTo05'] = self.timeTo05Dict.get(client)
        featuresDict['timeTo09'] = self.timeTo09Dict.get(client)        
        featuresDict['r10_Clumpiness'] = self.r10_Clumpiness.get(client)
        featuresDict['r10_Frequency'] = self.r10_Frequency.get(client)
        featuresDict['r10_Recency'] = self.r10_Recency.get(client)
        featuresDict['r10_Loyalty'] = self.r10_Loyalty.get(client)
        featuresDict['r10_MoneyDaily'] = self.r10_MoneyDaily.get(client)
        featuresDict['r10_MoneyMedian'] = self.r10_MoneyMedian.get(client)
        featuresDict['r10_MoneySum'] = self.r10_MoneySum.get(client)
        featuresDict['r10_RF'] = self.r10_RF.get(client)
        featuresDict['r10_RFM'] = self.r10_RFM.get(client)
        featuresDict['r10_RFMC'] = self.r10_RFMC.get(client)        
        featuresDict['r10_FM'] = self.r10_FM.get(client)
        featuresDict['r10_MC'] = self.r10_MC.get(client)
        featuresDict['r10_FC'] = self.r10_FC.get(client)
        featuresDict['r10_FMC'] = self.r10_FMC.get(client)
        featuresDict['pEventPoisson'] = self.pEventPoissonDict.get(client)
        featuresDict['q05Poisson'] = self.q05PoissonDict.get(client)
        featuresDict['q09Poisson'] = self.q09PoissonDict.get(client)
        featuresDict['timeTo05Poisson'] = self.timeTo05PoissonDict.get(client)
        featuresDict['timeTo09Poisson'] = self.timeTo09PoissonDict.get(client)                        
        featuresDict['trendShort_FM'] = self.trendShort_FM.get(client, 0)
        featuresDict['trendShort_MC'] = self.trendShort_MC.get(client, 0)
        featuresDict['trendShort_FC'] = self.trendShort_FC.get(client, 0)
        featuresDict['trendShort_FMC'] = self.trendShort_FMC.get(client, 0)
        featuresDict['trend_FM'] = self.trend_FM.get(client, 0)
        featuresDict['trend_MC'] = self.trend_MC.get(client, 0)
        featuresDict['trend_FC'] = self.trend_FC.get(client, 0)
        featuresDict['trend_FMC'] = self.trend_FMC.get(client, 0)
        if not State.checkDictForNaN(featuresDict):
            raise ValueError('NaN in features, client: {}, date: {}'.format(client, self.dState))
        return featuresDict


# make additive and multiplicative composite
# trends 