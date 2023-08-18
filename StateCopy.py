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
# D_State: time interval between birth state and death state if dead or between birth and dState if alive
# D_TS: int(D_State) / TS_STEP
# C_Dict : clumpiness from first to last purchase (last restrictd by state)
# C_OrigDict : clumpiness from first purchase to state
# statsDict : client -> Statistics

# status : see setStatus
# stats : statistical models fitted on inter-purchase periods, see setStats

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
from Distributions import LogNormal
from Distributions import GeneralizedGamma
from Distributions import Gamma
from Distributions import Normal
from Distributions import Weibull
from Distributions import Exponential
from collections.abc import Iterable
# from tqdm import tqdm
# from datetime import date
from ClientTS import ClientTS as Client
from pathlib import Path

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
    return float(1 + np.sum(np.log(durationsScaled) * durationsScaled) / np.log(n))

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
def makeTsDates(transactionsDf, step = 7, dZeroState=None, holdOut=None):
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
        dState += timedelta(days=step)
    return sorted(datesList)

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
    """
    Use only Lognormal and Exponential
    Exponential is necessary if sequence looks like [10, 10, 10]
    Lognormal fits to zero variance which is crap; ppf returns nan for any number
    Exponential fits well for any sequence
    """
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

    
class State(object):
    """
    
    
    """
    
    def __init__(self, dDate, dZeroState, transactionsDf):
    
        # state variables           
        self.bTrends = False # when set trends, make it true
        self.longTrendLen = None
        self.shortTrendLen = None
        self.stateId = None # startes from 0; assigned by makeTS_Dates
        self.dState = dDate
        self.dZeroState = dZeroState
        self.firstTransaction = transactionsDf['ds'].min()
        self.lastTransaction = transactionsDf['ds'].max()        
        self.transactionsCut = None
        
        self.Clients = {} # clientId -> Client(object)
        
        # from all (non-cut) transactions; info will not be in Clients
        self.mapUserToPurchases = None # 
        
        # clients
        self.clientsAll = None
        self.clientsActive = None # active that have next purchase within short period
        self.clientsOldDead = None # died before
        self.clientsJustLost = None # set of clients that died at this state
        self.clientsNotBorn = None # nonBorn, died before zero state
        self.clientsCensored = None # active that have last purchase close to ent of transactions        
      
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
        
        firstEventDict = lib3.getMarginalEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='first')  
        self.putDescriptor('dFirstEvent', firstEventDict)
            
        lastEventDict = lib3.getMarginalEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START, 
            bIncludeEnd=INCLUDE_END, eventType='last')
        self.putDescriptor('dLastEvent', lastEventDict)

        thirdEventDict = lib3.getNthEvent(self.transactionsCut, 'iId', 'ds', 3)
        self.putDescriptor('dThirdEvent', thirdEventDict)
                
        periodsDict = lib3.getTimeBetweenEvents(self.transactionsCut, 'iId', 'ds', 
            leftMarginDate=None, rightMarginDate=self.dState, 
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY)
        self.putDescriptor('periods', periodsDict)

        frequencyDict = lib3.getFrequency(self.transactionsCut, 'iId', 'ds',\
            leftMarginDate=None, rightMarginDate=self.dState, bIncludeStart=INCLUDE_START,\
            bIncludeEnd=INCLUDE_END, minFrequency=MIN_FREQUENCY, exceptFirst=False)
        self.putDescriptor('frequency', frequencyDict)
                    
        
        self.setStats() # self.statsDict
        self.setRecency() # self.recencyDict, self.recencyStateDict, self.recencyTS_Dict                         
        self.setBirth() # self.birthStateDict
        self.setLoyalty() # loyaltyDict, loyaltyStateDict, loyaltyTS_Dict
        self.setT() # T_Dict, T_StateDict, T_TS_Dict
        self.getC() # C_Dict, C_OrigDict        
        self.setStatus() # active, dead, just died, not born
        self.setClients() # lists of clients according to status        
        self.setD() # D_Dict, D_StateDict, D_TS_Dict
        self.getRatePoisson() # Poisson features, see method for details
        # total spending            
        moneySumDict = lib3.getValue(self.transactionsCut, 'iId', 'ds', 'Sale',
            value='sum', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)
        self.putDescriptor('moneySum', moneySumDict)
        
        moneyMedianDict = lib3.getValue(self.transactionsCut, 'iId', 'ds', 'Sale',
            value='median', leftMarginDate=None, rightMarginDate=self.dState,
            bIncludeStart=INCLUDE_START, bIncludeEnd=INCLUDE_END, dropZeroValue=True, 
            dropFirstEvent=False)
        self.putDescriptor('moneyMedian', moneyMedianDict)

        self.setMoneyDaily() # daily spending         

        rank10 = Rank(method='Quantiles', nBins=10, dropZero=False, \
            returnZeroClass=True, includeMargin='right', zeroCenter=False)

#!!! Rank only alive customers, even if dead are just dormant
                    
        # Rank recency
        recencyActiveDict = self.getDescriptor('recency', status=[0, 4], error='raise')
        rank10.fit(recencyActiveDict)
        recencyDict = self.getDescriptor('recency', status=None, error='raise')                
        r10_R = rank10.getClusters(recencyDict, reverse=True)
        self.putDescriptor('r10_R', r10_R)
        
        # Rank frequency 
        frequencyActiveDict = self.getDescriptor('frequency', status=[0, 4], error='raise')
        rank10.fit(frequencyActiveDict)
        frequencyDict = self.getDescriptor('frequency', status=None, error='raise')                
        r10_F = rank10.getClusters(frequencyDict, reverse=False)
        self.putDescriptor('r10_F', r10_F)
        
        # Rank loyalty
        loyaltyActiveDict = self.getDescriptor('loyaltyState', status=[0, 4], error='raise')
        rank10.fit(loyaltyActiveDict)
        loyaltyDict = self.getDescriptor('loyaltyState', status=None, error='raise')                
        r10_L = rank10.getClusters(loyaltyDict, reverse=False)
        self.putDescriptor('r10_L', r10_L)
        
        # Rank climpiness; step, fit on active and dead
        cDict = self.getDescriptor('C', status=None, error='raise')
        rank10.fit(cDict)
        r10_C = rank10.getClusters(cDict, reverse=True)
        self.putDescriptor('r10_C', r10_C)        
        
        # Rank climpiness original, fit on active and dead
        cOrigDict = self.getDescriptor('C_Orig', status=None, error='raise')
        rank10.fit(cOrigDict)
        r10_C_Orig = rank10.getClusters(cOrigDict, reverse=True)
        self.putDescriptor('r10_C_Orig', r10_C_Orig)         
        
        # Rank daily monetary value
        moneyDailyActiveDict = self.getDescriptor('moneyDaily', status=[0, 4], error='raise')
        rank10.fit(moneyDailyActiveDict)
        moneyDailyDict = self.getDescriptor('moneyDaily', status=None, error='raise')                
        r10_M = rank10.getClusters(moneyDailyDict, reverse=False)
        self.putDescriptor('r10_M', r10_M)
        
        # M * C
        r10_MC = State.makeComplexRank([r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_MC', r10_MC)

        # F * M
        r10_FM = State.makeComplexRank([r10_F, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_FM', r10_FM)

        # F * C
        r10_FC = State.makeComplexRank([r10_F, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_FC', r10_FC)

        # R * F * C
        r10_FMC = State.makeComplexRank([r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_FMC', r10_FMC)
        
        # L * F * M
        r10_LFM = State.makeComplexRank([r10_L, r10_F, r10_M], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_LFM', r10_LFM)

        # L * F * M * C
        r10_LFMC = State.makeComplexRank([r10_L, r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_LFMC', r10_LFMC)
        
#   with recency
        # R * F
        r10_RF = State.makeComplexRank([r10_R, r10_F], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RF', r10_RF)

        # R * F * M
        r10_RFM = State.makeComplexRank([r10_R, r10_F, r10_M], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RFM', r10_RFM)
        
        # R * F * M * C        
        r10_RFMC = State.makeComplexRank([r10_R, r10_F, r10_M, r10_C_Orig], model='multiplicative', weights=None, nBins=10)
        self.putDescriptor('r10_RFMC', r10_RFMC)
 
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
    def getDescriptor(self, sDescriptor, status=None, error='ignore'):
        """
        make dict : client -> value
        get values from self.Clients that correspond to descriptor sDescriptor

        Parameters
        ----------
        sDescriptor : str
            feature name from Clients object.
        status : int, optional
            filter clients according to status. One value or list. See self.setStatus for details. The default is None: all clients. 
        error : str, optional
            if 'raise' : raises error if status or feature are not found in data for any client. The default is 'ignore'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        oneDescriptorDict : dict
            client -> value

        """
        if status is not None:
            if isinstance(status, Iterable):
                mask = list(status)
            else:
                mask = [status]
        
        oneDescriptorDict = {}
        for client, clientObj in self.Clients.items():
            value = clientObj.descriptors.get(sDescriptor)
            if error == 'raise' and value is None:
                raise ValueError('Customer {} has no descriptor {} at {}'.format(client, sDescriptor, self.dState))
            if status is not None:
                s = clientObj.descriptors.get('status')
                if error == 'raise' and s is None:
                    raise ValueError('Customer {} has no status at {}'.format(client, self.dState))
                if s in mask:
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
            clientObj.stats = Statistic2(periods)
            self.Clients[client] = clientObj
            
        return

#%%
    def setStatus(self):
        """
        Check customers activity, assigne dead or alive
        If dead, get death and observed death dates

        Set : 'status' , 'dDeath', 'dDeathState', 'dDeathObserved', 'tRemainingObserved', 'tRemainingEstimate'
           
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
            dEventsList = self.mapUserToPurchases.get(client) # all clients purchases GLOBAL
            if dEventsList is None:
                raise RuntimeError('State Error for customer {}. No purchases in self.mapUserToPurchases'.format(client))            
            tRemainingObserved = round(stats.bestModel.ppf(P_DEATH)) # from last event
            tRemainingEstimate = round(stats.bestModel.ppf(P_LIFETIME)) # from last event                            
            if lib2.isNaN(tRemainingObserved) or tRemainingObserved == np.inf:
                raise ValueError('Customer {} has tRemainingObserved = {}. Params: {}. Data: {}'.\
                    format(client, tRemainingObserved, stats.bestModel.params, stats.data))            

            dDeath = dLastEvent + timedelta(days=tRemainingEstimate) # date of death
            dDeathState = State.roundToNext(dDeath, self.datesList)
            dDeathObserved = dLastEvent + timedelta(days=tRemainingObserved) # date of death
            
            dNextEvent = State.roundToNext(self.dState, dEventsList)
            if dNextEvent is None:
                # no more purchases
                status = 4
                inactiveSpan = (self.lastTransaction - dLastEvent).days
            else:
                inactiveSpan = (dNextEvent - dLastEvent).days
                # there is next purchase
                status = 0
                                    
            if inactiveSpan >= tRemainingObserved:
                # observed death within transactions length; dead or will be soon
                inactiveSeen = (self.dState - dLastEvent).days
                if inactiveSeen >= tRemainingEstimate:
                    # dead
                    status = 1 # dead, but when is to be determined
                    if dDeathState is None:
                        raise ValueError('Crap 1 in setStatus')                        
                    # check non-born
                    if dDeath <= self.dZeroState:
                        # died before zero state                
                        status = 3
                    elif self.dState == self.dZeroState:
                        # if zero state - no previos state than
                        status = 2 # dies at zero state
                    # find when died, at this state or before?
                    # from point of view of previos state
                    else:
                        prevStateInactiveSeen = (self.dState - timedelta(days=TS_STEP) - dLastEvent).days
                        if prevStateInactiveSeen < tRemainingEstimate:
                            # dies at this state
                            status = 2
                            # check
                            if dDeathState != self.dState:
                                # they must be equal
                                print()
                                print('client', client, 
                                      'dLastPurchase', dLastEvent,
                                      'dDeath', dDeath, 
                                      'dDeathState', dDeathState, 
                                      'self.dState', self.dState, 
                                      'inactiveSpan', inactiveSpan,
                                      'inactiveSeen', inactiveSeen,
                                      'prevStateInactiveSeen', prevStateInactiveSeen, 
                                      'tRemainingEstimate', tRemainingEstimate, 
                                      'tRemainingObserved', tRemainingObserved)
                                raise ValueError('Crap 2 in setStatus')
                else:
                    pass
            # elif dLastEvent + timedelta(days=tRemainingObserved) > self.lastTransaction:
                # censored; observed death lies beyond transactions
                # status = 4

            clientObj.descriptors['status'] = status
            clientObj.descriptors['dDeath'] = dDeath
            clientObj.descriptors['dDeathState'] = dDeathState
            clientObj.descriptors['dDeathObserved'] = dDeathObserved
            clientObj.descriptors['tRemainingObserved'] = tRemainingObserved
            clientObj.descriptors['tRemainingEstimate'] = tRemainingEstimate            
                            
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
            clientObj.descriptors['tPoissonLifetime'] = log(1-P_LIFETIME) / (-ratePoisson)
            clientObj.descriptors['tPoissonDeath'] = log(1-P_DEATH) / (-ratePoisson)             
            self.Clients[client] = clientObj
             
        return

#%%
    def setClients(self):
        """
        set : 'clientsActive', 'clientsOldDead', 'clientsJustLost', 'clientsNotBorn', 'clientsCensored', 'clientsAll'

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
        clientsActive = []
        clientsOldDead = []
        clientsJustLost = []
        clientsNotBorn = []
        clientsCensored = []
        
        for client in self.Clients.keys():
            clientObj = self.Clients.get(client)
            if clientObj is None:
                raise RuntimeError('customer {} has no instance'.format(client))                  
            status = clientObj.descriptors.get('status')
            if status is None:
                raise RuntimeError('customer {} has no status'.format(client))  
            
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
        self.clientsAll = sorted(list(self.Clients.keys()))                
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
            dDeath = clientObj.descriptors.get('dDeath')
            if dDeath is None:
                raise RuntimeError('customer {} has no dDeath'.format(client))
            dDeathState = clientObj.descriptors.get('dDeathState')
            status = clientObj.descriptors.get('status')
            if status is None:
                raise RuntimeError('customer {} has no status'.format(client))
                          
            if status in [0, 4]:
                # active
                D = (self.dState - dFirstEvent).days
                D_State = (self.dState - dBirthState).days
            elif status in [1, 2]:
                # old dead or just died
                if dDeathState is None:
                    raise RuntimeError('customer {} has no dDeathState'.format(client))                
                D = (dDeath - dFirstEvent).days
                D_State = (dDeathState - dBirthState).days
            elif status == 3:
                # not born
                D = (dDeath - dFirstEvent).days
                D_State = 0
            else:
                raise ValueError('Wrong status: {}'.format(status))
                
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
        
        