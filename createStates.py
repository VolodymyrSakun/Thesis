# make states and save to disk

    # primary features
    # frequency
    # recency
    # loyalty
    # C
    # C_Orig
    # ratePoisson
    # pPoisson
    # tPoissonLifetime
    # tPoissonDeath
    # moneySum
    # moneyMedian
    # moneyDailyStep
    # moneyDaily
    
    # primary ranks
    # r10_R
    # r10_F
    # r10_L
    # r10_C
    # r10_C_Orig
    # r10_M # moneyDaily
    
    # composite ranks
    # r10_RF
    # r10_RFM
    # r10_RFMC # r10_C_Orig
    # r10_FM
    # r10_FC
    # r10_RFC
    
    # Long and short Trends    
    # r10_RF
    # r10_RFM
    # r10_RFMC # r10_C_Orig
    # r10_FM
    # r10_RFC
    
import os
import lib3
from datetime import date
# import State
from States import States
from States import loadCDNOW
from State import MIN_FREQUENCY
from State import TS_STEP


# features to be used to make trends
trendDescriptorsList = ['C', 'C_Orig', 'ratePoisson', 'pPoisson', 
    'r10_RF', 'r10_FM', 'r10_RFM', 'r10_RFML']
        
mergeTransactionsPeriod = 1

def main(inputFilePath, statesDir, **kwargs): 
    """
    kwargs = args
    statesDir = dataSubDir

    Parameters
    ----------
    inputFilePath : TYPE
        DESCRIPTION.
    statesDir : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    dStart = kwargs.get('dStart', None)
    holdOut = kwargs.get('holdOut', None)
    dEnd = kwargs.get('dEnd', None)
    dZeroState = kwargs.get('dZeroState', None)
    trendLong = kwargs.get('trendLong', 50)
    trendShort = kwargs.get('trendShort', 10)
    trends = kwargs.get('trends', trendDescriptorsList)
    forceAll = kwargs.get('forceAll', False)    
    
    if dZeroState is None and holdOut is None:
        raise ValueError('Either holdOut or dZeroState must not be None')    

    transactionsDf = loadCDNOW(inputFilePath, 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    # transactionsDf['ds'].hist(bins=100)
    # g = transactionsDf.groupby('iId').agg({'Sale': 'count'})

    transactionsDf = lib3.cutTransactions(transactionsDf, 'iId', 'ds', dStart=dStart, dEnd=dEnd,\
        bIncludeStart=True, bIncludeEnd=False, minFrequency=MIN_FREQUENCY, dMaxAbsence=None)     
    # transactionsDf['ds'].hist()
    # g = transactionsDf.groupby('iId').agg({'Sale': 'count'})
    
    states = States(statesDir, transactionsDf, tsStep=TS_STEP, dZeroState=dZeroState, holdOut=holdOut)
    states.save()
    
    print('Start creating states')
    states.update(transactionsDf, forceAll=forceAll)
    print('Start updating trends')
    states.updateTrends(trends, short=trendShort, long=trendLong, forceAll=forceAll)
    states.save()

    return

###############################################################################

# CDNOW casa_2 plex_45 Retail Simulated1 Simulated5 SimulatedShort001
if __name__ == "__main__":
    
    sData = 'Retail'
    
    if sData.find('SimulatedShort') != -1: # SimulatedShort001
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2000, 7, 1), 'dEnd': date(2004, 7, 1)}
    elif sData == 'casa_2':
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2016, 1, 1), 'dEnd': date(2019, 1, 1)}
    elif sData == 'CDNOW':
        args = {'dStart': None, 'holdOut': 90,
            'dZeroState': None, 'dEnd': None}
    elif sData == 'Retail':
        args = {'dStart': None, 'holdOut': 90,
            'dZeroState': None, 'dEnd': None}
    elif sData == 'Simulated1':
        args = {'dStart': None, 'holdOut': None,
            'dZeroState': date(2000, 7, 1), 'dEnd': date(2003, 7, 1)}   
        
    # kwargs = args
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, sData)
    # dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)    
    inputFilePath = os.path.join(dataSubDir, '{}.txt'.format(sData))
    
    main(inputFilePath, dataSubDir, **args)






