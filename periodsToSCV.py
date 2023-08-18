


import os
# from pathlib import Path
import lib2
from tqdm import tqdm


if __name__ == "__main__":

    dataName = 'casa_2' # 'CDNOW' casa_2 plex_45 Retail

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    # resultsSubDir = os.path.join(resultsDir, dataName)    

    # if not Path(resultsSubDir).exists():
        # Path(resultsSubDir).mkdir(parents=True, exist_ok=True)

    statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    nStates = len(statesDict)
    transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    del(statesDict['transactionsDf'])
    dStateList = sorted(list(statesDict.keys()))
    dLast = dStateList[-1]
    lastState = statesDict[dLast]    
    del(statesDict)
    
    spendingSeries = transactionsDf.groupby('iId')['Sale'].apply(list)
    spendingDict = spendingSeries.to_dict()
    

    # export periods to R    
    rowsPeriod = []
    rowsSpending = []
    rowsRecency = []
    rowsLoyalty = []
    for client, clientObj in tqdm(lastState.Clients.items()):
        spending = spendingDict[client]
        periods = clientObj.descriptors['periods']
        recency = clientObj.descriptors['recency']
        loyalty = clientObj.descriptors['loyalty']
        
        l = [str(x) for x in periods]
        s = ','.join(l)
        s += '\n'
        rowsPeriod.append(s)
        
        l = [str(x) for x in spending]
        s = ','.join(l)
        s += '\n'    
        rowsSpending.append(s)
        
        s = '{}\n'.format(recency)
        rowsRecency.append(s)
        
        s = '{}\n'.format(loyalty)
        rowsLoyalty.append(s)        
    
    f = open(os.path.join(dataSubDir, 'interevent.csv'), 'w')
    f.writelines(rowsPeriod)
    f.close()
    
    f = open(os.path.join(dataSubDir, 'spending.csv'), 'w')
    f.writelines(rowsSpending)
    f.close()    
    
    f = open(os.path.join(dataSubDir, 'recency.csv'), 'w')
    f.writelines(rowsRecency)
    f.close()   
    
    f = open(os.path.join(dataSubDir, 'loyalty.csv'), 'w')
    f.writelines(rowsLoyalty)
    f.close() 
    
    # clientObj.show()
    
    
    
    
    
    
    
    
    