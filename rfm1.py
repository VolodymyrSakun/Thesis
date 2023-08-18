# plot rfm features


import os
import globalDef
import gen_cfg as cfg
import lib3
import lib2
import libChurn2
from random import sample
import libPlot
from datetime import timedelta
import sqlGeneral

def loadTransactions(fileName, login, iCieId):
    dataDf = lib2.loadObject(fileName)
    if dataDf is None:
        dataDf = sqlGeneral.getTransactions(iCieId, loginMsSQL, dStart=None,\
            dEnd=None, bIncludeStart=True, bIncludeEnd=False, minFrequency=1) 
        
        dataDf = lib3.groupTransactions2(dataDf, 'iId', 'ds', 'Sale',
            freq=globalDef.iCieIdDict[iCieId]['mergeTransactionsPeriod'])
                    
        dataDf.sort_values(['iId', 'ds'], inplace=True)
        dataDf.reset_index(drop=True, inplace=True)        
        
        lib2.saveObject(fileName, dataDf)

    return dataDf

if __name__ == "__main__":
    
    iCieId = 46
    iMaxInactine = 365
    nTest = 20
    workDir = os.path.dirname(os.path.realpath('__file__'))
    if 'Thesis' not in workDir:
        raise RuntimeError('Change working dir')
        
    dbNameMySQL = globalDef.iCieIdDict[iCieId]['dbNameMySQL']
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dbNameMySQL)
    plotDir = os.path.join(dataSubDir, 'plot')
    resultsDir = os.path.join(dataSubDir, 'results')
    loginMsSQL = globalDef.loginMsSqlReal
    loginMySQL = globalDef.loginMySqlLocal
    
    sTransactions = 'transactions.dat'
    
    # first run; directory service_id_xx does not exist
    if os.path.exists(dataSubDir):
        if not os.path.exists(plotDir):            
            os.mkdir(plotDir)
        if not os.path.exists(resultsDir):            
            os.mkdir(resultsDir)            
        
    files = lib3.findFiles(dataSubDir, libChurn2.sTsDict)
    if len(files) > 0:
        tsDict = lib2.loadObject(files[-1])
        churnRateDict = tsDict['fChurnRateYearly'].copy()
        del(tsDict['fChurnRateYearly'])
        observationPeriodDict = tsDict['iObservationPeriod'].copy()
        del(tsDict['iObservationPeriod'])    
    else:
        raise RuntimeError()
        
    dataDf = loadTransactions(os.path.join(dataSubDir, sTransactions), loginMsSQL, iCieId)
    
    deadDf = sqlGeneral.readSQLTable(loginMySQL, dbNameMySQL,\
        libChurn2.sTimeSeriesTable2Lifetime, schema=None)
    deadDf.set_index('sCustomerNumber', inplace=True)
        
    features = sorted(list(tsDict.keys()))
    features = [str(x) for x in features]
    dates = list(tsDict[features[0]].columns)
    users = sorted(list(tsDict[features[0]].index))
    testUsers = sample(users, nTest)

    dLast = max(dates)
    dRightMargin = dLast - timedelta(days=iMaxInactine)
    
    lostDict = lib3.getLost(dataDf, columnId='iId', columnEvent='ds', 
        checkPointDate=dRightMargin, leftMarginDate=None, 
        bIncludeStart=True, bIncludeEnd=False)

        
    ['iRecency', 'iFrequency', 'iLoyalty', 'iPeriod05', 'iPeriod09', 'iT', 
     'fClumpiness', 'fMoneySum', 'fMoneyMean', 'fMoneyMedian', 'fMoneyDaily', 
     'fCLV_Mean', 'fCLV_BGF', 'fCLV_Univariate', 'fCLV_AFT', 'pAliveBGF', 
     'pAliveUnivariate', 'pAliveAFT', 'iDaysLeftBGF', 'iDaysLeftUnivariate', 
     'iDaysLeftAFT', 'pEvent', 'iDaysTo05', 'iDaysTo09', 'rParettoRF', 
     'rParettoRFM', 'rParettoRFC', 'rParettoRFMC', 'r10Recency', 'r10Frequency',
     'r10Loyalty', 'r10Period05', 'r10Period09', 'r10T', 'r10C', 'rParettoMoneySum',
     'rParettoMoneyMean', 'rParettoMoneyMedian', 'rParettoMoneyDaily']
    
    for user in testUsers:
    # user = testUsers[0]
    # user = '92790'
        bLost = lostDict.get(user)

        libPlot.plotBehaviour(user, data=tsDict, key='rParettoRFMC', columnDate=None,
            plotDirName=plotDir, dEventsList=None,\
            namePostfix='', window=None, prominence=None, lost=bLost)  
    
        libPlot.plotTsFeatures2(user, tsDict, config='CLV', smooth=False, transactionsDf=dataDf, 
            deadDf=deadDf, dLeftMargin=None, visitsConfDict=None, plotDirName=plotDir, 
            fileName='CLV', lost=bLost)

        libPlot.plotTsFeatures2(user, tsDict, config='Probability', smooth=False, 
            transactionsDf=dataDf, deadDf=deadDf, dLeftMargin=None, visitsConfDict=None, 
            plotDirName=plotDir, fileName='Probability', lost=bLost)    
 
        libPlot.plotTsFeatures2(user, tsDict, config='RankPrimary2', smooth=False, 
            transactionsDf=dataDf, deadDf=deadDf, dLeftMargin=None, visitsConfDict=None, 
            plotDirName=plotDir, fileName='RankPrimary2', lost=bLost)  

        libPlot.plotTsFeatures2(user, tsDict, config='RankComplex', smooth=False, 
            transactionsDf=dataDf, deadDf=deadDf, dLeftMargin=None, visitsConfDict=None, 
            plotDirName=plotDir, fileName='RankComplex', lost=bLost)  

        libPlot.plotTsFeatures2(user, tsDict, config='RankMoney', smooth=False, 
            transactionsDf=dataDf, deadDf=deadDf, dLeftMargin=None, visitsConfDict=None, 
            plotDirName=plotDir, fileName='RankMoney', lost=bLost)          


    # user = '56993419'
    # user = '5570738'
    # user in deadDf.index
    
    
    