
import os
import sqlGeneral
import globalDef
from pathlib import Path
from tqdm import tqdm
import libRecommender2
import lib2
from Encoder import LabelEncoderRobust

if __name__ == "__main__":
    
    iCieId = 2
    dbNameMySQL = globalDef.iCieIdDict[iCieId]['dbNameMySQL']
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dbNameMySQL)
    if not Path(dataSubDir).exists():
        Path(dataSubDir).mkdir(parents=True, exist_ok=True)
    
    loginMsSQL = globalDef.loginMsSqlReal

    encoder = LabelEncoderRobust()
    
    transactionsDf = sqlGeneral.getSalesMailboxGrabb(loginMsSQL, iCieId, dStart=None,
        dEnd=None, version='longDet')

    if transactionsDf is not None:
        transactionsDf.sort_values('dTransactDate', inplace=True)    
        idList = encoder.fit_transform(transactionsDf['iId'])
    
        toCsvList = []
        for i, row in tqdm(transactionsDf.iterrows()):    
            iId = idList[i]
            sDate = '{}{:02d}{:02d}'.format(row['dTransactDate'].year, row['dTransactDate'].month, row['dTransactDate'].day)
            q = row['iQty']
            sale = row['fNetSales']
            s = '{}\t{}\t{}\t{}'.format(iId, sDate, q, sale)
            toCsvList.append(s)

    else:                
        transactionsDf = lib2.loadObject(os.path.join(dataSubDir, 'transactions.dat'))
        if transactionsDf is None:
            raise RuntimeError('Cannot load transactions')
        transactionsDf.sort_values('ds', inplace=True)

        idList = encoder.fit_transform(transactionsDf['iId'])
        
        toCsvList = []
        for i, row in tqdm(transactionsDf.iterrows()):    
            iId = idList[i]
            sDate = '{}{:02d}{:02d}'.format(row['ds'].year, row['ds'].month, row['ds'].day)
            q = 1
            sale = row['Sale']
            s = '{}\t{}\t{}\t{}'.format(iId, sDate, q, sale)
            toCsvList.append(s)
        
    libRecommender2.saveListToTextFile(toCsvList, os.path.join(dataSubDir, '{}.txt'.format(dbNameMySQL)))
    
    
    grabb_eng.casa