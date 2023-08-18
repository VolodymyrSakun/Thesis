
# merge 5 simulations in one transactions data

import os
from States import cdNowRawToDf
import libStent
import pandas as pd


if __name__ == "__main__":

    
    # kwargs = args
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    # dataSubDir = os.path.join(dataDir, sData)
    # dataSubDir = os.path.join(dataDir, 'Simulated_20', sData)    

    dataList = []
    for i in range(1, 6, 1):
        inputFilePath = os.path.join(dataDir, 'SimulatedShort00{}.txt'.format(i))
        rawData = libStent.loadTextFile(inputFilePath)
        dataDf = cdNowRawToDf(rawData)
        dataDf['iId'] = dataDf['iId'] + '_{}'.format(i)        
        dataList.append(dataDf)
        
    df = pd.concat(dataList, axis=0)
    df.sort_values('ds', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Qty'] = 1
    df['ds'] = df['ds'].astype(str)
    df['ds'] = df['ds'].str.replace('-', '')    
    df = df[['iId', 'ds', 'Qty', 'Sale']]    
    
    df.to_csv(os.path.join(dataDir, 'Simulated1.txt'), sep='\t', 
        na_rep='', float_format=None, columns=None, header=False, index=False, 
        index_label=None, mode='w', encoding=None, compression='infer', 
        quoting=None, quotechar='"', lineterminator=None, chunksize=None, 
        date_format=None, doublequote=True, escapechar=None, decimal='.', 
        errors='strict', storage_options=None)
    
    
    
    
              
