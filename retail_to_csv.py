

import os
import sqlGeneral
import globalDef
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook

# import libRecommender2
# import lib2
from Encoder import LabelEncoderRobust

def saveListToTextFile(corpus, filepath, encoding="utf-8"):
    if corpus is None:
        return
    # to disk        
    f = open(filepath, 'w', encoding=encoding)
    for line in corpus:
        f.write(line + "\n")
        # f.writelines(message)
        # f.writelines('\n')
    f.close()
    return  

if __name__ == "__main__":
    
    dataName = 'Retail' # 'CDNOW' casa_2 plex_45
    excelName = 'online_retail_II.xlsx'
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    
    data = pd.read_excel(os.path.join(dataSubDir, excelName), sheet_name=None)  
    
    data.keys()
    
    data1 = pd.concat([x for x in data.values()], axis=0)
    data1['dTransactDate'] = data1['InvoiceDate'].dt.date
    data1['Money'] =  data1['Quantity'] * data1['Price']
    
    data2 = data1[['Customer ID', 'dTransactDate', 'Quantity', 'Money']].copy(deep=True)
    data2.dropna(how='any', inplace=True)    
    data3 = data2[data2['Quantity'] > 0].copy(deep=True)
        
    g = data3.groupby(['dTransactDate', 'Customer ID']).agg({'Quantity': 'sum', 'Money': 'sum'}).reset_index()
    
    encoder = LabelEncoderRobust()
    idList = encoder.fit_transform(g['Customer ID'])

    toCsvList = []
    for i, row in tqdm(g.iterrows()):  
        iId = idList[i]
        sDate = '{}{:02d}{:02d}'.format(row['dTransactDate'].year, row['dTransactDate'].month, row['dTransactDate'].day)
        q = row['Quantity']
        sale = row['Money']
        s = '{}\t{}\t{}\t{}'.format(iId, sDate, q, sale)
        toCsvList.append(s)

    saveListToTextFile(toCsvList, os.path.join(dataSubDir, '{}.txt'.format(dataName)))



