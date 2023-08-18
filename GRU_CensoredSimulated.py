
# create states for N simuylated transactions

import os
from GRU_Censored import main as GRU_CensoredMain
# import lib2
import lib3
# import shutil
# from datetime import date

if __name__ == "__main__":
    
    args = {'nTimeSteps': 50, 
            'validFrac': 0.15,
            'batch_size': 256, 
            'epochs': 100,
            'patience': 5,
            'maxNumTrain': 200000,
            'frac': 1} # for large data use fraction of training data to avoid shortage memory problem
 
    
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    inputDir = os.path.join(dataDir, 'Simulated_20')
    
    files = sorted(lib3.findFiles(inputDir, '*.txt', exact=False))

    for filePath in files:    
        fileName = os.path.split(filePath)[-1]
        sData = fileName.split('.')[0]
        print(sData)          
        if sData < 'SimulatedShort012':
            continue
        localDir = os.path.join(inputDir, sData)
        GRU_CensoredMain(localDir, **args)

        
        