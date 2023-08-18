# create states for N simulated transaction datasets

import os
from createStates import main as createStatesMain
import lib2
import lib3
import shutil
from datetime import date

if __name__ == "__main__":    
    
    args = {'dStart': None, 'holdOut': None,
    'dZeroState': date(2000, 7, 1), 'dEnd': date(2003, 7, 1)}
            
    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    inputDir = os.path.join(dataDir, 'Simulated20')
    
    files = sorted(lib3.findFiles(inputDir, '*.txt', exact=False))

    for filePath in files: 
        fileName = os.path.split(filePath)[-1]
        print(fileName)
        if fileName < 'SimulatedShort017.txt':
            continue
        sData = fileName.split('.')[0]
        localDir = os.path.join(inputDir, sData)
        lib2.checkDir(localDir)
        outFilePath = os.path.join(localDir, fileName)
        shutil.copyfile(filePath, outFilePath)
        createStatesMain(outFilePath, localDir, **args)
        
        
        
        
        


    
    



