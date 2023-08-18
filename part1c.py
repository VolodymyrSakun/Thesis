
# Cox, AFT, univariate
# collect results from 20 simulations 
# make box plot and summary tables

import os
import lib2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def seriesToList(s):    
    nestedList = []
    for _, value in s.iteritems():
        newValue = [x for x in value if not lib2.isNaN(x)]
        nestedList.append(newValue)
    return nestedList
        
def boxPlotAssumption(df, resultsDir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19,15))
    
    column = 'nViolating' # 'nViolating', 'fracViolating', 'population']
    _ = ax1.boxplot(df[column].squeeze(), vert = 1)
    ax1.set_xticklabels(['Number of customers violating churn assumption'], rotation=0)            
    ax1.set_ylabel('Value')
    ax1.title.set_text('Box plot number of violations')

    column = 'fracViolating' # 'nViolating', 'fracViolating', 'population']
    _ = ax2.boxplot(df[column].squeeze(), vert = 1)
    ax2.set_xticklabels(['Fraction of customers violating churn assumption'], rotation=0)
    ax2.set_ylabel('Value')
    ax2.title.set_text('Box plot fraction of violations')
    
    column = 'population' # 'nViolating', 'fracViolating', 'population']
    _ = ax3.boxplot(df[column].squeeze(), vert = 1)
    ax3.set_xticklabels(['Size of customers population'], rotation=0)
    ax3.set_ylabel('Value')
    ax3.title.set_text('Box plot population size')    

    title = 'Box plot violation of churn assumption'
    fig.suptitle(title, fontsize=16)

    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsDir, fileName), bbox_inches='tight')
    plt.close()         
    return
    
def boxPlotError(s, resultsDir, **kwargs):

    yLabel = kwargs.get('yLabel', '')
    title = kwargs.get('title', 'Box plot')        
    title1 = kwargs.get('title1', '')
    title2 = kwargs.get('title2', '')
    modelsToPlot = kwargs.get('modelsToPlot', 'both')
    addTrue = kwargs.get('addTrue', False)
    
    if addTrue:
        modelsUnivariate_ = modelsUnivariate + ['True']
    else:
        modelsUnivariate_ = modelsUnivariate
        
    if isinstance(s, list):
        sPlot = pd.concat(s, axis=0)
        s1 = s[0]
        s2 = s[1]
    elif modelsToPlot == 'univariate':
        sPlot = s.loc[modelsUnivariate_]
    elif modelsToPlot == 'regression':
        sPlot = s.loc[moelsRegression]
    else:
        sPlot = s
        s1 = sPlot.loc[modelsUnivariate_]
        s2 = sPlot.loc[moelsRegression]
    
    yMax = 0
    for _, data1List in sPlot.iteritems():
        yMax = max(yMax, max(data1List))
    
    yMin = np.inf
    for _, data1List in sPlot.iteritems():
        yMin = min(yMin, min(data1List))  

    gap = (yMax - yMin) * 0.05
    yMax += gap
    yMin -= gap
    
    if modelsToPlot == 'both':        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,15))
    
        _ = ax1.boxplot(seriesToList(s1), vert = 1)            
        ax1.set_xticklabels(s1.index, rotation=45)            
        ax1.set_ylabel(yLabel)
        ax1.title.set_text(title1)

        _ = ax2.boxplot(seriesToList(s2), vert = 1)
        ax2.set_xticklabels(s2.index, rotation=45)
        ax2.set_ylabel(yLabel)
        ax2.title.set_text(title2)
    
        ax1.set_ylim([yMin, yMax])
        ax2.set_ylim([yMin, yMax])
    
    else:
        fig, ax = plt.subplots(1, 1, figsize=(19,15))
        _ = ax.boxplot(seriesToList(sPlot), vert = 1)
        ax.set_xticklabels(sPlot.index, rotation=45)        
        ax.set_ylabel(yLabel)
    
    fig.suptitle(title, fontsize=16)
    
    fileName = '{}.png'.format(title)
    plt.savefig(os.path.join(resultsDir, fileName), bbox_inches='tight')
    plt.close() 
    
    return
    
if __name__ == "__main__":

    modelsUnivariate = ['Exponential', 'Weibull', 'LogNormal', 'LogLogistic', 'GeneralizedGamma']
    moelsRegression =['CoxPHFitter', 'CoxPHSurvivalAnalysis','GradientBoostingSurvivalAnalysis',
        'LogNormalAFT', 'LogLogisticAFT', 'WeibullAFT']
    
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'o', 'k']

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data', 'Simulated20')
    resultsDir = os.path.join(workDir, 'results', 'Simulated20')
    # 'SimulatedShort001' to 'SimulatedShort020'
    dirsList = [x for x in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, x))]
    
    def readAssumption(path):
        f = open(path, 'r')
        text = f.read()
        f.close()
        l = text.split()
        return int(l[0]), float(l[3]), int(l[8])
    
    
    scoresList = []
    scoresCensoredList = []
    profitList = []
    assumptionsList = []
    for sData in dirsList:   
        
        assumption = readAssumption(os.path.join(resultsDir, sData, 'assumption.txt'))
        assumptionsList.append(assumption)
        
        fitResults = lib2.loadObject(os.path.join(resultsDir, sData, 'fitResultsDict.dat'))
    
        scoresDf = pd.read_excel(os.path.join(resultsDir, sData, 'Scores.xlsx'))
        scoresCensoredDf = pd.read_excel(os.path.join(resultsDir, sData, 'Scores Censored.xlsx'))
        profitDf = pd.read_excel(os.path.join(resultsDir, sData, 'Profit.xlsx'))
    
        scoresDf['simulation'] = sData
        scoresCensoredDf['simulation'] = sData
        profitDf['simulation'] = sData
    
        scoresList.append(scoresDf)
        scoresCensoredList.append(scoresCensoredDf)
        profitList.append(profitDf)
    
        print(sData)
    
    aDf = pd.DataFrame(assumptionsList, columns=['nViolating', 'fracViolating', 'population'])
    boxPlotAssumption(aDf, resultsDir)

    scoresDf = pd.concat(scoresList, axis=0)        
    scoresCensoredDf = pd.concat(scoresCensoredList, axis=0)
    profitDf = pd.concat(profitList, axis=0)
    
    columns = list(scoresDf.columns)
    columns[0] = 'Model'
    scoresDf.columns = columns
    
    columns = list(scoresCensoredDf.columns)
    columns[0] = 'Model'
    scoresCensoredDf.columns = columns
    
    columns = list(profitDf.columns)
    columns[0] = 'Model'
    profitDf.columns = columns
    
    # train / test split, training interval only
    scoresAvgDf = scoresDf.groupby(['Model']).agg({'CI': 'mean', 'CI IPCW': 'mean', 'IBS': 'mean',
        'AUC': 'mean', 'IAE Median': 'mean', 'ISE Median': 'mean', 'tExpected': 'mean',
        'sizeExpected': 'mean', 'tMedian': 'mean', 'sizeMedian': 'mean'})

    # univariate models    
    scoresAvgUniDf = scoresAvgDf.loc[['Exponential', 'Weibull', 'LogNormal','LogLogistic', 'GeneralizedGamma']].copy(deep=True)    
    scoresAvgUniDf.drop(columns=['CI', 'CI IPCW', 'AUC'], inplace=True)
    scoresAvgUniDf.to_excel(os.path.join(resultsDir, 'scoresAvgUni.xlsx'))
    
    # regression models
    scoresAvgRegressDf = scoresAvgDf[~scoresAvgDf.index.isin(scoresAvgUniDf.index)].copy(deep=True)
    scoresAvgRegressDf.sort_values('CI IPCW', ascending=False, inplace=True)
    scoresAvgRegressDf = scoresAvgRegressDf.T
    scoresAvgRegressDf.to_excel(os.path.join(resultsDir, 'scoresAvgRegress.xlsx'))
    
    # Metrics are calculated on test clients that were censored (‘alive’) at 
    # the end of training interval. ‘True’ survival time comes from simulations.
    scoresCensoredAvgDf = scoresCensoredDf.groupby(['Model']).agg({'MAE Expected': 'mean',
        'sizeExpected': 'mean', 'MedianAE': 'mean', 'sizeMedian': 'mean'})
    scoresCensoredAvgDf.sort_values('MedianAE', ascending=True, inplace=True)
    scoresCensoredAvgDf.to_excel(os.path.join(resultsDir, 'scoresCensoredAvg.xlsx'))

    profitAvgDf = profitDf.groupby(['Model', 'Remaining', 'Descriptor']).agg({ \
        'Predicted': 'mean', 'True': 'mean', 'AbsError': 'mean'}).reset_index()
    

    profitAvgSmoothDf = profitAvgDf[profitAvgDf['Descriptor'] == 'Smooth'].copy(deep=True)    
    profitAvgSmoothExpectedDf = profitAvgSmoothDf[profitAvgSmoothDf['Remaining'] == 'Expected'].copy(deep=True)
    profitAvgSmoothMedianDf = profitAvgSmoothDf[profitAvgSmoothDf['Remaining'] == 'Median'].copy(deep=True)
    profitAvgSmoothExpectedDf.sort_values('AbsError', ascending=True, inplace=True)
    profitAvgSmoothMedianDf.sort_values('AbsError', ascending=True, inplace=True)
    profitAvgSmoothExpectedDf.drop(columns=['Remaining', 'Descriptor'], inplace=True)
    profitAvgSmoothMedianDf.drop(columns=['Remaining', 'Descriptor'], inplace=True)
    profitAvgSmoothExpectedDf.to_excel(os.path.join(resultsDir, 'profitAvgSmoothExpected.xlsx'), index=False)
    profitAvgSmoothMedianDf.to_excel(os.path.join(resultsDir, 'profitAvgSmoothMedian.xlsx'), index=False)

    
    
    
    scoresBoxDf = scoresDf.groupby(['Model']).agg({'CI IPCW': list, 'AUC': list, 'tExpected': list, 'tMedian': list})
    
    scoresCensoredBoxDf = scoresCensoredDf.groupby(['Model']).agg({'MAE Expected': list, 'MedianAE': list})
    
    
    profitSmoothDf = profitDf[profitDf['Descriptor'] == 'Smooth'].copy(deep=True)    
    profitSmoothExpectedDf = profitSmoothDf[profitSmoothDf['Remaining'] == 'Expected'].copy(deep=True)    
    profitSmoothMedianDf = profitSmoothDf[profitSmoothDf['Remaining'] == 'Median'].copy(deep=True)
    
    
    profitSmoothExpectedBoxDf = profitSmoothExpectedDf.groupby(['Model']).agg({'AbsError': list, 'Predicted': list})
    profitSmoothMedianBoxDf = profitSmoothMedianDf.groupby(['Model']).agg({'AbsError': list, 'Predicted': list})

    profitTrue = list(profitSmoothExpectedDf['True'].unique())
    
    # profitSmoothExpectedBoxSeries = profitSmoothExpectedDf.groupby(['Model']).agg({'AbsError': list}).squeeze()
    # profitSmoothMedianBoxSeries = profitSmoothMedianDf.groupby(['Model']).agg({'AbsError': list}).squeeze()
    
    
    # s = profitSmoothExpectedBoxDf
    # s = profitSmoothMedianBoxDf
    # s = scoresCensoredBoxDf['MAE Expected']
    # s = scoresCensoredBoxDf['MedianAE']
    # s = scoresBoxDf['CI IPCW']
    # s = scoresBoxDf['tExpected']
    # s = scoresBoxDf['tMedian']
    
    

    
    # Profit plots
    # Error expected
    args1 = {'title': 'Abs error predicted profit expected remaining life', 
             'yLabel': 'Abs error predicted profit',
             'title1': 'Univariate models',
             'title2': 'Regression models',
             'modelsToPlot': 'both'}
    boxPlotError(profitSmoothExpectedBoxDf['AbsError'], resultsDir, **args1)
    # Error medien
    args2 = {'title': 'Abs error predicted profit median remaining life', 
             'yLabel': 'Abs error predicted profit',
             'title1': 'Univariate models',
             'title2': 'Regression models',
             'modelsToPlot': 'both'}
    boxPlotError(profitSmoothMedianBoxDf['AbsError'], resultsDir, **args2)
    # Predicted expected profit 
    profitSmoothExpectedBoxSeries = profitSmoothExpectedBoxDf['Predicted']
    profitSmoothExpectedBoxSeries = pd.concat([profitSmoothExpectedBoxSeries, 
        pd.Series(data={'True': profitTrue}, name='Predicted')], axis=0)    
    args3 = {'title': 'Profit predicted with expected remaining life', 
         'yLabel': 'Predicted profit',
         'title1': 'Univariate models + True profit',
         'title2': 'Regression models',
         'modelsToPlot': 'both',
         'addTrue': True}
    boxPlotError(profitSmoothExpectedBoxSeries, resultsDir, **args3)
    # Predicted median profit
    profitSmoothMedianBoxSeries = profitSmoothMedianBoxDf['Predicted']
    profitSmoothMedianBoxSeries = pd.concat([profitSmoothMedianBoxSeries, 
        pd.Series(data={'True': profitTrue}, name='Predicted')], axis=0)    
    args4 = {'title': 'Profit predicted with median remaining life', 
         'yLabel': 'Predicted profit',
         'title1': 'Univariate models + True profit',
         'title2': 'Regression models',
         'modelsToPlot': 'both',
         'addTrue': True}
    boxPlotError(profitSmoothMedianBoxSeries, resultsDir, **args4)
    # MAE expected lifetime
    args5 = {'title': 'Mean absolute error of expected remaining life', 
         'yLabel': 'MAE',
         'title1': 'Univariate models',
         'title2': 'Regression models'}
    boxPlotError(scoresCensoredBoxDf['MAE Expected'], resultsDir, **args5)
    # MAE median lifetime
    args6 = {'title': 'Mean absolute error of median remaining life', 
         'yLabel': 'MAE',
         'title1': 'Univariate models',
         'title2': 'Regression models'}
    boxPlotError(scoresCensoredBoxDf['MedianAE'], resultsDir, **args6)    
    # CI IPCW regression
    args7 = {'title': 'CI IPCW and AUC of regression models', 
         'yLabel': '[0..1]',
         'title1': 'CI IPCW',
         'title2': 'AUC'}    
    boxPlotError([scoresBoxDf['CI IPCW'].loc[moelsRegression], scoresBoxDf['AUC'].loc[moelsRegression]], resultsDir, **args7)
    # Expected lifetime
    args8 = {'title': 'Expected remaining life', 
         'yLabel': 'Expected remaining life',
         'title1': 'Univariate models',
         'title2': 'Regression models'}
    boxPlotError(scoresBoxDf['tExpected'], resultsDir, **args8)
    # Median lifetime
    args9 = {'title': 'Median remaining life', 
         'yLabel': 'Median remaining life',
         'title1': 'Univariate models',
         'title2': 'Regression models'}
    boxPlotError(scoresBoxDf['tMedian'], resultsDir, **args9)


    a = profitDf[((profitDf['Model'] == 'CoxPHSurvivalAnalysis') & (profitDf['Remaining'] == 'Median') & (profitDf['Descriptor'] == 'Smooth'))]
    
    a.mean()