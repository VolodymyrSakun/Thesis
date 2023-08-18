
# defining churn
# plots
# 'Density histogram of inter-purchase intervals'
# 'Density histogram of number of purchases'

import os
import lib3
import State
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from createStates import loadCDNOW
from Distributions import Statistics
from Distributions import Normal
from Distributions import LogNormal
from Distributions import Fisk
from Distributions import Exponential
from Distributions import Gamma
from Distributions import Weibull
from tqdm import tqdm


   
class Statistics6(Statistics):
    """
    Select 6 distributions from Statistics
    """
    modelsLocDict = {'Normal': {'model': Normal(), 'args': {}},
                      'Lognormal': {'model': LogNormal(), 'args': {'floc': 0}},
                      'Fisk': {'model': Fisk(), 'args': {'floc': 0}},
                      'Exponential': {'model': Exponential(), 'args': {'floc': 0}},
                      'Gamma': {'model': Gamma(), 'args': {'floc': 0}},
                      'Weibull': {'model': Weibull(), 'args': {'floc': 0}}}
        
    modelsFreeDict = {'Normal': {'model': Normal(), 'args': {}},
                      'Lognormal': {'model': LogNormal(), 'args': {}},
                      'Fisk': {'model': Fisk(), 'args': {}},
                      'Exponential': {'model': Exponential(), 'args': {}},
                      'Gamma': {'model': Gamma(), 'args': {}},
                      'Weibull': {'model': Weibull(), 'args': {}}}    

    def __init__(self, data, models='all', testStats='ks', criterion='pvalue', locZero=True):
        super().__init__(data, models=models, testStats=testStats, criterion=criterion, locZero=locZero)
        return



if __name__ == "__main__":

    MIN_FREQUENCY = State.MIN_FREQUENCY
    mergeTransactionsPeriod=State.TS_STEP
    mergeTransactionsPeriod = 1
    pDeath = 0.95
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y', 'o']

    dataName = 'Simulated1' # 'CDNOW' casa_2 plex_45 Retail Simulated1

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    resultsSubDir = os.path.join(resultsDir, dataName) 
    
    transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(dataName)), 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    
    print('Start:', transactionsDf['ds'].min())
    print('End:', transactionsDf['ds'].max())
    
    # transactionsDf['ds'].max() - transactionsDf['ds'].min()
    
    periodsDict = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
        leftMarginDate=None, rightMarginDate=None, 
        bIncludeStart=True, bIncludeEnd=True, minFrequency=MIN_FREQUENCY)
    
    frequencyDict = lib3.getFrequency(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, bIncludeStart=True, bIncludeEnd=True,\
        minFrequency=3, exceptFirst=False)
    
    # get sequences
    seqList = []
    iptList = [] # inter-purchase time intervals, all clients
    for key, value in periodsDict.items():
        iptList.extend(value)
        n = np.array(value)
        seqList.append(n)
        
    frequencyList = []
    for value in frequencyDict.values():
        frequencyList.append(value)

    
#   'Density histogram of inter-purchase intervals'    
    fig1, ax1 = plt.subplots(1, 1, figsize=(19,15))  
    plt.hist(iptList, bins=200, color='blue', edgecolor='black', density=True)  # used to be normed=True in older versions    
    title = 'Density histogram of inter-purchase intervals'
    plt.title(title)
    plt.xlabel('IPI')
    plt.ylabel('Relative frequency')
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
#   'Density histogram of number of purchases'    
    fig2, ax2 = plt.subplots(1, 1, figsize=(19,15))      
    plt.hist(frequencyList, bins=100, color='blue', edgecolor='black', density=True)  # used to be normed=True in older versions    
    title = 'Density histogram of number of purchases'
    plt.title(title)
    plt.xlabel('Number of purchases')
    plt.ylabel('Relative frequency')
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    # y = np.array([14, 21, 63, 49, 21, 35, 56, 84, 21, 50])
    # statsCVM = Statistics6(y, testStats='cvm', criterion='pvalue')
    # statsKS = Statistics6(y, testStats='ks', criterion='pvalue')

#   fit distributions to clients
    ttcDict = {}
    ttdDict = {}
    pValuesDict = {'KS': {}, 'CvM': {}}
    # clientsList = []
    for client, y in tqdm(periodsDict.items()):
        # clientsList.append(client)
        stats = Statistics6(y, testStats='cvm', criterion='pvalue')
        for sDistribution, modelDict in stats.modelsDict.items():
            if sDistribution not in ttcDict.keys():
                ttcDict[sDistribution] = {}
            if sDistribution not in ttdDict.keys():
                ttdDict[sDistribution] = {}
            model = modelDict['model']
            ttc = model.median()
            ttcDict[sDistribution][client] = ttc
            ttd = model.ppf(pDeath)
            ttdDict[sDistribution][client] = ttd
            
            if sDistribution not in pValuesDict['KS'].keys():
                pValuesDict['KS'][sDistribution] = {}
            if sDistribution not in pValuesDict['CvM'].keys():
                pValuesDict['CvM'][sDistribution] = {} 
        
            pValuesDict['KS'][sDistribution][client] = model.pvalueKS
            pValuesDict['CvM'][sDistribution][client] = model.pvalueCvM
    
    
    clientsList = [i for i in periodsDict.keys()]    
    distrList = [i for i in stats.modelsLocDict.keys()]
    
    ttdDf = pd.DataFrame(index=clientsList)    
    for sDistribution, clientsDict in ttdDict.items():
        ttdDf[sDistribution] = ttdDf.index.map(clientsDict)
    ttdDf['frequency'] = ttdDf.index.map(frequencyDict)

    
    ttcDf = pd.DataFrame(index=clientsList)    
    for sDistribution, clientsDict in ttcDict.items():
        ttcDf[sDistribution] = ttcDf.index.map(clientsDict)    
    ttcDf['frequency'] = ttcDf.index.map(frequencyDict)
    
    # max for pvalues
    ksPvalueDf = pd.DataFrame(index=clientsList)
    for sDistribution, clientsDict in pValuesDict['KS'].items():
        ksPvalueDf[sDistribution] = ksPvalueDf.index.map(clientsDict)     

    cvmPvalueDf = pd.DataFrame(index=clientsList)
    for sDistribution, clientsDict in pValuesDict['CvM'].items():
        cvmPvalueDf[sDistribution] = cvmPvalueDf.index.map(clientsDict) 
        
    # best distributions
    ksBestSeries = ksPvalueDf.idxmax(axis=1)
    cvmBestSeries = cvmPvalueDf.idxmax(axis=1)

    ksBestCountSeries = ksBestSeries.value_counts()
    ksBestCountSeries = ksBestCountSeries / ksBestCountSeries.sum()
    cvmBestCountSeries = cvmBestSeries.value_counts()
    cvmBestCountSeries = cvmBestCountSeries / cvmBestCountSeries.sum()
    
    death_list = []
    for distr in distrList:
        filtered = ttdDf[[distr]].dropna(axis=0, how='any')
        death_list.append(filtered[distr].values)
        
    churn_list = []
    for distr in distrList:
        filtered = ttcDf[[distr]].dropna(axis=0, how='any')
        churn_list.append(filtered[distr].values)
        
#   Box plot TTD TTC
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(21,15))

    _ = ax3_1.boxplot(death_list, vert = 1)
    ax3_1.set_xticklabels(distrList)
    ax3_1.set_ylabel('Time to death (days)')
    ax3_1.title.set_text('Box plot time to death')
    
    _ = ax3_2.boxplot(churn_list, vert = 1)
    ax3_2.set_xticklabels(distrList)
    ax3_2.set_ylabel('Time to churn')
    ax3_2.title.set_text('Box plot time to churn')
    
    fileName = '{}{}'.format('Box plot TTD TTC', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
# Plot Median TTD and TTC vs. number of purchases
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(19,15))
    
    for i, distr in enumerate(distrList):
        g = ttdDf.groupby('frequency').agg({distr: 'median'})
        ax4_1.plot(g, c=colors[i], lw=1, alpha=0.6, label=distr)        
        
    ax4_1.title.set_text('Median TTD vs. number of purchases'.format()) 
    ax4_1.set_xlabel('Number of purchases')
    ax4_1.set_ylabel('Median TTD')
    ax4_1.legend()
    
    for i, distr in enumerate(distrList):
        g = ttcDf.groupby('frequency').agg({distr: 'median'})
        ax4_2.plot(g, c=colors[i], lw=1, alpha=0.6, label=distr)        
        
    ax4_2.title.set_text('Median TTC vs. number of purchases'.format()) 
    ax4_2.set_xlabel('Number of purchases')
    ax4_2.set_ylabel('Median TTC')
    ax4_2.legend()
    
    fileName = '{}{}'.format('Median TTD and TTC vs. number of purchases', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
#   Bar plot Hishest p-value KS and CvM
    fig5, (ax5_1, ax5_2) = plt.subplots(1, 2, figsize=(19,15))

    ax5_1.bar(ksBestCountSeries.index, ksBestCountSeries.values, label=ksBestCountSeries.index)
    ax5_1.title.set_text('Kolmogorov–Smirnov test') 
    ax5_1.set_ylabel('Relative frequency')

    ax5_2.bar(cvmBestCountSeries.index, cvmBestCountSeries.values, label=cvmBestCountSeries.index)
    ax5_2.title.set_text('Cramér–von Mises criterion') 
    ax5_2.set_ylabel('Relative frequency')

    title = 'Highest p-value KS and CvM'
    fig5.suptitle(title, fontsize=14)
        
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
# Histogram of p-value distributions Kolmogorov–Smirnov test
    M = 2
    N = 3
    bins=100
    fig6, ax6 = plt.subplots(M, N, figsize=(19,15))  
    for i in range(0, M):
        for j in range(0, N):
            k = i * N + j
            distr = distrList[k]            
            # ax[i][j].hist(list(ksPvalueDf[distr].values), bins=bins, color='blue', edgecolor='black', density=True, stacked=True)
            ax6[i][j].hist(ksPvalueDf[distr].values, bins=bins, color='blue', edgecolor='black')
            ax6[i][j].title.set_text('{}'.format(distr))
            ax6[i][j].set_ylabel('Frequency')

    title = 'Histogram of p-value distributions Kolmogorov–Smirnov test'
    fig6.suptitle(title, fontsize=14)

    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()             

# Histogram of p-value distributions Cramér–von Mises criterion
    M = 2
    N = 3
    bins=100
    fig7, ax7 = plt.subplots(M, N, figsize=(19,15))  
    for i in range(0, M):
        for j in range(0, N):
            k = i * N + j
            distr = distrList[k]            
            ax7[i][j].hist(ksPvalueDf[distr].values, bins=bins, color='blue', edgecolor='black')
            ax7[i][j].title.set_text('{}'.format(distr))
            ax7[i][j].set_ylabel('Frequency')

    title = 'Histogram of p-value distributions Cramér–von Mises criterion'
    fig7.suptitle(title, fontsize=14)

    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()  








