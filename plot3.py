# simulation 1
# Plot 3
#!!! check

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import weibull_min
from scipy.stats import gumbel_r
from scipy.stats import gengamma

from scipy.stats import kstest
from scipy.stats import cramervonmises
from matplotlib import pyplot as plt
from pathlib import Path
import State
import lib3
from createStates import loadCDNOW
# from astropy.stats import kuiper
# from FrameArray import FrameArray
# from astropy.stats import kuiper 
import lib2
from Distributions import Statistics
from collections import Counter

def plotBarDistribution(model, nPoints, title, cut=None, random_state=None):
    # generate random sample
    randomList = model.rvs(size=nPoints, random_state=random_state)
    randomIntList = sorted(randomList)
    randomIntList = [round(x) for x in randomIntList]
    if cut is not None:
        randomIntList = [x for x in randomIntList if x <= cut]    
    c = Counter(randomIntList)
    
    fig, ax = plt.subplots(1, 1, figsize=(19,15))  
    ax.bar(c.keys(), c.values())
    ax.title.set_text(title)
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()    
    
    return

def plotScatter(x, y, title, xLabel, yLabel):    
    fig, ax = plt.subplots(1, 1, figsize=(19,15))  
    ax.scatter(x, y , s=1, c='b')
    ax.title.set_text(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()    
    return

class Quantiles(object):    
    
    def __init__(self, nQuantiles, trunc=4):
        self.nQuantiles = nQuantiles
        self.grid = list(np.linspace(0, 1, nQuantiles, endpoint=False))
        if trunc is not None:
            self.grid = [round(x, trunc) for x in self.grid]
        return

    def fit(self, model):
        separators = []
        model = stats4.bestModel    
        for q in self.grid:
            v = model.ppf(q)
            v = round(v)
            separators.append(v)
        separators.append(np.inf)
                
        self.margins = []    
        for i in range(0, self.nQuantiles, 1):
            v = (separators[i], separators[i+1])
            self.margins.append(v)

        weightsCumDict = {}
        for s1 in separators[1:]: # skip leading zero
            value = weightsCumDict.get(s1, 0)
            value += 1
            weightsCumDict[s1] = value
        
        weights = []
        for value in weightsCumDict.values():
            weights.append(value)
                        
        # divide by number of quantiles
        self.weights = [x / self.nQuantiles for x in weights]
        self.nPartitions = len(self.weights)   
        
        return
    
    def fillPartitions(self, dataDict):
        partitions = [[] for x in range(0, self.nQuantiles)]
        for client, value in dataDict.items():
            for i, (left, right) in enumerate(self.margins):
                if value >= left and value < right:
                    partitions[i].append(client)
                    break
        return partitions
            
    
    
################################################################################

if __name__ == "__main__":
    
    MIN_FREQUENCY = State.MIN_FREQUENCY
    # mergeTransactionsPeriod=State.TS_STEP
    random_state = 101

    testDict = {'ks': 'Kolmogorov–Smirnov', 'cr': 'Cramer–von Mises'}
    distributionsList = ['Norm', 'Lognorm', 'Expon', 'Gamma', 'Weibull', 'Gengamma']
    colors = ['k', 'b', 'm', 'r', 'g', 'c']    
    mapDistribToColor = {}
    for distrib, color in zip(distributionsList, colors):
        mapDistribToColor[distrib] = color    
    nDistributions = len(distributionsList)
    
    dataName = 'CDNOW' # 'CDNOW' casa_2 plex_45 Retail Simulated1

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    resultsSubDir = os.path.join(resultsDir, dataName)    

    if not Path(resultsSubDir).exists():
        Path(resultsSubDir).mkdir(parents=True, exist_ok=True)

    transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(dataName)), 
        mergeTransactionsPeriod=1, minFrequency=MIN_FREQUENCY)
    
    dStart = transactionsDf['ds'].min()
    dEnd = transactionsDf['ds'].max()
    
    periodsDict = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
        leftMarginDate=None, rightMarginDate=None, 
        bIncludeStart=True, bIncludeEnd=True, minFrequency=MIN_FREQUENCY)
        
    recencyDict = lib3.getRecencyLoyaltyT(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, checkPointDate=dEnd, bIncludeStart=True, bIncludeEnd=True,\
        eventType='recency')
        
    loyaltyDict = lib3.getRecencyLoyaltyT(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, checkPointDate=dEnd, bIncludeStart=True, bIncludeEnd=True,\
        eventType='loyalty')    
        
    frequencyDict = lib3.getFrequency(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, bIncludeStart=True, bIncludeEnd=True,\
        minFrequency=1, exceptFirst=False)
    
    mapClientToFirstEvent = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, eventType='first')
        
    mapClientToLastEvent = lib3.getMarginalEvents(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, bIncludeStart=True, bIncludeEnd=False, eventType='last')

    mapClientToC = lib3.getClumpiness(transactionsDf, 'iId', 'ds', leftMarginDate=None,\
        rightMarginDate=None, bIncludeStart=True, bIncludeEnd=True)
    
    mapClientToT = lib3.getRecencyLoyaltyT(transactionsDf, 'iId', 'ds',\
        leftMarginDate=None, rightMarginDate=None,\
        bIncludeStart=True, bIncludeEnd=True, eventType='T')    
        
    mapClientToStandingCum = lib3.getValue(transactionsDf, 'iId', 'ds', 'Sale', value='sum',\
        leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, \
        bIncludeEnd=True, dropZeroValue=True, dropFirstEvent=False)

    mapClientToStandingMean = lib3.getValue(transactionsDf, 'iId', 'ds', 'Sale', value='mean',\
        leftMarginDate=None, rightMarginDate=None, bIncludeStart=True, \
        bIncludeEnd=True, dropZeroValue=True, dropFirstEvent=False)

        
    frequencySeq = []
    timeToFirstEventSeq = []
    recencySeq = []
    loyaltySeq = []
    spendingMeanSeq = []
    spendingCumSeq = []  
    for client, dFirstEvent in mapClientToFirstEvent.items():
        span1 = (dFirstEvent - dStart).days + 1
        timeToFirstEventSeq.append(span1)
        dLastEvent = mapClientToLastEvent[client]
        loyalty = loyaltyDict[client]
        loyaltySeq.append(loyalty)
        frequency = frequencyDict[client]
        frequencySeq.append(frequency)
        recency = recencyDict[client]
        recencySeq.append(recency)
        spendingMean = mapClientToStandingMean[client]
        spendingMeanSeq.append(spendingMean)
        spendingCum = mapClientToStandingCum[client]
        spendingCumSeq.append(spendingCum)
        

    # a = [x for x in mapClientToFirstEvent.values()]
                
    nPoints = len(frequencySeq)        
    
    stats1 = Statistics(frequencySeq, testStats='ks', criterion='pvalue') # gen gamma
    stats2 = Statistics(timeToFirstEventSeq, testStats='ks', criterion='pvalue')
    stats3 = Statistics(loyaltySeq, testStats='ks', criterion='pvalue')
    stats4 = Statistics(recencySeq, testStats='ks', criterion='pvalue')
    stats5 = Statistics(spendingMeanSeq, testStats='ks', criterion='pvalue')
    stats6 = Statistics(spendingCumSeq, testStats='ks', criterion='pvalue')
    
    
    
    
    plotBarDistribution(stats1.bestModel, nPoints, 'Frequency distribution simulated', 
                        cut=None, random_state=random_state)
    plotBarDistribution(stats2.bestModel, nPoints, 'timeToFirstEvent distribution simulated', 
                        cut=None, random_state=random_state)
    plotBarDistribution(stats3.bestModel, nPoints, 'Loyalty distribution simulated', 
                        cut=None, random_state=random_state)
    plotBarDistribution(stats4.bestModel, nPoints, 'Recency distribution simulated', 
                        cut=1500, random_state=random_state)
    plotBarDistribution(stats5.bestModel, nPoints, 'Spending mean distribution simulated', 
                        cut=None, random_state=random_state)    
    plotBarDistribution(stats6.bestModel, nPoints, 'Spending cumulative distribution simulated', 
                        cut=None, random_state=random_state)       
    
    
    


    plotScatter(timeToFirstEventSeq, frequencySeq, 'Frequency vs. time to first purchase',\
        'Time to first purchase', 'Frequency') 

    plotScatter(recencySeq, frequencySeq, 'Frequency vs. Recency',\
        'Recency', 'Frequency') 
        
    plotScatter(timeToFirstEventSeq, loyaltySeq, 'Loyalty vs. time to first purchase',\
        'Time to first purchase', 'Loyalty') 

    plotScatter(recencySeq, loyaltySeq, 'Loyalty vs. Recency',\
        'Recency', 'Loyalty') 
        
    plotScatter(timeToFirstEventSeq, spendingMeanSeq, 'Spending Mean vs. time to first purchase',\
        'Time to first purchase', 'Spending Mean') 

    plotScatter(recencySeq, spendingMeanSeq, 'Spending Mean vs. Recency',\
        'Recency', 'Spending Mean') 

    plotScatter(timeToFirstEventSeq, spendingCumSeq, 'Spending Cumulative vs. time to first purchase',\
        'Time to first purchase', 'Spending Cumulative') 

    plotScatter(recencySeq, spendingCumSeq, 'Spending Cumulative vs. Recency',\
        'Recency', 'Spending Cumulative')         
        
        
        
        
        
        
        
        
    # plt.scatter(timeToFirstEventSeq, frequencySeq , s=1, c='b')
    # plt.scatter(recencySeq, frequencySeq , s=1, c='b')    
    # plt.scatter(timeToFirstEventSeq, loyaltySeq , s=1, c='b')
    # plt.scatter(recencySeq, loyaltySeq , s=1, c='b')

    
    
    # raise RuntimeError()
    
    

    
    
    
    
    
    
    # quantilesRecency = Quantiles(10)
    # quantilesRecency.fit(stats4.bestModel)
    # partitionsRecency = quantilesRecency.fillPartitions(recencyDict)

    # qRecency = 0
    # partitionsRecency0 = partitionsRecency[qRecency]
    # frequencyDict0 = {x : y for x, y in frequencyDict.items() if x in partitionsRecency0}
    # frequencySeq0 = [x for x in frequencyDict0.values()]
    # statsFrequency0 = Statistics(frequencySeq0, testStats='ks', criterion='pvalue') # gen gamma


    # plotBarDistribution(statsFrequency0.bestModel, nPoints, 'frequency distribution of {} partition of recency distribution'.format(qRecency),
    #     cut=None, random_state=random_state)
    
    # subQuantilesFrequency = Quantiles(10)
    # subQuantilesFrequency.fit(statsFrequency0.bestModel)
    # subPrtitionsFrequency = subQuantilesFrequency.fillPartitions(frequencyDict0)
    
    # qFrequency = 0
    # subPartitionsFrequency0 = partitionsRecency[qFrequency]

    
    

    
    
    
    # # make quantiles
    # nQuantiles = 10
    # quantiles = list(np.linspace(0, 1, nQuantiles, endpoint=False))
    # # _ = quantiles.pop(0)
    # quantiles = [round(x, 4) for x in quantiles]
    
    # separators = []
    # model = stats4.bestModel    
    # for q in quantiles:
    #     v = model.ppf(q)
    #     v = round(v)
    #     separators.append(v)
    # separators.append(np.inf)
    
    # separators = sorted(list(set(separators)))
    
    # quantileMargins = []    
    # for i in range(0, nQuantiles, 1):
    #     v = (separators[i], separators[i+1])
    #     quantileMargins.append(v)
    
    
    
    # quantilesDict = {x: [] for x in range(0, nQuantiles)}
    # for client, dLastEvent in mapClientToLastEvent.items():
    #     span = (dEnd - dLastEvent).days
    #     for i, (left, right) in enumerate(quantileMargins):
    #         if span >= left and span < right:
    #             quantilesDict[i].append(client)
    #             break
        
    # idxQ = 0
    # quantile0 = quantilesDict[idxQ]
    # frequency0Seq = []
    # loyalty0Seq = []    
    # for client in quantile0:
    #     frequency = frequencyDict[client]
    #     frequency0Seq.append(frequency)        
    #     dFirstEvent = mapClientToFirstEvent[client]
    #     dLastEvent = mapClientToLastEvent[client]
    #     span = (dLastEvent - dFirstEvent).days
    #     loyalty0Seq.append(span)
        
    
    # stats01 = Statistics(frequency0Seq, testStats='ks', criterion='pvalue') # gen gamma
    # stats02 = Statistics(loyalty0Seq, testStats='ks', criterion='pvalue') # gen gamma
    
    # plotBarDistribution(stats01.bestModel, nPoints, 'Frequency {} quantile'.format(idxQ), cut=None, random_state=random_state)
    # plotBarDistribution(stats02.bestModel, nPoints, 'Loyalty {} quantile'.format(idxQ), cut=None, random_state=random_state)
    
    
    # stats01.bestModel.median()
    
    
    
    # get clients from one ABSENCE quantile
    #   get clients from one FREQUENCY subquantile
    
    # subQuantiles0Dict = {x: [] for x in range(0, nQuantiles)}
    # for client in quantile0:
    #     frequency = frequencyDict[client]
    #     for i, (left, right) in enumerate(quantileMargins):
    #         if span >= left and span < right:
    #             quantilesDict[i].append(client)
    #             break    
    
    
    # quantileMargins = []
    # for i in range(0, nQuantiles, 1):
    #     if i == nQuantiles - 1:
    #         m = (quantiles[i], np.inf)
    #     else:
    #         m = (quantiles[i], quantiles[i+1])
    #     quantileMargins.append(m)
    
    
    # model = stats4.bestModel
    # margins = []
    # for left, right in quantileMargins:
    #     leftMargin = model.ppf(left)
    #     rightMargin = model.ppf(right)
    #     if lib2.isNaN(rightMargin):
    #         rightMargin = np.inf
    #     margin = (leftMargin, rightMargin)
    #     margins.append(margin)


    
    
    
    # for client, dFirstEvent in mapClientToFirstEvent.items():
    #     span1 = (dFirstEvent - dStart).days + 1   
    #     timeToFirstEventSeq.append(span1)
    #     dLastEvent = mapClientToLastEvent[client]
    #     span2 = (dLastEvent - dFirstEvent).days + 1
    #     loyaltySeq.append(span2)
    #     frequency = frequencyDict[client]
    #     frequencySeq.append(frequency)
    #     span3 = (dEnd - dLastEvent).days + 1
        
        
    
    
    
    
    
    
    
    # stats4.bestModel.ppf(np.inf)

    
    
    
    
    # plt.scatter(frequencySeq, timeToFirstEventSeq, s=1, c='b')
    # plt.scatter(frequencySeq, loyaltySeq)
    
    
    
    # a = [7, 7]
    
    # statsA = Statistics(a, testStats='ks', criterion='pvalue') # gen gamma

    # mapClientToT['4069']
    
    # periodsDict['4080']
    
    
    # norm.pdf(0.05, loc=6, scale=30)
    
    
    # se = 30 / pow(6, 0.5)
    
    # (np.mean([1, 5, 5, 5, 5,5]) - 6) / se
    
    
    
    