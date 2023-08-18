# Plot many statistics
# fit distributions to clients inter-purchase sequences
# use quantiles

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
from FrameArray import FrameArray
# from astropy.stats import kuiper 
# import lib2

class Nan():
    statistic = np.nan
    pvalue = np.nan
    
    def __init__(self):
        return
    
def getbin(value, margins):
    n = len(margins) - 1
    for i in range(0, n, 1):
        if value > margins[i] and value <= margins[i+1]:
            return i+1
    raise ValueError('value {} not within margins'.format(value))
    
def getBinsDesc():
    bins = list(range(1, nBins+1, 1))
    sX = []
    x1 = [x for x in mapBinToCount.values()]
    x2 = [x for x in mapBinToMargins.values()]
    for i, j in zip(x1, x2):
        s = 'Pop: {}\n{}'.format(i, j)
        sX.append(s)
    return bins, sX
        
def countTest(prefix, distributions, distResultsDf):
#   best distribution by Kolmogorov–Smirnov
    columns = ['{}{}'.format(prefix, x) for x in distributions]
    tmpDf = distResultsDf[columns].copy(deep=True)
    tmpDf.columns = distributions
    ksSeries = tmpDf.idxmin(axis=1)
    ksDf = ksSeries.to_frame()
    ksDf.columns = ['Distribution']
    ksDf['count'] = 1
    ksDf['Size'] = distResultsDf['Size']
    countKsDf = ksDf.groupby('Distribution').agg({'count': 'count'}).reset_index()
    countKsSizeDf = ksDf.groupby(['Size', 'Distribution']).agg({'count': 'count'}).reset_index()
    countKsSizeDf['bin'] = countKsSizeDf['Size'].map(mapSizeToBin)  
    countKsBinDf = countKsSizeDf.groupby(['bin', 'Distribution']).agg({'count': 'sum'}).reset_index()   
    return countKsDf, countKsSizeDf, countKsBinDf
    
def plot1(distResultsDf):
# Plot Frequency vs. number of repeat purchases
    fig, ax = plt.subplots(1, 1, figsize=(21,15))  
    g = distResultsDf.groupby('Size').agg({'count': 'count'}).reset_index()  
    ax.bar(g['Size'], g['count'])
    ax.title.set_text('Frequency vs. number of repeat purchases')
    ax.set_xlabel('Number of repeat purchases')
    ax.set_ylabel('Frequency')
    fileName = '{}{}'.format('Frequency vs. number of repeat purchases', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    return
    
def plot2(distResultsDf):

    death_vs_n_list = []
    for distr in distributionsList:
        death_vs_n = distResultsDf.groupby('Size').agg({'death{}'.format(distr): 'mean'}).reset_index()
        death_vs_n_list.append(death_vs_n)
        
    observed_vs_n_list = []
    for distr in distributionsList:
        observed_vs_n = distResultsDf.groupby('Size').agg({'observed{}'.format(distr): 'mean'}).reset_index()
        observed_vs_n_list.append(observed_vs_n)
        
# Plot mean time to death vs. number of repeat purchases   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21,15))
    
    for i, distr in enumerate(distributionsList):
        ax1.plot(death_vs_n_list[i]['Size'], death_vs_n_list[i]['death{}'.format(distr)].values, '{}-'.format(colors[i]), lw=1, alpha=0.6, label=distr)

    ax1.title.set_text('Mean time to death after last transaction vs. number of repeat purchases. CDF={}'.format(pDeath))
    ax1.set_xlabel('Number of repeat purchases')
    ax1.set_ylabel('Mean time to death after last transaction')
    ax1.legend()

    for i, distr in enumerate(distributionsList):
        ax2.plot(observed_vs_n_list[i]['Size'], observed_vs_n_list[i]['observed{}'.format(distr)].values, '{}-'.format(colors[i]), lw=1, alpha=0.6, label=distr)

    ax2.title.set_text('Mean time to observed death vs. number of repeat purchases. CDF={}'.format(pDeathObserved))
    ax2.set_xlabel('Number of repeat purchases')
    ax2.set_ylabel('Mean time to observed death after last transaction')
    ax2.legend()    

    fileName = '{}{}'.format('Mean time to death vs number of repeat purchases', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    return

def plot3(distResultsDf):
    death_list = []
    for distr in distributionsList:
        death = distResultsDf.groupby('Size').agg({'death{}'.format(distr): 'mean'}).reset_index()
        filtered = death.dropna(axis=0, how='any')
        death_list.append(filtered['death{}'.format(distr)].values)
        
    observed_list = []
    for distr in distributionsList:
        observed = distResultsDf.groupby('Size').agg({'observed{}'.format(distr): 'mean'}).reset_index()
        filtered = observed.dropna(axis=0, how='any')
        observed_list.append(filtered['observed{}'.format(distr)].values)
                
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(21,15))

    _ = ax3.boxplot(death_list, vert = 1)
    ax3.set_xticklabels(distributionsList)
    ax3.set_ylabel('Mean time to death from last transaction')
    ax3.title.set_text('Box plot mean time to death after last transaction')
        
    _ = ax4.boxplot(observed_list, vert = 1)
    ax4.set_xticklabels(distributionsList)
    ax4.set_ylabel('Mean time to observed death from last transaction')
    ax4.title.set_text('Box plot mean time to observed death after last transaction')
    
    fileName = '{}{}'.format('Box plot mean time to death', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    return

def plot4(prefix):
# Box plot time to {} after last transaction for each quantile
    columns = ['{}{}'.format(prefix, x) for x in distributionsList]

    dataIn = distResultsDf[columns + ['Size', 'bin']].copy(deep=True)

    bins, sX = getBinsDesc()

    N = 4
    if nBins <= 9:
        N = 3
        
    fig, ax = plt.subplots(N, N, figsize=(21,15))        
    for i in range(0, N, 1):
        for j in range(0, N, 1):
            k = i * N + j
            if k >= nBins:
                break            
            b = bins[k]
             
            df1 = dataIn[dataIn['bin'] == b] 
            dataOut = []
            for column in columns:
                df2 = df1[column]
                df2.dropna(how='any', inplace=True)                    
                dataOut.append(df2.values)
            _ = ax[i][j].boxplot(dataOut, vert = 1)
            ax[i][j].title.set_text(sX[k])
            ax[i][j].set_xticklabels(distributionsList)    
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.5)    

    title = 'Box plot time to {} after last transaction for each quantile'.format(prefix)
    fig.suptitle(title, fontsize=14)
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    return

def plot5(sTest, countBinDf):
    # move data to frameArray    
    frameArray = FrameArray(rowNames=distributionsList, columnNames=list(range(1, nBins+1)), 
        df=np.zeros(shape=(len(distributionsList), nBins)), dtype=float)

    for i, row in countBinDf.iterrows():
        frameArray.put(row['Distribution'], row['bin'], row['count'])
        
    barsSpan = 1
    barWidth = 0.15
    assert nDistributions * barWidth < barsSpan
    x = np.arange(start=0, stop=barsSpan*nBins , step=barsSpan)
    
    xList = []
    for i in range(0, nDistributions):
        x1 = x + i * barWidth
        xList.append(x1)        
        
    bins, sX = getBinsDesc()  
      
    fig, ax = plt.subplots(1, 1, figsize=(19,15))
    for i in range(0, len(distributionsList)):
        distr = distributionsList[i]
        rowIdx = frameArray.getIdxRow(distr)
        y = frameArray.array[rowIdx, :]
        plt.bar(xList[i], y, color=mapDistribToColor.get(distr), width=barWidth, label=distr)

    plt.xticks(xList[int(nDistributions / 2)], sX)
    plt.xlabel('Number of customers at each bin \n Margins for bins: number of repeat purchases', fontsize = 12)
    plt.ylabel('Number of customers',  fontsize = 12)
    title = 'Test statistics {}. Frequency of distributions per quantile'.format(sTest)
    plt.title(title)
    plt.legend()

    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    return

def plot6(sTest, countBinDf):
# Bar plots for distributions for number of transactions 3 .. 17     

    N = 4
    if nBins <= 9:
        N = 3
    
    bins, sX = getBinsDesc()  

    fig3, ax = plt.subplots(N, N, figsize=(21,18))
    
    for i in range(0, N, 1):
        for j in range(0, N, 1):
            k = i * N + j
            if k >= nBins:
                break
            b = bins[k]                
            df1 = countBinDf[countBinDf['bin'] == b]    
            ax[i][j].bar(df1['Distribution'].values, df1['count'].values)
            ax[i][j].title.set_text(sX[k])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.5)    
        
    title = 'Bar plot best distribution per quantile. {}'.format(sTest)
    fig3.suptitle(title, fontsize=14)
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    return

def plot7(sTest, minSize=2):
    
    data = distResultsDf[distResultsDf['Size'] >= minSize].copy(deep=True)

    # all distributions plot
    M = 2
    N = 3
    alpha = 0.05
    bins = round(1 / alpha)
    nObservations = len(data)
    fig, ax = plt.subplots(M, N, figsize=(19,15))  
    for i in range(0, M):
        for j in range(0, N):
            k = i * N + j
            distr = distributionsList[k]
            ax[i][j].hist(data['p_{}{}'.format(sTest, distr)].values, bins=bins)
            count = np.sum(np.where(data['p_{}{}'.format(sTest, distr)].values <= alpha, 1, 0))
            fraction = round(count / nObservations, 4)
            ax[i][j].title.set_text('{}\nfraction bad={}'.format(distr, fraction))

    title = 'Histogram of p-values of 6 distributions, {} test. Min number of repeat purchases = {}'.format(testDict[sTest], minSize)
    fig.suptitle(title, fontsize=14)

    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()


    # plot best p-value only
    columns = ['p_{}{}'.format(sTest, distr) for distr in distributionsList]

    data = data[columns].max(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(19,15))  
    ax.hist(data.values, bins=bins)

    count = np.sum(np.where(data.values <= alpha, 1, 0))
    fraction = round(count / nObservations, 4)    
    ax.title.set_text('fraction bad={}'.format(fraction))

    title = 'Histogram of p-values taken from best distribution. {} test. Min number of repeat purchases = {}'.format(testDict[sTest], minSize)
    fig.suptitle(title, fontsize=14)

    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close()
    
    return

################################################################################

if __name__ == "__main__":
    
    MIN_FREQUENCY = State.MIN_FREQUENCY
    mergeTransactionsPeriod=State.TS_STEP
    nan = Nan()
    
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

#   Index(['iId', 'ds', 'Sale'], dtype='object')
    transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(dataName)), 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    
    periodsDict = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
        leftMarginDate=None, rightMarginDate=None, 
        bIncludeStart=True, bIncludeEnd=True, minFrequency=MIN_FREQUENCY)
    
    # get sequences
    seqList = []
    for key, value in periodsDict.items():
        n = np.array(value)
        seqList.append(n)
    
    # fit distributions
    pDeath = 0.5
    pDeathObserved = 0.95
    sizeList = []
    
    ksNormList = []
    ksLognormList = []
    ksExponList = []
    ksGammaList = []
    ksWeibullList = []
    ksGumbelList = []
    ksGengammaList = []
    
    p_ksNormList = []
    p_ksLognormList = []
    p_ksExponList = []
    p_ksGammaList = []
    p_ksWeibullList = []
    p_ksGumbelList = []
    p_ksGengammaList = []
    
    crNormList = []
    crLognormList = []
    crExponList = []
    crGammaList = []
    crWeibullList = []
    crGumbelList = []
    crGengammaList = []

    p_crNormList = []
    p_crLognormList = []
    p_crExponList = []
    p_crGammaList = []
    p_crWeibullList = []
    p_crGumbelList = []
    p_crGengammaList = []
    
    deathNormList = []
    observedNormList = []
    deathLognormList = []
    observedLognormList = []
    deathExponList = []
    observedExponList = []
    deathGammaList = []
    observedGammaList = []
    deathWeibullList = []
    observedWeibullList = []
    deathGumbelList = []
    observedGumbelList = []
    deathGengammaList = []
    observedGengammaList = []
    
    for i in tqdm(range(0, len(seqList), 1)):
        y = seqList[i]
        sizeList.append(len(y))        

        paramsNorm = norm.fit(y, method='MLE') # scale = 1 / lambda
        paramsLognorm = lognorm.fit(y, floc=0, method='MLE') # s = sigma and scale = exp(mu).    
        paramsExpon = expon.fit(y, floc=0, method='MLE') # scale = 1 / lambda
        try:
            paramsGamma = gamma.fit(y, floc=0, method='MLE') # scale = 1 / beta
        except:
            paramsGamma = None
        paramsWeibull = weibull_min.fit(y, floc=0, method='MLE')
        paramsGumbel = gumbel_r.fit(y, method='MLE')
        paramsGengamma = gengamma.fit(y, floc=0, method='MLE')

# Kolmogorov–Smirnov test
# statistics smaller == better; p-value greater - better; p-value > 0.05 == good
# kstest, cramervonmises, kuiper good: statistic -> 0; p-Value -> 1

        ksNorm = kstest(y, norm.cdf, args=paramsNorm, alternative='two-sided', mode='exact')    
        ksLognorm = kstest(y, lognorm.cdf, args=paramsLognorm, alternative='two-sided', mode='exact')
        ksExpon = kstest(y, expon.cdf, args=paramsExpon, alternative='two-sided', mode='exact')
        if paramsGamma is not None:
            ksGamma = kstest(y, gamma.cdf, args=paramsGamma, alternative='two-sided', mode='exact')
        else:
            ksGamma = nan        
        ksWeibull = kstest(y, weibull_min.cdf, args=paramsWeibull, alternative='two-sided', mode='exact')
        ksGumbel = kstest(y, gumbel_r.cdf, args=paramsGumbel, alternative='two-sided', mode='exact')
        ksGengamma = kstest(y, gengamma.cdf, args=paramsGengamma, alternative='two-sided', mode='exact')

        ksNormList.append(ksNorm.statistic)
        ksLognormList.append(ksLognorm.statistic)
        ksExponList.append(ksExpon.statistic)
        ksGammaList.append(ksGamma.statistic)
        ksWeibullList.append(ksWeibull.statistic)
        ksGumbelList.append(ksGumbel.statistic)
        ksGengammaList.append(ksGengamma.statistic)

        p_ksNormList.append(ksNorm.pvalue)
        p_ksLognormList.append(ksLognorm.pvalue)
        p_ksExponList.append(ksExpon.pvalue)
        p_ksGammaList.append(ksGamma.pvalue)
        p_ksWeibullList.append(ksWeibull.pvalue)
        p_ksGumbelList.append(ksGumbel.pvalue)
        p_ksGengammaList.append(ksGengamma.pvalue)

# Cramér–von Mises criterion
        crNorm = cramervonmises(y, norm.cdf, paramsNorm)
        crLognorm = cramervonmises(y, lognorm.cdf, args=paramsLognorm)
        crExpon = cramervonmises(y, expon.cdf, args=paramsExpon)
        if paramsGamma is not None:
            crGamma = cramervonmises(y, gamma.cdf, args=paramsGamma)
        else:
            crGamma = nan
        crWeibull = cramervonmises(y, weibull_min.cdf, args=paramsWeibull)
        crGumbel = cramervonmises(y, gumbel_r.cdf, args=paramsGumbel)
        crGengamma = cramervonmises(y, gengamma.cdf, args=paramsGengamma)

        crNormList.append(crNorm.statistic)
        crLognormList.append(crLognorm.statistic)
        crExponList.append(crExpon.statistic)
        crGammaList.append(crGamma.statistic)
        crWeibullList.append(crWeibull.statistic)
        crGumbelList.append(crGumbel.statistic)
        crGengammaList.append(crGengamma.statistic)

        p_crNormList.append(crNorm.pvalue)
        p_crLognormList.append(crLognorm.pvalue)
        p_crExponList.append(crExpon.pvalue)
        p_crGammaList.append(crGamma.pvalue)
        p_crWeibullList.append(crWeibull.pvalue)
        p_crGumbelList.append(crGumbel.pvalue)
        p_crGengammaList.append(crGengamma.pvalue)


# time to death and time to observed death
        deathNorm = norm.ppf(pDeath, loc=paramsNorm[0], scale=paramsNorm[1])
        observedNorm = norm.ppf(pDeathObserved, loc=paramsNorm[0], scale=paramsNorm[1])
        deathLognorm = lognorm.ppf(pDeath, paramsLognorm[0], loc=paramsLognorm[1], scale=paramsLognorm[2])
        observedLognorm = lognorm.ppf(pDeathObserved, paramsLognorm[0], loc=paramsLognorm[1], scale=paramsLognorm[2])
        deathExpon = expon.ppf(pDeath, loc=paramsExpon[0], scale=paramsExpon[1])
        observedExpon = expon.ppf(pDeathObserved, loc=paramsExpon[0], scale=paramsExpon[1])
        if paramsGamma is not None:        
            deathGamma = gamma.ppf(pDeath, paramsGamma[0], loc=paramsGamma[1], scale=paramsGamma[2])
            observedGamma = gamma.ppf(pDeathObserved, paramsGamma[0], loc=paramsGamma[1], scale=paramsGamma[2])
        else:
            deathGamma = np.nan
            observedGamma = np.nan
        deathWeibull = weibull_min.ppf(pDeath, paramsWeibull[0], loc=paramsWeibull[1], scale=paramsWeibull[2])
        observedWeibull = weibull_min.ppf(pDeathObserved, paramsWeibull[0], loc=paramsWeibull[1], scale=paramsWeibull[2])
        deathGumbel = gumbel_r.ppf(pDeath, loc=paramsGumbel[0], scale=paramsGumbel[1])
        observedGumbel = gumbel_r.ppf(pDeathObserved, loc=paramsGumbel[0], scale=paramsGumbel[1])
        deathGengamma = gengamma.ppf(pDeath, paramsGengamma[0], paramsGengamma[1], loc=paramsGengamma[2], scale=paramsGengamma[3])
        observedGengamma = gengamma.ppf(pDeathObserved, paramsGengamma[0], paramsGengamma[1], loc=paramsGengamma[2], scale=paramsGengamma[3])

        deathNormList.append(deathNorm)
        observedNormList.append(observedNorm)
        deathLognormList.append(deathLognorm)
        observedLognormList.append(observedLognorm)        
        deathExponList.append(deathExpon)
        observedExponList.append(observedExpon)
        deathGammaList.append(deathGamma)
        observedGammaList.append(observedGamma)
        deathWeibullList.append(deathWeibull)
        observedWeibullList.append(observedWeibull)   
        deathGumbelList.append(deathGumbel)
        observedGumbelList.append(observedGumbel) 
        deathGengammaList.append(deathGengamma)
        observedGengammaList.append(observedGengamma) 
    
    distResultsDf = pd.DataFrame(columns=['Size'])
    distResultsDf['Size'] = sizeList
    
    distResultsDf['ksNorm'] = ksNormList
    distResultsDf['ksLognorm'] = ksLognormList
    distResultsDf['ksExpon'] = ksExponList
    distResultsDf['ksGamma'] = ksGammaList
    distResultsDf['ksWeibull'] = ksWeibullList
    # distResultsDf['ksGumbel'] = ksGumbelList
    distResultsDf['ksGengamma'] = ksGengammaList

    distResultsDf['p_ksNorm'] = p_ksNormList
    distResultsDf['p_ksLognorm'] = p_ksLognormList
    distResultsDf['p_ksExpon'] = p_ksExponList
    distResultsDf['p_ksGamma'] = p_ksGammaList
    distResultsDf['p_ksWeibull'] = p_ksWeibullList
    # distResultsDf['p_ksGumbel'] = p_ksGumbelList
    distResultsDf['p_ksGengamma'] = p_ksGengammaList
    
    distResultsDf['crNorm'] = crNormList
    distResultsDf['crLognorm'] = crLognormList
    distResultsDf['crExpon'] = crExponList
    distResultsDf['crGamma'] = crGammaList
    distResultsDf['crWeibull'] = crWeibullList
    # distResultsDf['crGumbel'] = crGumbelList
    distResultsDf['crGengamma'] = crGengammaList

    distResultsDf['p_crNorm'] = p_crNormList
    distResultsDf['p_crLognorm'] = p_crLognormList
    distResultsDf['p_crExpon'] = p_crExponList
    distResultsDf['p_crGamma'] = p_crGammaList
    distResultsDf['p_crWeibull'] = p_crWeibullList
    # distResultsDf['p_crGumbel'] = p_crGumbelList
    distResultsDf['p_crGengamma'] = p_crGengammaList
    
    distResultsDf['deathNorm'] = deathNormList
    distResultsDf['deathLognorm'] = deathLognormList
    distResultsDf['deathExpon'] = deathExponList
    distResultsDf['deathGamma'] = deathGammaList
    distResultsDf['deathWeibull'] = deathWeibullList
    # distResultsDf['deathGumbel'] = deathGumbelList
    distResultsDf['deathGengamma'] = deathGengammaList

    distResultsDf['observedNorm'] = observedNormList
    distResultsDf['observedLognorm'] = observedLognormList
    distResultsDf['observedExpon'] = observedExponList
    distResultsDf['observedGamma'] = observedGammaList
    distResultsDf['observedWeibull'] = observedWeibullList
    # distResultsDf['observedGumbel'] = observedGumbelList
    distResultsDf['observedGengamma'] = observedGengammaList
    distResultsDf['count'] = 1
    
###############################################################################

#   quantiles by frequency
    frequencyList = list(distResultsDf['Size'])
    frequencyUniqueList = sorted(list(set(frequencyList)))

    nQuantiles = 16 # numper of plots
    marginsQ = list(range(1, nQuantiles, 1))
    marginsQ = [i/nQuantiles for i in marginsQ]

    marginsV = np.quantile(frequencyList, marginsQ, method='inverted_cdf')
    marginsV = list(marginsV)
    
    marginsUniqueV = sorted(list(set(marginsV)))
    marginsUniqueV.insert(0, 0)
    marginsUniqueV.append(np.inf)
    
    mapSizeToCount = distResultsDf.groupby('Size').agg({'count': 'count'}).reset_index()
    mapSizeToCount = mapSizeToCount.set_index('Size')
    mapSizeToCount = mapSizeToCount['count'].to_dict()
    
    # dict for bins; independent on test
    mapSizeToBin = {}
    for f in frequencyUniqueList:
        b = getbin(f, marginsUniqueV)
        mapSizeToBin[f] = b
    distResultsDf['bin'] = distResultsDf['Size'].map(mapSizeToBin)

    mapBinToCount = distResultsDf.groupby('bin').agg({'count': 'sum'}).reset_index()
    mapBinToCount.set_index('bin', inplace=True)
    mapBinToCount = mapBinToCount['count'].to_dict() 
    
    mapBinToMargins = {}
    for i in range(0, len(marginsUniqueV)-1, 1):
        s = '{}<x<={}'.format(marginsUniqueV[i], marginsUniqueV[i+1])
        mapBinToMargins[i+1] = s
        
    nBins = max(mapSizeToBin.values())
    
    ksCountDf, ksCountSizeDf, ksCountBinDf = countTest('ks', distributionsList, distResultsDf)
    crCountDf, crCountSizeDf, crCountBinDf = countTest('cr', distributionsList, distResultsDf)

    # independent on test stats
    plot1(distResultsDf)
    plot2(distResultsDf)
    plot3(distResultsDf)
    plot4('death')
    plot4('observed')
    
    # dependent on test stats
    plot5('Cramer–von Mises', crCountBinDf)
    plot5('Kolmogorov–Smirnov', ksCountBinDf)
    
    plot6('Cramer–von Mises', crCountBinDf)
    plot6('Kolmogorov–Smirnov', ksCountBinDf)

    plot7('cr', minSize=2)
    plot7('ks', minSize=2)

    plot7('cr', minSize=5)
    plot7('ks', minSize=5)   
    
    plot7('cr', minSize=10)
    plot7('ks', minSize=10)    



