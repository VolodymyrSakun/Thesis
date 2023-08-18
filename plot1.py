# fit distributions to clients inter-purchase sequences
# Plot different from analysis7

import os
import lib2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import weibull_min
from scipy.stats import ks_1samp
from scipy.stats import kstest
from scipy.stats import cramervonmises
from math import log
from math import pi
from math import exp
from scipy.special import gamma as gammaFunction
from scipy.optimize import minimize
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from pathlib import Path
import State
import lib3
from createStates import loadCDNOW


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# normal
def likelihoodNorm(x):
    n = len(x)    
    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    return -0.5 * n * log(2 * pi) - 0.5 * n * log(var) - 0.5 * sum(((x - mean) / std)**2)

def f_norm(x):
    n = len(y)
    return 0.5 * n * log(2 * pi) + 0.5 * n * log(x[1]) + 0.5 / x[1] * sum(np.power(y - x[0], 2))
    
# log-normal
def likelihoodLognorm(x, mu, sigma):
    n = len(x)
    return-0.5 * n * log(2 * pi * sigma**2) - sum(np.log(x)) - sum(np.log(x)**2) / (2 * sigma**2) + sum(np.log(x)) * mu / (sigma**2) - n * mu**2 / (2 * sigma**2)

def f_lognorm(x):
    # x[0] == mu, x[1] = sigma
    n = len(y)
    return -(-0.5 * n * log(2 * pi * x[1]**2) - sum(np.log(y)) - sum(np.log(y)**2) / (2 * x[1]**2) + sum(np.log(y)) * x[0] / (x[1]**2) - n * x[0]**2 / (2 * x[1]**2))

# exponential
def likelihoodExpon(x, lambd):
    n = len(x)    
    return n * log(lambd) - lambd * sum(x)

def f_expon(x):
    n = len(y)       
    return -(n * log(x[0]) - x[0] * sum(y))

# gamma
def likelihoodGamma(x, alpha, beta):
    n = len(x)       
    return (alpha - 1) * sum(np.log(x)) - 1 / beta * sum(x) - n * alpha * log(beta) - n * log(gammaFunction(alpha))

def f_gamma(x):
    n = len(y)       
    return -((x[0] - 1) * sum(np.log(y)) - 1 / x[1] * sum(y) - n * x[0] * log(x[1]) - n * log(gammaFunction(x[0])))

# weibull
def likelihoodWeibull(x, shape, scale):
    n = len(x)
    lambd = (1 / shape)**(1 / scale)
    return n * log(shape) - n * shape * log(lambd) - sum(np.power(x / lambd, shape)) + (shape - 1) * sum(np.log(x))

def f_weibull(x):
    # do not use
    n = len(y)
    lambd = (1 / x[0])**(1 / x[1])
    return -(n * log(x[0]) - n * x[0] * log(lambd) - sum(np.power(y / lambd, x[0])) + (x[0] - 1) * sum(np.log(y)))

def f_weibull2(x):
    n = len(y)
    return -(n * log(x[0]) - n * x[0] * log(x[1]) - sum(np.power(y / x[1], x[0])) + (x[0] - 1) * sum(np.log(y)))

def f_weibull3(x):
#    a, alpha, beta
#   x[0], x[1], x[2]
    n = len(y)
    return -(n * log(x[0]) + n * log(x[1]) - n * log(x[2]) + (x[1] - 1) * \
        sum(np.log(y / x[2])) - x[0] * sum(np.power(y / x[2], x[1])))
        
################################################################################

if __name__ == "__main__":
    
    MIN_FREQUENCY = State.MIN_FREQUENCY
    mergeTransactionsPeriod=State.TS_STEP

    dataName = 'CDNOW' # 'CDNOW' casa_2 plex_45 Retail Simulated1

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    resultsSubDir = os.path.join(resultsDir, dataName)    

    if not Path(resultsSubDir).exists():
        Path(resultsSubDir).mkdir(parents=True, exist_ok=True)

    # statesDict = lib2.loadObject(os.path.join(dataSubDir, 'saveDict.dat'))
    # nStates = len(statesDict)
    # transactionsDf = statesDict['transactionsDf'].copy(deep=True)
    # del(statesDict['transactionsDf'])
    # g = transactionsDf.groupby('iId')['ds'].apply(list)
    # visitsDict = g.to_dict() 
    
    # dStateList = sorted(list(statesDict.keys()))

    # dLast = dStateList[-1]
    # lastState = statesDict[dLast]


    # a = lastState.periodsDict

    
#   Index(['iId', 'ds', 'Sale'], dtype='object')
    transactionsDf = loadCDNOW(os.path.join(dataSubDir, '{}.txt'.format(dataName)), 
        mergeTransactionsPeriod=mergeTransactionsPeriod, minFrequency=MIN_FREQUENCY)
    
    periodsDict = lib3.getTimeBetweenEvents(transactionsDf, 'iId', 'ds', 
        leftMarginDate=None, rightMarginDate=None, 
        bIncludeStart=True, bIncludeEnd=True, minFrequency=MIN_FREQUENCY)


    
#   export to R    
    # rows = []
    # for key, value in a.items():
    #     l = [str(x) for x in value]
    #     s = ','.join(l)
    #     s += '\n'
    #     rows.append(s)
    
    # f = open('csv.csv', 'w')
    # f.writelines(rows)
    # f.close()

# imort from R
    # df1 = pd.read_csv(os.path.join(dataSubDir, 'x1.csv'))
    # df2 = pd.read_csv(os.path.join(dataSubDir, 'x2.csv'))
    # df = pd.concat([df1, df2], axis=1)
    
    # df1['count'] = 1
    # df2['count'] = 1

    # g1 = df1.groupby('x').agg({'count': 'count'}).reset_index()
    # g2 = df2.groupby('x').agg({'count': 'count'}).reset_index()
    
    # g3 = g1.merge(g2, on='x')
    # g3['count'] = g3['count_x'] + g3['count_y']
    
    # g1.plot(x='x', y='count',kind="bar")
    # plt.show()
    
    # g3.plot(x='x', y='count',kind="bar")
    # plt.show()    


    seqList = []
    for key, value in periodsDict.items():
        n = np.array(value)
        seqList.append(n)
    
    # i = 0
    # y = seqList[i]

    # paramsNorm = norm.fit(y, method='MLE') # scale = 1 / lambda
    # paramsLognorm = lognorm.fit(y, floc=0, method='MLE') # s = sigma and scale = exp(mu).    
    # paramsExpon = expon.fit(y, floc=0, method='MLE') # scale = 1 / lambda
    # paramsGamma = gamma.fit(y, floc=0, method='MLE') # scale = 1 / beta
    # paramsWeibull = weibull_min.fit(y, floc=0, method='MLE')

    # ll_expon = likelihoodExpon(y, 1 / paramsExpon[1])
    # ll_gamma = likelihoodGamma(y, paramsGamma[0], paramsGamma[2])
    # ll_norm = likelihoodNorm(y)    
    # ll_lognorm = likelihoodLognorm(y, log(paramsLognorm[2]), paramsLognorm[0])
    # ll_weibull = likelihoodWeibull(y, paramsWeibull[0], paramsWeibull[2])    
    
    
    pDeath = 0.5
    pDeathObserved = 0.98
    sizeList = []
    normList = []
    lognormList = []
    exponList = []
    gammaList = []
    weibullList = []
    statNormList = []
    statLognormList = []
    statExponList = []
    statGammaList = []
    statWeibullList = []
    pNormList = []
    pLognormList = []
    pExponList = []
    pGammaList = []
    pWeibullList = []

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
    
    x0 = np.array([1, 1])
    for i in tqdm(range(0, len(seqList), 1)):
        y = seqList[i]
        sizeList.append(len(y))
        
        resNorm = minimize(f_norm, x0, tol=1e-6, bounds=[(1e-99, np.inf), (1e-99, np.inf)])
        resLognorm = minimize(f_lognorm, x0, tol=1e-6, bounds=[(1e-99, np.inf), (1e-99, np.inf)])
        resExpon = minimize(f_expon, np.array([0.001]), tol=1e-6, bounds=[(1e-99, np.inf)])
        resGamma = minimize(f_gamma, x0, tol=1e-6, bounds=[(1e-99, np.inf), (1e-99, np.inf)])
        resWeibull = minimize(f_weibull2, x0, tol=1e-6, bounds=[(1e-99, np.inf), (1e-99, np.inf)])
        # resW = minimize(f_weibull3, [1, 1, 1], tol=1e-6, bounds=[(1e-99, np.inf), (1e-99, np.inf), (1e-99, np.inf)])
    
        normList.append(resNorm.fun)
        lognormList.append(resLognorm.fun)
        exponList.append(resExpon.fun)
        gammaList.append(resGamma.fun)
        weibullList.append(resWeibull.fun)

        paramsNorm = norm.fit(y, method='MLE') # scale = 1 / lambda
        paramsLognorm = lognorm.fit(y, floc=0, method='MLE') # s = sigma and scale = exp(mu).    
        paramsExpon = expon.fit(y, floc=0, method='MLE') # scale = 1 / lambda
        try:
            paramsGamma = gamma.fit(y, floc=0, method='MLE') # scale = 1 / beta
        except:
            paramsGamma = None
        paramsWeibull = weibull_min.fit(y, floc=0, method='MLE')

# Kolmogorov–Smirnov test
# statistics smaller == better; p-value greater - better
        ksNorm = ks_1samp(y, norm.cdf, paramsNorm)    
        ksLognorm = ks_1samp(y, lognorm.cdf, args=paramsLognorm)
        ksExpon = ks_1samp(y, expon.cdf, args=paramsExpon)
        if paramsGamma is not None:
            ksGamma = ks_1samp(y, gamma.cdf, args=paramsGamma)
        else:
            ksGamma = [np.nan, np.nan]        
        ksWeibull = ks_1samp(y, weibull_min.cdf, args=paramsWeibull)

        statNormList.append(ksNorm[0])
        pNormList.append(ksNorm[1])
        statLognormList.append(ksLognorm[0])
        pLognormList.append(ksLognorm[1])        
        statExponList.append(ksExpon[0])
        pExponList.append(ksExpon[1])
        statGammaList.append(ksGamma[0])
        pGammaList.append(ksGamma[1])
        statWeibullList.append(ksWeibull[0])
        pWeibullList.append(ksWeibull[1])        


        # ksNorm = cramervonmises(y, norm.cdf, paramsNorm)    
        # ksLognorm = cramervonmises(y, lognorm.cdf, args=paramsLognorm)
        # ksExpon = cramervonmises(y, expon.cdf, args=paramsExpon)
        # if paramsGamma is not None:
        #     ksGamma = cramervonmises(y, gamma.cdf, args=paramsGamma)
        # else:
        #     ksGamma = [np.nan, np.nan]        
        # ksWeibull = cramervonmises(y, weibull_min.cdf, args=paramsWeibull)

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


    distResultsDf = pd.DataFrame(columns=['Size', 'Norm', 'Lognorm', 'Expon', 'Gamma', 'Weibull'])
    distResultsDf['Size'] = sizeList
    
    distResultsDf['Norm'] = normList
    distResultsDf['Lognorm'] = lognormList
    distResultsDf['Expon'] = exponList
    distResultsDf['Gamma'] = gammaList
    distResultsDf['Weibull'] = weibullList

    distResultsDf['StatNorm'] = statNormList
    distResultsDf['StatLognorm'] = statLognormList
    distResultsDf['StatExpon'] = statExponList
    distResultsDf['StatGamma'] = statGammaList
    distResultsDf['StatWeibull'] = statWeibullList

    distResultsDf['pNorm'] = pNormList
    distResultsDf['pLognorm'] = pLognormList
    distResultsDf['pExpon'] = pExponList
    distResultsDf['pGamma'] = pGammaList
    distResultsDf['pWeibull'] = pWeibullList

    distResultsDf['deathNorm'] = deathNormList
    distResultsDf['deathLognorm'] = deathLognormList
    distResultsDf['deathExpon'] = deathExponList
    distResultsDf['deathGamma'] = deathGammaList
    distResultsDf['deathWeibull'] = deathWeibullList
    
    distResultsDf['observedNorm'] = observedNormList
    distResultsDf['observedLognorm'] = observedLognormList
    distResultsDf['observedExpon'] = observedExponList
    distResultsDf['observedGamma'] = observedGammaList
    distResultsDf['observedWeibull'] = observedWeibullList

    


    frequencyList = list(distResultsDf['Size'])
    paramsNormFreq = norm.fit(frequencyList, floc=0, method='MLE')
    quantileFreq = 0.95
    q90Freq = norm.ppf(quantileFreq, loc=paramsNormFreq[0], scale=paramsNormFreq[1])

    # n = 10
    # qList = list(range(1, n, 1))
    # qList = [i/n for i in qList]
    # # quantilesList = []
    # # for q in qList:
    # #     quantilesList.append(norm.ppf(q, loc=paramsNormFreq[0], scale=paramsNormFreq[1]))

    # quantilesList = np.quantile(frequencyList, qList, method='inverted_cdf')
    
    # norm.ppf(0.9, loc=paramsNormFreq[0], scale=paramsNormFreq[1])


    # x = np.linspace(-100, 300, 1000)
    # p = norm.pdf(x, loc=paramsNormFreq[0], scale=paramsNormFreq[1])
    # plt.plot(x, p)





    m = distResultsDf[['Norm', 'Lognorm', 'Expon', 'Gamma', 'Weibull']].idxmin(axis=1)
    mDf = m.to_frame()
    mDf.columns = ['Distribution']
    mDf['count'] = 1
    g1 = mDf.groupby('Distribution').agg({'count': 'count'}).reset_index()

    tmpDf = distResultsDf[['StatNorm', 'StatLognorm', 'StatExpon', 'StatGamma', 'StatWeibull']]
    tmpDf.columns = ['Norm', 'Lognorm', 'Expon', 'Gamma', 'Weibull']
    m2 = tmpDf.idxmin(axis=1)
    m2Df = m2.to_frame()
    m2Df.columns = ['Distribution']
    m2Df['count'] = 1
    m2Df['Size'] = distResultsDf['Size']
    g2 = m2Df.groupby('Distribution').agg({'count': 'count'}).reset_index()
    g3 = m2Df.groupby(['Size', 'Distribution']).agg({'count': 'count'}).reset_index()
    
    g4 = m2Df.groupby('Size').agg({'count': 'count'}).reset_index()
    mapSizeToCount = g4.set_index('Size')
    mapSizeToCount = mapSizeToCount['count'].to_dict()

    g3['total'] = g3['Size'].map(mapSizeToCount)
    g3['fraction'] = g3['count'].div(g3['total'])

#   Death vs number of inter-purchase delays
    death_vs_n = distResultsDf.groupby('Size').agg({'deathNorm': 'mean', 'deathLognorm': 'mean',
        'deathExpon': 'mean', 'deathGamma': 'mean', 'deathWeibull': 'mean',
        'observedNorm': 'mean', 'observedLognorm': 'mean',
        'observedExpon': 'mean', 'observedGamma': 'mean', 'observedWeibull': 'mean'}).reset_index()

    x = death_vs_n['Size'].values
    
    
# Plot mean time to death vs. number of inter-transactions    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21,15))
    
    ax1.plot(x, death_vs_n['deathNorm'].values, 'k-', lw=1, alpha=0.6, label='normal')
    ax1.plot(x, death_vs_n['deathLognorm'].values, 'b-', lw=1, alpha=0.6, label='log-normal')
    ax1.plot(x, death_vs_n['deathExpon'].values, 'm-', lw=1, alpha=0.6, label='expon')    
    ax1.plot(x, death_vs_n['deathGamma'].values, 'r-', lw=1, alpha=0.6, label='gamma')
    ax1.plot(x, death_vs_n['deathWeibull'].values, 'g-', lw=1, alpha=0.6, label='Weibull')
    ax1.title.set_text('Death vs. number of inter-purchase delays. CDF={}'.format(pDeath)) 
    ax1.set_xlabel('Number of transactions - 1')
    ax1.set_ylabel('Mean time to death after last transaction')
    ax1.legend()

    ax2.plot(x, death_vs_n['observedNorm'].values, 'k-', lw=1, alpha=0.6, label='normal')
    ax2.plot(x, death_vs_n['observedLognorm'].values, 'b-', lw=1, alpha=0.6, label='log-normal')
    ax2.plot(x, death_vs_n['observedExpon'].values, 'm-', lw=1, alpha=0.6, label='expon')    
    ax2.plot(x, death_vs_n['observedGamma'].values, 'r-', lw=1, alpha=0.6, label='gamma')
    ax2.plot(x, death_vs_n['observedWeibull'].values, 'g-', lw=1, alpha=0.6, label='Weibull')
    ax2.title.set_text('Observed death vs. number of inter-purchase delays. CDF={}'.format(pDeathObserved))
    ax2.legend()
    
    ax2.set_xlabel('Number of transactions - 1')
    ax2.set_ylabel('Mean time to observed death after last transaction')

    fileName = '{}{}'.format('Death vs number of inter-purchase delays', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 


# box plot dime to death distribution
    filtered_data = death_vs_n.dropna(axis=0, how='any')

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(21,15))

    data3 = [filtered_data['deathNorm'].values, filtered_data['deathLognorm'].values, 
            filtered_data['deathExpon'].values, filtered_data['deathGamma'].values,
            filtered_data['deathWeibull'].values]
    bp3 = ax3.boxplot(data3, vert = 1)
    ax3.set_xticklabels(['normal', 'log-normal', 'expon', 'gamma', 'Weibull'])
    ax3.set_ylabel('Mean time to death from last transaction')
    ax3.title.set_text('Mean time to death distribution')
        
    data4 = [filtered_data['observedNorm'].values, filtered_data['observedLognorm'].values, 
            filtered_data['observedExpon'].values, filtered_data['observedGamma'].values,
            filtered_data['observedWeibull'].values]
    bp4 = ax4.boxplot(data4, vert = 1)
    ax4.set_xticklabels(['normal', 'log-normal', 'expon', 'gamma', 'Weibull'])
    ax4.set_ylabel('Mean time to observed death from last transaction')
    ax4.title.set_text('Mean time to observed death distribution')
    
    fileName = '{}{}'.format('Mean time to death distribution', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    
# Bar plots for distributions for number of transactions 3 .. 17     
    nList = sorted(list(g3['Size'].unique()))
    nMax = max(int(q90Freq), 15)
    
    f = 0
    idx = 0
    while idx < nMax and idx < len(nList):
        fig3, ax = plt.subplots(4, 4, figsize=(21,15))
        
        for i in range(0, 4, 1):
            for j in range(0, 4, 1):
                k = i * 4 + j
                if k > nMax:
                    break
                idx = f * 16 + k
                if idx >= len(nList):
                    break                
                n = nList[idx]    
                g3_2 = g3[g3['Size'] == n]    
                ax[i][j].bar(g3_2['Distribution'].values, g3_2['count'].values)
                ax[i][j].title.set_text('# transactions = {}'.format(n+1))
    
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        
        fileName = '{}{}'.format('Multiple bar plot of best distriburions part {}'.format(f+1), '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close()         
        f += 1
        
    
# Plot frequency vs. number of transactions  
    fig4, ax4 = plt.subplots(1, 1, figsize=(21,15))
  
    ax4.bar(g4['Size'], g4['count'])
    ax4.title.set_text('Frequency vs. number of inter-transactions')
    ax4.set_xlabel('Number of inter-transactions')
    ax4.set_ylabel('Frequency')

    fileName = '{}{}'.format('Frequency vs. number of transactions', '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    

# Plot inverse frequence vs number of inter-transactions
    mapDistrToXY = {}
    for i, row in g3.iterrows():
        distr = mapDistrToXY.get(row['Distribution'])
        if not isinstance(distr, dict):
            distr = {'x': [], 'y': []}
        distr['x'].append(row['Size'])
        distr['y'].append(row['fraction'])
        mapDistrToXY[row['Distribution']] = distr
        
    
# Plot frequency vs. number of transactions  
    fig5, ax5 = plt.subplots(1, 1, figsize=(19,15))
    for distr, xyDict in mapDistrToXY.items():
        x = [i for i in xyDict['x'] if i <= nMax]
        y = xyDict['y'][0:len(x)]
        ax5.plot(x, y, lw=2, alpha=0.6, label=distr)
    title = 'Fraction of distribution frequency vs number of inter-transactions limited by max({} quantile, 16)'.format(quantileFreq)
    ax5.title.set_text(title)
    ax5.set_xlabel('Number of inter-transactions')
    ax5.set_ylabel('Fraction of distribution frequency')        
    ax5.legend()
    
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 


    
# Box plots for Time to death distribution for number of transactions 3 .. 17     
    nList = sorted(list(distResultsDf['Size'].unique()))
    
    columns = ['deathNorm', 'deathLognorm', 'deathExpon', 'deathGamma', 'deathWeibull']
    
    f = 0
    idx = 0
    while idx < nMax and idx < len(nList):        
        fig3, ax = plt.subplots(4, 4, figsize=(21,15))        
        for i in range(0, 4, 1):
            for j in range(0, 4, 1):
                k = i * 4 + j
                if k > nMax:
                    break
                idx = f * 16 + k
                if idx >= len(nList):
                    break                  
                n = nList[idx]                 
                df1 = distResultsDf[distResultsDf['Size'] == n]
                df1 = df1[columns]            
                df2 = df1.dropna(axis=0, how='any')    
                data = []
                for column in columns:
                    data.append(df2[column].values)                    
                bp = ax[i][j].boxplot(data, vert = 1)
                ax[i][j].title.set_text('# transactions = {}'.format(n+1))
                ax[i][j].set_xticklabels(['normal', 'log-normal', 'expon', 'gamma', 'Weibull'])    
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)
        
        title = 'Time to death distribution for N number of transactions part {}'.format(f+1)
        fig3.suptitle(title, fontsize=14)
        
        fileName = '{}{}'.format(title, '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close() 
        f += 1

# Box plots for Time to observed death distribution for number of transactions 3 .. 17     
    columns = ['observedNorm', 'observedLognorm', 'observedExpon', 'observedGamma', 'observedWeibull']
    
    f = 0
    idx = 0
    while idx < nMax and idx < len(nList):      
        fig3, ax = plt.subplots(4, 4, figsize=(21,15))
        
        for i in range(0, 4, 1):
            for j in range(0, 4, 1):
                k = i * 4 + j
                if k > nMax:
                    break
                idx = f * 16 + k
                if idx >= len(nList):
                    break                  
                n = nList[idx]
                df1 = distResultsDf[distResultsDf['Size'] == n]
                df1 = df1[columns]
            
                df2 = df1.dropna(axis=0, how='any')
    
                data = []
                for column in columns:
                    data.append(df2[column].values)
                    
                bp = ax[i][j].boxplot(data, vert = 1)
                ax[i][j].title.set_text('# transactions = {}'.format(n+1))
                ax[i][j].set_xticklabels(['normal', 'log-normal', 'expon', 'gamma', 'Weibull'])
    
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)    
        title = 'Time to observed death distribution for N number of transactions part {}'.format(f+1)
        fig3.suptitle(title, fontsize=14)
        
        fileName = '{}{}'.format(title, '.png')
        plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
        plt.close() 
        f += 1








