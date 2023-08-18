
# Defining churn 
# 2 plots: PDF and CDF for 6 distributiion for one sample


import matplotlib.pyplot as plt
import os
import numpy as np
from Distributions import Statistics
from datetime import date
from datetime import timedelta


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":

    
    dataName = 'CDNOW' # 'CDNOW' casa_2 plex_45

    workDir = os.path.dirname(os.path.realpath('__file__'))
    dataDir = os.path.join(workDir, 'data')
    dataSubDir = os.path.join(dataDir, dataName)
    resultsDir = os.path.join(workDir, 'results')
    resultsSubDir = os.path.join(resultsDir, dataName)   
    
    plt.rcParams.update({'font.size': 14})

    pChurn = 0.5
    pLeft = 0.98
    # select some of them to plot
    distNameSelList = ['Normal', 'Lognormal', 'Fisk', 
        'Weibull', 'Exponential', 'Gamma']
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y', 'o']

    y = np.array([14, 21, 63, 49, 21, 35, 56, 84, 21, 50])
    
    dFirstBuy = date(2000, 1, 1)
    
    cumSum = 0
    dList = []
    dList.append(dFirstBuy)
    for i in y:
        cumSum += i
        dNext = dFirstBuy + timedelta(days=int(cumSum))
        dList.append(dNext)
        
    s = ', '.join([str(i) for i in dList])
    
    for i in dList:
        print(i)
    
    str(dNext)
    
    x = np.linspace(0, max(y)*2, max(y)*2+1)

    stats1 = Statistics(y, testStats='cvm', criterion='pvalue')
    stats2 = Statistics(y, testStats='ks', criterion='pvalue')

    stats = stats1

    distNameList = []
    pdfList = []
    cdfList = []
    medianList = []
    ppfList = []
    for sDistribution, modelDict in stats.modelsDict.items():
        if sDistribution not in distNameSelList:
            continue
        distNameList.append(sDistribution)
        model = modelDict['model']
        pdfList.append(list(model.pdf(x)))
        cdfList.append(list(model.cdf(x)))
        medianList.append(round(model.median(), 2))
        ppfList.append(round(model.ppf(pLeft), 2))
        
        
    

#   plot PDF
    fig, ax = plt.subplots(1, 1, figsize=(21,15))  
    
    for i, pdf in enumerate(pdfList):
        ax.plot(x, pdf, '{}-'.format(colors[i]), lw=1, alpha=0.6, label=distNameList[i])
        median = medianList[i]
        ax.axvline(x=median, c=colors[i], linestyle=':', label='median={}'.format(median))
        q098 = ppfList[i]
        ax.axvline(x=q098, c=colors[i], linestyle='-', label='Q({})={}'.format(pLeft, q098))

    ax.legend()
    
    title = 'PDF of distributions. Best CvM - {}. Best KS - {}'.format(stats1.bestModel.sName, stats2.bestModel.sName)
    
    ax.title.set_text(title)
    ax.set_xlabel('Interval between purchases. Sample: {}'.format(list(y)))
    ax.set_ylabel('Probability that event occurs')
        
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    
    
#   PLOT CDF
    fig, ax = plt.subplots(1, 1, figsize=(21,15))  
    
    for i, cdf in enumerate(cdfList):
        ax.plot(x, cdf, '{}-'.format(colors[i]), lw=1, alpha=0.6, label=distNameList[i])
        median = medianList[i]        
        ax.axvline(x=median, c=colors[i], linestyle=':', label='median={}'.format(median))
        q098 = ppfList[i]        
        ax.axvline(x=q098, c=colors[i], linestyle='-', label='Q({})={}'.format(pLeft, q098))

    ax.legend()
    
    title = 'CDF of distributions. Best CvM - {}. Best KS - {}'.format(stats1.bestModel.sName, stats2.bestModel.sName)
    
    ax.title.set_text(title)
    ax.set_xlabel('Interval between purchases. Sample: {}'.format(list(y)))
    ax.set_ylabel('Probability that event occurs')
        
    fileName = '{}{}'.format(title, '.png')
    plt.savefig(os.path.join(resultsSubDir, fileName), bbox_inches='tight')
    plt.close() 
    
    
    

    # print parameters of distributions
    for sModel, model in stats1.modelsDict.items():
        print('{}:'.format(sModel))
        for param, value in model['model'].namedParams.items():
            print('{} = {}'.format(param, round(value, 4)))

            

    
#     2000-01-01
# 2000-01-15
# 2000-02-05
# 2000-04-08
# 2000-05-27
# 2000-06-17
# 2000-07-22
# 2000-09-16
# 2000-12-09
# 2000-12-30
# 2001-02-18
# Normal:
# mean = 41.4
# std = 21.6019
# Lognormal:
# stdLog = 0.565
# mean = 35.6273
# meanLog = 3.5731
# Fisk:
# shape = 2.8731
# median = 36.2365
# Exponential:
# scale = 41.4
# rate = 0.0242
# Gamma:
# shape = 3.4874
# scale = 11.8712
# rate = 0.0842
# Weibull:
# shape = 2.0546
# scale = 46.9612
# Gumbel:
# GeneralizedGamma:

    
    
    """
    Normal : good
    Weibull : good
    Gamma : s - same, loc=0, scale = 1 / rate (R), rate = 1 / scale
    Expon : loc=0, scale = 1 / rate (R), rate = 1 / scale
    Lognorm : s = sdlog(R) [same], loc=0, scale = exp(meanlog(R)), meanlog = ln(scale)
    Fisk: s = shape, scale == median
    """
    
    
#     y = np.array([14, 21, 63])
#     stats = Statistics(y, testStats='cvm', criterion='pvalue')
#     stats.modelsDict['Fisk']['model'].params
#     stats.modelsDict['Fisk']['model'].median()
#     stats.modelsDict['Fisk']['model'].ppf(0.95)



#     date(1997, 4, 8) + timedelta(days=78)

#     date(1997, 4, 8) - date(1997, 2, 5)  + timedelta(days=78)

# date(1997, 4, 19) - date(1997, 2, 5)


# 1997-02-05


# 14, 21, 63


# from State import calcC


# calcC([14, 21, 63], span=None)


# np.median([10, 15, 7, 5])



