# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:02:25 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns
import statsmodels.tsa.stattools as ts

#price data
cryptos = ['Bitcoin','LINK','ETH','ADA']
#cryptos=['Bitcoin']
Y={}
T={}
P ={}
for crypto in cryptos:
    data = pd.read_csv(str('Crypto Data Repo/'+crypto+' Historical Data_outSample.csv'))
    try:
        price = np.flip(pd.Series(data['Price'].str.replace(',','').astype(float)))
    except:
        price = np.flip(pd.Series(data['Price']))
         
    price.index = np.flip(data['Date'])
    P[crypto] = price
    Y[crypto] = np.log(price/price.shift(1))[1:]*np.sqrt(365) #return process Y
    T[crypto] = Y[crypto].shape[0] #T of process
    #format_str = "%b %d, %Y"
    #Y[crypto].index = [datetime.datetime.strptime(Y[crypto].index[j],format_str) for j in range(T[crypto])]

def censoredLikScore(y,mu,sigma2,r):
    logfhat = np.log(stats.norm.pdf(y,loc=mu,scale=sigma2))
    wlogfhat = stats.norm.cdf(r,loc=mu,scale=sigma2)
    
    w = np.zeros(logfhat.shape[0])
    w[np.where(y <= r)[0]]=1 # or len np.where
    
    score = w*logfhat+(1-w)*np.log((1-wlogfhat))

    return score

# GBM
m ={'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642, 'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}#/np.sqrt(365)
sigma2_y = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903, 'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}#/np.sqrt(365)
r = {'Bitcoin': -1, 'LINK': -2, 'ETH': -1.5, 'ADA': -1.7}#/np.sqrt(365)
#r = {'Bitcoin': 1, 'LINK': 2, 'ETH': 1.5, 'ADA': 1.7}#/np.sqrt(365)

scoreGBM={}
scoreMeanGBM={}
for crypto in cryptos:
    scoreGBM[crypto] = censoredLikScore(Y[crypto],m[crypto],sigma2_y[crypto],r[crypto])
    scoreMeanGBM[crypto] = np.mean(scoreGBM[crypto])
    
# SV
m = {'Bitcoin': -0.032996832287487184, 'LINK': 0.040777862720945406, 'ETH': 0.13105496907506411, 'ADA': 0.019951308554821443}
sigma2_y = {'Bitcoin': 1.1353596954308005, 'LINK': 2.5180595490744087, 'ETH': 1.6219566894267081, 'ADA': 1.8105371646411286}

scoreSV={}
scoreMeanSV={}
for crypto in cryptos:
    scoreSV[crypto] = censoredLikScore(Y[crypto],m[crypto],sigma2_y[crypto],r[crypto])
    scoreMeanSV[crypto] = np.mean(scoreSV[crypto])
    
# GBMJ
m ={'Bitcoin': 0.33723856804279545, 'LINK': 0.07350790244795152, 'ETH': -0.13193574849128004, 'ADA': -0.342408545740982}
sigma2_y = {'Bitcoin': 0.6969039835192633, 'LINK': 1.7246845080166926, 'ETH': 1.1282169361226804, 'ADA': 1.3454950393943221}
r = {'Bitcoin': -1, 'LINK': -2, 'ETH': -2, 'ADA': -3}#/np.sqrt(365)

scoreGBMJ={}
scoreMeanGBMJ={}
for crypto in cryptos:
    scoreGBMJ[crypto] = censoredLikScore(Y[crypto],m[crypto],sigma2_y[crypto],r[crypto])
    scoreMeanGBMJ[crypto] = np.mean(scoreGBMJ[crypto])    
    
# SVCJ
m= {'Bitcoin': -0.018366030170317075, 'LINK': -0.07877636392697288, 'ETH': 0.047659214745455696, 'ADA': 0.07799008884514173} 
sigma2_y = {'Bitcoin': 0.8611939710809794, 'LINK': 2.8324609618616545, 'ETH': 1.7077178897667367, 'ADA': 2.0180350842515793}

scoreSVCJ={}
scoreMeanSVCJ={}
for crypto in cryptos:
    scoreSVCJ[crypto] = censoredLikScore(Y[crypto],m[crypto],sigma2_y[crypto],r[crypto])
    scoreMeanSVCJ[crypto] = np.mean(scoreSVCJ[crypto])
    
    
# Testing


def predictAbilityTest(scoreVec1,scoreVec2,n,m):
    d = scoreVec1-scoreVec2
    d_bar = np.sum(d)/n
    K = np.floor(n**(0.25))
    a = 1-np.arange(1,K)/K
    gamma_hat = ts.acovf(d,nlag=K-1,fft=True)
    sigma2_mn = np.abs(gamma_hat[0]**2+2*np.sum(a*gamma_hat[1:]))
    t_mn = d_bar/np.sqrt(sigma2_mn/n)
    
    alpha=np.array([0.1,0.05,0.01])
    sign = ['*','**','***']
    checksign = np.where(np.abs(t_mn)>stats.norm.ppf(1-alpha))[0]
    if checksign.size==0:
        signLevel = ''
    else:
        signLevel= sign[max(checksign)]
    return str(t_mn)+signLevel

cryptos = ['Bitcoin','LINK','ETH','ADA']
crypto='Bitcoin'
n =  {'Bitcoin': 1369, 'LINK': 713, 'ETH': 1381, 'ADA': 1130}
m=Y[crypto].size

# GBM
gbmTest = []
gbmTest.append(0)
gbmTest.append(predictAbilityTest(scoreGBM[crypto],scoreSV[crypto],n[crypto],m))
gbmTest.append(predictAbilityTest(scoreGBM[crypto],scoreGBMJ[crypto],n[crypto],m))
gbmTest.append(predictAbilityTest(scoreGBM[crypto],scoreSVCJ[crypto],n[crypto],m))

# SVCJ
svTest = []
svTest.append(predictAbilityTest(scoreSV[crypto],scoreGBM[crypto],n[crypto],m))
svTest.append(0)
svTest.append(predictAbilityTest(scoreSV[crypto],scoreGBMJ[crypto],n[crypto],m))
svTest.append(predictAbilityTest(scoreSV[crypto],scoreSVCJ[crypto],n[crypto],m))

# GBMJ
gbmjTest = []
gbmjTest.append(predictAbilityTest(scoreGBMJ[crypto],scoreGBM[crypto],n[crypto],m))
gbmjTest.append(predictAbilityTest(scoreGBMJ[crypto],scoreSV[crypto],n[crypto],m))
gbmjTest.append(0)
gbmjTest.append(predictAbilityTest(scoreGBMJ[crypto],scoreSVCJ[crypto],n[crypto],m))

# SVCJ
svcjTest = []
svcjTest.append(predictAbilityTest(scoreSVCJ[crypto],scoreGBM[crypto],n[crypto],m))
svcjTest.append(predictAbilityTest(scoreSVCJ[crypto],scoreSV[crypto],n[crypto],m))
svcjTest.append(predictAbilityTest(scoreSVCJ[crypto],scoreGBMJ[crypto],n[crypto],m))
svcjTest.append(0)

test = pd.DataFrame([gbmTest,svTest,gbmjTest,svcjTest]).T