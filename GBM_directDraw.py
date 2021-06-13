# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:25:56 2021

@author: Dasno7
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import gamma
import datetime

#BTC price data
cryptos = ['Bitcoin','LINK','ETH','ADA']
Y={}
T={}
P = {}
for crypto in cryptos:
    data = pd.read_csv(str('Crypto Data Repo/'+crypto+' Historical Data.csv'))
    try:
        price = np.flip(pd.Series(data['Price'].str.replace(',','').astype(float)))
    except:
        price = np.flip(pd.Series(data['Price']))
         
    price.index = np.flip(data['Date'])
    P[crypto] = price
    Y[crypto] = np.log(price/price.shift(1))[1:]*np.sqrt(365) #return process Y
    T[crypto] = Y[crypto].shape[0] #T of process
    format_str = "%b %d, %Y"
    Y[crypto].index = [datetime.datetime.strptime(Y[crypto].index[j],format_str) for j in range(T[crypto])]



def get_hyperGamma(annualized_returns, x):
    Dayx_vol = np.zeros(annualized_returns.shape[0])    
    Day1_vol = np.zeros(annualized_returns.shape[0]) 
    for i in range(annualized_returns[x:].shape[0]):
        Dayx_vol[i] = np.std(annualized_returns[:(x-1+i)])
        Day1_vol[i] = Dayx_vol[i]/np.sqrt(x)
    mu_gamma = np.var(annualized_returns)
    var_gamma = np.var(Day1_vol)*np.sqrt(365)
    a= mu_gamma**2/var_gamma+2
    b= (a-1)*mu_gamma
    return a,b


sig2_save = {}
mu_save = {}

for crypto in cryptos:
    #Prior distribution hyperparameter values
    a=np.mean(Y[crypto]);A=np.var(Y[crypto]);
    f,F = get_hyperGamma(Y[crypto],30)
    
    
    #first draw sigma2
    f_star = T[crypto]/2+f
    F_star = F+0.5*(T[crypto]*np.var(Y[crypto])+(T[crypto]*(np.mean(Y[crypto])-a)**2)/(T[crypto]+1))
    sigma2 = F_star/(f_star+1)
    sig2_save[crypto]=sigma2
    
    #secondly draw mu
    a_star = (a+np.mean(Y[crypto])*T[crypto])/(T[crypto]+1)
    A_star = sigma2/(T[crypto]+1)
    mu_save[crypto] = a_star


print(mu_save,sig2_save)

    
