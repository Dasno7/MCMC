# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:23:23 2021

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
btc_data = pd.read_csv("Crypto Data Repo/Bitcoin Historical Data.csv")
btc_price = np.flip(pd.Series(btc_data['Price'].str.replace(',','').astype(float)))
btc_price.index = np.flip(btc_data['Date'])
Y = np.log(btc_price/btc_price.shift(1))[1000:]*np.sqrt(365) #return process Y
#Y = ((btc_price-btc_price.shift(1))/btc_price.shift(1))[1:]
T = Y.shape[0] #T of process
format_str = "%b %d, %Y"
Y.index = [datetime.datetime.strptime(Y.index[j],format_str) for j in range(T)]


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

#Prior distribution hyperparameter values
a=np.mean(Y);A=np.var(Y);
f,F = get_hyperGamma(Y,30)

#starting values parameters
sigma2_y =F/(f-2)

#Initialize saving parameters for MC
mtot=0;mtot2=0 #drawing mu
sigma2_ytot=0;sigma2_ytot2=0 #drawing sigma2_y

N=100000;burn=3000 #Number of draws and burn period
sig2Y_save = np.zeros(N)
mu_save = np.zeros(N)
for i in tqdm(range(N)):
    #Draw mu (derive A_star)
    A_star = 1/(T/sigma2_y+1/A)
    a_star = A_star*(np.sum(Y)/sigma2_y+a/A)
    m=np.random.normal(a_star,A_star**0.5)
    
    if i>burn:
        mtot += m
        mtot2 =+ m**2
    mu_save[i]=m
    
    #Draw sigma2_y
    # f_star = f+T
    # F_star = (F*f+np.sum((Y-m)**2))/f_star
    f_star = T +f
    F_star = F+np.sum((Y-m)**2)
    sigma2_y = stats.invgamma.rvs(f_star, scale=F_star)
    if i > burn:
        sigma2_ytot += sigma2_y
        sigma2_ytot2 += sigma2_y**2
    sig2Y_save[i]=sigma2_y
    
#Monte carlo estimates
m=mtot/(N-burn)        
sigma2_y=sigma2_ytot/(N-burn)
print(m,sigma2_y)
        
plt.plot(sig2Y_save)
plt.show()
        

# def inv_gamma(x,a,b):
#     y = (1/x)**(a+1)*np.exp(-b/x)*(b**a/gamma(a))
#     #b/(a-1)
#     #b**2/((a-1)**2*(a-2))
#     return (y)
    
# x=np.arange(0.01,10,0.01)
# plt.plot(x,inv_gamma(x,40,110))
# plt.show()
    
