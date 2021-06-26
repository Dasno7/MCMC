# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:14:12 2021

@author: Sameer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

#BTC price data
btc_data = pd.read_csv("Crypto Data Repo/Bitcoin Historical Data.csv")
btc_price = np.flip(pd.Series(btc_data['Price'].str.replace(',','').astype(float)))
btc_price.index = np.flip(btc_data['Date'])
Y = np.log(btc_price/btc_price.shift(1))[1:]*np.sqrt(365) #return process Y
T = Y.shape[0] #T of process
format_str = "%b %d, %Y"
Y.index = [datetime.datetime.strptime(Y.index[j],format_str) for j in range(T)]

N      = T  # Number of observaions in each simulation (will use n-1 since Y1 = 0)
beta = 0.002505416013268381
alpha = 0.343820808290336
kappa = 1-beta
theta  = (alpha/kappa)# Part of the long term mean of volatility
s2V   = 0.3526566060798969/np.sqrt(365) # Diffusion parameter of volatility (sig_v)
rho    = -0.5348454274115587   # Brownin motion correlation
mJ     = 2.450566305166653/np.sqrt(365)   # Mean of price jump size
s2J    = 2.8667963646114725/np.sqrt(365)  # Variance of price jump size
lambd =0.014506061668120155/np.sqrt(365)  # Intensitiy of the pure jump process
mV     = 2.69634775159012/np.sqrt(365)  # Mean of volatility jump size
rhoJ   =  -0.15902310327822086     # Jumps correlation
mu     = 0.002505416013268381 # Drift parameter of Returns

#kappa  = 0.187  # Speed of volatility mean reversio
#theta  = 0.054  # Part of the long term mean of volatility
#s2V   = 0.0069  # Diffusion parameter of volatility (sig_v)
#rho    = 0.2748  # Brownin motion correlation
#mJ     = -0.049  # Mean of price jump size
#s2J    = 2.06  # Variance of price jump size
#lambd = 0.051  # Intensitiy of the pure jump process
#mV     = 0.709  # Mean of volatility jump size
#rhoJ   = -0.21  # Jumps correlation
#mu     = 0.0421  # Drift parameter of Returns

# Create empty vectors to store the simulated values
E    = np.zeros(N - 1)  # Residuals of standarized log return
v    = np.zeros(N)  # sqrt Volatility of log return
v[0] =  np.sqrt((0.1*(Y-np.mean(Y))**2+0.9*(np.var(Y)))[0])
Jv   = np.zeros(N)  # Jumps in volatility
Jy   = np.zeros(N) # Jumps in log return
Y[0] = np.log(btc_price/btc_price.shift(1))[1:][0]*np.sqrt(365)
S = np.zeros(N)
S[0] = btc_price[0]

btc_sim=np.zeros([int(1e3),T])
v_sim=np.zeros([int(1e3),T])
for iters in tqdm(range(int(1e3))): 
# Run the simulation T times and save the calculated values
    for i in range(1,N):
        J        = np.random.binomial(1,lambd)  # Poisson distributed random value with lambda = 0.051 for determining whether a jump exists
        XV       = np.random.exponential(1/mV)  # Exponential distributed random value with mV = 0.709 for jump size in volatility
        X        = np.random.normal((mJ + rhoJ * XV),s2J)  # Jump size of log return
        Jv[i]    = XV * J  # Jumps in volatilty (0 in case of no jump)
        Jy[i]    = X * J  # Jumps in log return (0 in case of no jump)
        
        Zy        = np.random.normal(0,1)  # Standard normal random value
        Z_interim       = np.random.normal(0,1)  # Standard normal random value
        Zv       = rho*Zy + np.sqrt(1-rho**2)*Z_interim
        
        #transformed volatility scheme
        v[i] = v[i-1]+0.5*(kappa*(theta/v[i-1]-v[i-1])+s2V/(4*v[i-1])+s2V**0.5*Zv)+(np.sqrt(v[i-1]**2+XV)-v[i-1])*J#(Jv[i]/(2*v[i-1]+1))
        Y[i] = mu + v[i-1]*Zy+Jy[i]
    
        S[i] = np.exp(Y[i]/np.sqrt(365))*S[i-1]
        #S[i] = np.exp(Y[i]/np.sqrt(365))*S[i-1]
 
    btc_sim[iters,:] = S
    v_sim[iters,:] = v/np.sqrt(365)        

#plot volatility jumps
pd.Series(np.mean(btc_sim,axis=0),index=btc_price.index[:-1].T).plot(figsize=(10,7),loglog=True)
pd.Series(btc_sim[np.where(btc_sim==max(np.max(btc_sim,axis=1)))[0][0],:],index=btc_price.index[:-1]).plot(figsize=(10,7),loglog=True)


plt.hist(np.log(btc_sim[:,-1]),bins=500)
plt.show()

#pd.Series(np.exp(Y),index=np.flip(btc_price.index[:-1].T)).plot(figsize=(10,7))