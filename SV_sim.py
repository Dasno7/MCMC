# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:25:07 2021

@author: Dasno7
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


# vol_of_vol = np.zeros(btc_price[30:].shape[0])
# for i in range(btc_price[30:].shape[0]):
#     vol_of_vol[i] = np.std(btc_price[:(29+i)])/np.sqrt(30)
    
# np.std(vol_of_vol)

# # Parameters from MCMC for Crix log return simulation
# N      = T  # Number of observaions in each simulation (will use n-1 since Y1 = 0)
# beta = 0.02216727394250727
# alpha = 0.12527700015283355
# kappa  = 1-beta
# theta  = alpha/kappa  # Part of the long term mean of volatility
# s2V   = 0.47506761593671976/np.sqrt(365)  # Diffusion parameter of volatility (sig_v)
# mu     = 0.02216727394250727  # Drift parameter of Returns
# rho  = -0.34924475308635844

# Parameters from MCMC for Crix log return simulation
N      = T  # Number of observaions in each simulation (will use n-1 since Y1 = 0)
beta = 0.009604357835473927
alpha =0.34272263111516055
kappa  = 1-beta
theta  = alpha/kappa  # Part of the long term mean of volatility
s2V   = 0.3503119669913358/np.sqrt(365)  # Diffusion parameter of volatility (sig_v)
mu     = 0.009604357835473927  # Drift parameter of Returns
rho  = -0.36881915759498374

# Create empty vectors to store the simulated values
V    = np.zeros(N)  # Volatility of log return
v    = np.zeros(N)  # sqrt Volatility of log return
Y    = np.zeros(N)  # Log return
S = np.zeros(N)
Y[0] = np.log(btc_price/btc_price.shift(1))[1:][0]*np.sqrt(365)
v[0] =  np.sqrt((0.1*(Y-np.mean(Y))**2+0.9*(np.var(Y)))[0])#0.355 #np.var(np.log(btc_price/btc_price.shift(1))[1:]) # Initial value of volatility = mean of volatilty
#v[0] = np.sqrt(V[0])
S[0] = btc_price[0]

btc_sim=np.zeros([int(1e3),T])
v_sim=np.zeros([int(1e3),T])
for iters in tqdm(range(int(1e3))): 
    # Run the simulation T times and save the calculated values
    for i in range(1,N):
        Zy        = np.random.normal(0,1)  # Standard normal random value
        Z_interim       = np.random.normal(0,1)  # Standard normal random value
        Zv       = rho*Zy + np.sqrt(1-rho**2)*Z_interim
          
                  
        #V[i]  = kappa * theta + (1 - kappa) * V[i - 1] + s2V**0.5* np.sqrt(max(V[i - 1],0))* Zv
        #Y[i]  =  Y[i-1]+ mu+ np.sqrt(max(V[i - 1],0)) * Zy
        
        #transformed volatility scheme
        v[i] = v[i-1]+0.5*(kappa*(theta/v[i-1]-v[i-1])+s2V/(4*v[i-1])+s2V**0.5*Zv)
        Y[i] = mu + v[i-1]*Zy
    
        S[i] = np.exp(Y[i]/np.sqrt(365))*S[i-1]
    btc_sim[iters,:] = S
    v_sim[iters,:] = v/np.sqrt(365)
 
#pd.Series(S,index=btc_price.index[:-1].T).plot(figsize=(10,7))
#btc_price.plot(figsize=(10,7))
#pd.Series(v_sim[867,:],index=btc_price.index[:-1].T).plot(figsize=(10,7))
#pd.Series(btc_sim[867,:],index=btc_price.index[:-1].T).plot(figsize=(10,7))

#pd.Series(np.mean(v_sim,axis=0),index=btc_price.index[:-1].T).plot(figsize=(10,7))
pd.Series(np.mean(btc_sim,axis=0),index=btc_price.index[:-1].T).plot(figsize=(10,7),loglog=True)
pd.Series(btc_sim[np.where(btc_sim==max(np.max(btc_sim,axis=1)))[0][0],:],index=btc_price.index[:-1]).plot(figsize=(10,7),loglog=True)

plt.hist(np.log(btc_sim[:,-1]),bins=500)
plt.show()

