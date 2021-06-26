# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:00:33 2021

@author: Dasno7
"""
# print(m,sigma2_v,alpha,beta,rho,mV,mS,sigma2_s,rho_j,lamb)
# 0.002505416013268381 0.3526566060798969 0.343820808290336 0.002505416013268381 -0.5348454274115587 2.69634775159012 2.450566305166653 2.8667963646114725 -0.15902310327822086 0.014506061668120155

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#Final param results
#print(m,sigma2_v,alpha,beta,rho,mV,mS,sigma2_s,sigma2_v,rho_j,lamb)
# ]
# 0.03497079965610104 0.40649583389463223 0.37367411208660023 0.7958167385421553 -0.12095241548424931 18.996430289037885 17.348926363905395 11.528656249919617 0.40649583389463223 -0.08942473713484263 0.0023415514605686677


#BTC price data
btc_data = pd.read_csv("Crypto Data Repo/Bitcoin Historical Data.csv")
btc_price = np.flip(pd.Series(btc_data['Price'].str.replace(',','').astype(float)))
btc_price.index = np.flip(btc_data['Date'])
Y = np.log(btc_price/btc_price.shift(1))[1:]*np.sqrt(365) #return process Y
T = Y.shape[0] #T of process

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
beta = 0.002505416013268381
alpha = 0.343820808290336
kappa = 1-beta  
theta  = alpha/kappa# Part of the long term mean of volatility
s2V   = 0.3526566060798969 # Diffusion parameter of volatility (sig_v)
rho    = -0.5348454274115587   # Brownin motion correlation
mJ     = 2.450566305166653   # Mean of price jump size
s2J    = 2.8667963646114725  # Variance of price jump size
lambd =0.014506061668120155  # Intensitiy of the pure jump process
mV     = 2.69634775159012  # Mean of volatility jump size
rhoJ   =  -0.15902310327822086     # Jumps correlation
mu     = 0.002505416013268381  # Drift parameter of Returns

# Create empty vectors to store the simulated values
V    = np.zeros(N)  # Volatility of log return
v    = np.zeros(N)  # sqrt Volatility of log return
S = np.zeros(N)
Y[0] = np.log(btc_price/btc_price.shift(1))[1:][0]*np.sqrt(365)
V[0] =  np.sqrt((0.1*(Y-np.mean(Y))**2+0.9*(np.var(Y)))[0])#0.355 #np.var(np.log(btc_price/btc_price.shift(1))[1:]) # Initial value of volatility = mean of volatilty
v[0] = np.sqrt(V[0])
S[0] = btc_price[0]

btc_sim=np.zeros([int(1e4),T])
v_sim=np.zeros([int(1e4),T])
for iters in tqdm(range(int(1e4))): 
    # Run the simulation T times and save the calculated values
    for i in range(1,N):
        Zy        = np.random.normal(0,1)  # Standard normal random value
        Z_interim       = np.random.normal(0,1)  # Standard normal random value
        Zv       = rho*Zy + np.sqrt(1-rho**2)*Z_interim
        
          
                  
        #V[i]  = kappa * theta + (1 - kappa) * V[i - 1] + s2V**0.5* np.sqrt(max(V[i - 1],0))* Zv
        #Y[i]  =  Y[i-1]+ mu+ np.sqrt(max(V[i - 1],0)) * Zy
        
        #transformed volatility scheme
        v[i] = v[i-1]+0.5*(kappa*(theta/v[i-1]-v[i-1])+s2V/(4*v[i-1])+s2V**0.5*Zv)
        #v[i] = v[i-1]+0.5*kappa*((theta-s2V/(4*kappa))/v[i-1]-v[i-1])+0.5*s2V**0.5*Zv
        #print(v[i])
        Y[i] = mu + v[i-1]*Zy
        #print(np.exp(Y[i]))
    
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

