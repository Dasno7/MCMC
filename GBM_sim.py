# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:18:15 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#BTC price data
#cryptos = ['Bitcoin','LINK','ETH','ADA']
cryptos=['LINK']
Y={}
T={}
P ={}
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

N      =T
m =0.06792776181895642#/np.sqrt(365)
sigma2_y = 1.8445376492537082#/np.sqrt(365)


def stock_path_sim(N,TT,S_0,mu,sigma2):
    interval = np.array(range(N+1))/N
    t = interval*TT
    W = np.sqrt(TT/N)*np.append(0,np.cumsum(np.random.normal(0,1,N)))
    #W = np.append(0,np.cumsum(np.random.normal(0,1,N)))
    S = S_0*np.exp((mu)*t+(sigma2)**0.5*W)
    return S

#def gen_paths(S0, mu, sigma, T, M, I):
#    dt = float(T) / M
#    paths = np.zeros((M + 1, I), np.float64)
#    paths[0] = S0
#    for t in range(1, M + 1):
#        rand = np.random.standard_normal(I)
#        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt +
#                                         sigma * np.sqrt(dt) * rand)
#    return paths
for crypto in Y.keys():
    sim=np.zeros([int(1e5),T[crypto]+1])
    for i in tqdm(range(int(1e5))):  
        sim[i,:] = stock_path_sim(T[crypto],T[crypto]/365,P[crypto][0],m,sigma2_y)
    #btc_sim = stock_path_sim(T,1,btc_price[-1],m,sigma2_y)
    #btc_sim = gen_paths(btc_price[-1],m,sigma2_y**0.5,0.01,T,1)    
    
    pd.Series(np.mean(sim,axis=0)[:-1],index=Y[crypto].index).plot(figsize=(10,7),loglog=True)
    pd.Series(sim[np.where(sim==max(np.max(sim,axis=1)))[0][0],:-1],index=Y[crypto].index).plot(loglog=True,figsize=(10,7))

#plt.plot(np.mean(btc_sim,axis=0))
#plt.show()
#print(btc_sim)