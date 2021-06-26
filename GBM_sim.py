# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:18:15 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns

#price data
cryptos = ['Bitcoin','LINK','ETH','ADA']
#cryptos=['Bitcoin']
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
    #format_str = "%b %d, %Y"
    #Y[crypto].index = [datetime.datetime.strptime(Y[crypto].index[j],format_str) for j in range(T[crypto])]


N      =T
m ={'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642, 'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}#/np.sqrt(365)
sigma2_y = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903, 'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}#/np.sqrt(365)


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

cryptoNames = {'Bitcoin':'Bitcoin','LINK':'Chainlink','ETH':'Ethereum','ADA':'Cardano'}
fig, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2,figsize=(12,7.5),constrained_layout = True)
fig2, [[axis12, axis22],[axis32, axis42]] = plt.subplots(2,2,figsize=(12,7.5),constrained_layout = True)
listAx = {'Bitcoin':axis1,'LINK':axis2,'ETH':axis3,'ADA':axis4}
listAx2 = {'Bitcoin':axis12,'LINK':axis22,'ETH':axis32,'ADA':axis42}
for crypto in Y.keys():
    sim=np.zeros([int(1e4),T[crypto]+1])
    for i in tqdm(range(int(1e4))):  
        sim[i,:] = stock_path_sim(T[crypto],T[crypto]/365,P[crypto][0],m[crypto],sigma2_y[crypto])    
    
    P[crypto].name='Real data'
    pd.merge(
    pd.merge(pd.Series(np.mean(sim,axis=0)[:-1],index=Y[crypto].index, name ="Average"),\
    pd.Series(sim[np.where(sim==max(np.max(sim,axis=1)))[0][0],:-1]\
              ,index=Y[crypto].index, name ="Maximum"),left_index=True,right_index=True),\
         P[crypto],left_index=True,right_index=True).plot(figsize=(10,7),loglog=True,ax=listAx[crypto])
    listAx[crypto].set(xlabel=None)
    listAx[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
    
    sns.histplot(pd.DataFrame(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],columns = [Y[crypto].index[[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]]]),ax=listAx2[crypto]\
                 ,kde=True,stat="density",log_scale=True,color=sns.color_palette())
   
    listAx2[crypto].vlines(np.mean(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],axis=0)\
                           ,ymin=0,ymax=0.4,colors=sns.color_palette())
    #sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]]
    listAx2[crypto].set(xlabel=None)
    listAx2[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
    
    
    #np.append(np.arange(0,T[crypto]+1,(T[crypto]+1)//10),T[crypto])
    #sns.set_theme(style="darkgrid")
    # f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1],'wspace':0})
    # g = pd.DataFrame(pd.Series(np.mean(sim,axis=0)[:-1],index=Y[crypto].index)).reset_index().reset_index()
    # g1 = pd.DataFrame(pd.Series(sim[np.where(sim==max(np.max(sim,axis=1)))[0][0],:-1],index=Y[crypto].index)).reset_index().reset_index()
    # sns.lineplot(data=g,x='index',y=0,ax=a0)
    # sns.lineplot(data=g1,x='index',y=0,ax=a0)   
    # a0.set(yscale='log')#,xscale='log')
    # a0.tick_params(axis='x',labelrotation=30)
    # a0.set(xlabel=None)
    # sns.kdeplot(sim[:,-1],ax=a1,vertical=True,fill=True)
    # a1.set(xticklabels=[])
    # a1.set(xlabel=None)
    # a1.set(yticklabels=[])
    # a1.set(ylabel=None)
    # a1.grid(False)
    # a1.set_facecolor('White')
    # sns.despine(ax=a0)
    # sns.despine(ax=a1,left=True)#,bottom=True)
    # f.tight_layout()
fig2
    #np.mean(sim[:,-1],axis=0)
#plt.plot(np.mean(btc_sim,axis=0))
#plt.show()
#print(btc_sim)