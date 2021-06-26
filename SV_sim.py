# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:25:07 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

#price data
cryptos = ['Bitcoin','LINK','ETH','ADA']
#cryptos=['Bitcoin']
Ydict={}
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
    Ydict[crypto] = (np.log(price/price.shift(1))[1:]*np.sqrt(365)) #return process Y
    T[crypto] = Ydict[crypto].shape[0] #T of process
    #format_str = "%b %d, %Y"
    #Y[crypto].index = [datetime.datetime.strptime(Y[crypto].index[j],format_str) for j in range(T[crypto])]


# FINAL PARAMETERS
#-0.0036297164211186225 0.13684063167039534 0.15524133109243154 0.8750995884741214 -0.05028527322124073

mu ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }#/np.sqrt(365)
s2V = {'Bitcoin': 0.04033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012, 'ADA':0.13299513034831426 }#/np.sqrt(365)
alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
kappa ={'Bitcoin':(1-0.9578989881256078) , 'LINK':(1-beta['LINK']), 'ETH':(1- 0.9114176612486856), 'ADA': (1-beta['ADA'])}
theta ={'Bitcoin':  0.027300605314312377/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': 0.0927768090812799/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}

N=T

#explode_corr = {'Bitcoin': 2, 'LINK':80, 'ETH':  50, 'ADA':20 }
explode_corr = {'Bitcoin': 3, 'LINK':3.85, 'ETH':  3.5, 'ADA':3 }
plot_explode = {'Bitcoin': 0, 'LINK':0 , 'ETH':  0, 'ADA':0 }

#initialize RMSE
RMSE = {}

# Create empty vectors to store the simulated values
cryptoNames = {'Bitcoin':'Bitcoin','LINK':'Chainlink','ETH':'Ethereum','ADA':'Cardano'}
fig, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2,figsize=(12,7.5),constrained_layout = True)
fig2,  [axis12, axis22,axis32, axis42] = plt.subplots(4,1,figsize=(12,15),constrained_layout = True)
listAx = {'Bitcoin':axis1,'LINK':axis2,'ETH':axis3,'ADA':axis4}
listAx2 = {'Bitcoin':axis12,'LINK':axis22,'ETH':axis32,'ADA':axis42}
for crypto in Ydict.keys():
    V    = np.zeros(N[crypto]+1)  # Volatility of log return
    v    = np.zeros(N[crypto]+1)  # sqrt Volatility of log return
    Y    = np.zeros(N[crypto]+1)  # Log return
    S = np.zeros(N[crypto]+1)
    Y[0] = Ydict[crypto][0]
    v[0] =  np.sqrt((0.1*(Ydict[crypto]-np.mean(Ydict[crypto]))**2+0.9*(np.var(Ydict[crypto])))[0]) # Initial value of volatility = mean of volatilty
    S[0] = P[crypto][0]
    
    sim=np.zeros([int(1e5),T[crypto]+1])
    v_sim=np.zeros([int(1e5),T[crypto]+1])
    iters=0
    while iters <int(1e5): 
        # Run the simulation T times and save the calculated values
        interval = np.array(range(N[crypto]+1))/N[crypto] #tweak
        TT = N[crypto]/365
        dt = np.diff(interval*(TT))[0]
        a=(2*(2*rho[crypto]*s2V[crypto]**0.5-kappa[crypto])/s2V[crypto])**2
        i=1
        while i <N[crypto]+1:
            Zy        = np.random.normal(0,1)  # Standard normal random value
            Z_interim       = np.random.normal(0,1)  # Standard normal random value
            Zv       = rho[crypto]*Zy + np.sqrt(1-rho[crypto]**2)*Z_interim
              
                      
            #V[i]  = kappa * theta + (1 - kappa) * V[i - 1] + s2V**0.5* np.sqrt(max(V[i - 1],0))* Zv
            #Y[i]  =  Y[i-1]+ mu+ np.sqrt(max(V[i - 1],0)) * Zy
            
            #transformed volatility scheme
            v[i] = v[i-1]+0.5*(kappa[crypto]*(theta[crypto]/v[i-1]-v[i-1])+s2V[crypto]/(4*v[i-1])+s2V[crypto]**0.5*Zv)
            Y[i] = mu[crypto]*dt +np.sqrt(TT/N[crypto])* v[i-1]*Zy
            
            S[i] = np.exp(Y[i])*S[i-1]
            
            #if a+explode_corr[crypto]< abs(4*v[i-1]):
            if explode_corr[crypto]< abs(v[i-1]):
                print('out' +str(v[i-1])+"  "+str(i)+"   "+crypto)
                break
            else:
                i+=1
        if i ==N[crypto]+1:
                sim[iters,:] = S
                v_sim[iters,:] = v
                if iters%1000==0: print(crypto+":"+str(iters))
                iters+=1
        else:
                continue
    
    P[crypto].name='Real data'  
    pd.merge(
    pd.merge(pd.Series(np.mean(sim,axis=0)[:-1],index=Ydict[crypto].index, name ="Average"),\
    pd.Series(sim[np.where(sim==max(np.max(sim,axis=1)))[0][0],:-1]\
              ,index=Ydict[crypto].index, name ="Maximum"),left_index=True,right_index=True),\
         P[crypto],left_index=True,right_index=True).plot(figsize=(10,7),loglog=True,ax=listAx[crypto])
    listAx[crypto].set(xlabel=None)
    listAx[crypto].set_ylabel('Dollar price',rotation=90)
    listAx[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
    
    #RMSE
    RMSE[crypto] =np.sqrt(np.mean((np.mean(sim,axis=0)-P[crypto])**2))
 
    pd.DataFrame(v_sim[np.where(np.max(v_sim,axis=1)>=plot_explode[crypto])[0][:5],:-1].T,index=Ydict[crypto].index).plot(ax=   listAx2[crypto],colormap=sns.color_palette("flare", as_cmap=True),legend=False)       

    listAx2[crypto].hlines(np.std(Ydict[crypto]),xmin=-0,xmax=Ydict[crypto].index.size)
    listAx2[crypto].set(xlabel=None)
    listAx2[crypto].set(ylabel="Volatility")
    listAx2[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
fig2
