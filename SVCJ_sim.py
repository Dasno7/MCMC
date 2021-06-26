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

#Final param results

mu ={'Bitcoin': 0.021644682489966826, 'LINK':0.03645939802343439, 'ETH':  0.015283256840018638, 'ADA':-0.007316876526135481 }#/np.sqrt(365)
s2V ={'Bitcoin': 0.06756870157684675, 'LINK':0.40446812600230947, 'ETH':  0.17931239389025172, 'ADA':0.26116042635375053 }#/np.sqrt(365)
alpha ={'Bitcoin': 0.043193377631366804, 'LINK': 0.3855999310971581, 'ETH':   0.16119100120638954, 'ADA':0.2662085060789083 }
beta ={'Bitcoin': 0.9263876734631863 , 'LINK':0.7873614865856806 , 'ETH':  0.8491499506560084, 'ADA':0.7995479771756661 }
rho ={'Bitcoin': -0.09119104913669938, 'LINK':-0.10825275611533464, 'ETH':  -0.12041305370319762, 'ADA':-0.09317621524732624  }
kappa ={'Bitcoin':(1-beta['Bitcoin']) , 'LINK':(1-beta['LINK']), 'ETH':(1- beta['ETH']), 'ADA': (1-beta['ADA'])}
theta ={'Bitcoin':  alpha['Bitcoin']/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': alpha['ETH']/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}
mJ = {'Bitcoin': 0.9406651072820391, 'LINK':3.098086827567313 , 'ETH':  2.520408386900445, 'ADA':2.068795155113863 }
s2J = {'Bitcoin': 8.42181066157726 , 'LINK':27.90186571275088, 'ETH':  13.099952280133733, 'ADA':17.140749486332407 }
lambd = {'Bitcoin': 0.0053092423361834245, 'LINK':0.0025708713027121845, 'ETH':  0.002263724926372133, 'ADA':0.002458274763338694 }
mV = {'Bitcoin': 2.30966833400468, 'LINK': 2.002095925724665, 'ETH':  2.192366611567432 , 'ADA':1.986970474497482 }
rhoJ = {'Bitcoin': -0.05277382444932916, 'LINK':-0.0850455673163998 , 'ETH':  -0.16674148085665919, 'ADA':-0.16623534607963417 }

N=T

#explode_corr = {'Bitcoin': 2, 'LINK':80, 'ETH':  50, 'ADA':20 }
explode_corr = {'Bitcoin': 3, 'LINK':3.85, 'ETH':  3.5, 'ADA':3 }
plot_explode = {'Bitcoin': 0, 'LINK':0 , 'ETH':  0, 'ADA':0 }

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
    Jv   = np.zeros(N[crypto]+1)  # Jumps in volatility
    Jy   = np.zeros(N[crypto]+1) # Jumps in log return
    Y[0] = Ydict[crypto][0]
    v[0] =  np.sqrt((0.1*(Ydict[crypto]-np.mean(Ydict[crypto]))**2+0.9*(np.var(Ydict[crypto])))[0]) # Initial value of volatility = mean of volatilty
    S[0] = P[crypto][0]
    
    sim=np.zeros([int(1),T[crypto]+1])
    v_sim=np.zeros([int(1),T[crypto]+1])
    jv_sim = np.zeros([int(1),T[crypto]+1])
    iters=0
    while iters <int(1): 
        # Run the simulation T times and save the calculated values
        interval = np.array(range(N[crypto]+1))/N[crypto] #tweak
        TT = N[crypto]/365
        dt = np.diff(interval*(TT))[0]
        a=(2*(2*rho[crypto]*s2V[crypto]**0.5-kappa[crypto])/s2V[crypto])**2
        i=1
        while i <N[crypto]+1:
            J        = np.random.binomial(1,lambd[crypto])  # Poisson distributed random value with lambda = 0.051 for determining whether a jump exists
            XV       = np.random.exponential(1/mV[crypto])  # Exponential distributed random value with mV = 0.709 for jump size in volatility
            X        = np.random.normal((mJ[crypto] + rhoJ[crypto] * XV),s2J[crypto])  # Jump size of log return
            Jv[i]    = XV * J  # Jumps in volatilty (0 in case of no jump)
            Jy[i]    = X * J  # Jumps in log return (0 in case of no jump)
            
            Zy        = np.random.normal(0,1)  # Standard normal random value
            Z_interim       = np.random.normal(0,1)  # Standard normal random value
            Zv       = rho[crypto]*Zy + np.sqrt(1-rho[crypto]**2)*Z_interim
              
                      
            #V[i]  = kappa * theta + (1 - kappa) * V[i - 1] + s2V**0.5* np.sqrt(max(V[i - 1],0))* Zv
            #Y[i]  =  Y[i-1]+ mu+ np.sqrt(max(V[i - 1],0)) * Zy
            
            #transformed volatility scheme
            v[i] = v[i-1]+0.5*(kappa[crypto]*(theta[crypto]/v[i-1]-v[i-1])+s2V[crypto]/(4*v[i-1])+s2V[crypto]**0.5*Zv)+(np.sqrt(v[i-1]**2+XV)-v[i-1])*J
            Y[i] = mu[crypto]*dt +np.sqrt(TT/N[crypto])*(v[i-1]*Zy)+Jy[i]*(TT/N[crypto])
            
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
                jv_sim[iters,:] = Jv
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
    
    #sns.histplot(pd.DataFrame(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],columns = [Ydict[crypto].index[[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]]]),ax=listAx2[crypto]\
    #            ,kde=True,stat="density",log_scale=True,color=sns.color_palette())
    #np.mean(v_sim,axis=0)[:-1]
    #v_sim[999,:][:-1]
    #pd.Series(v_sim[np.where(v_sim==max(np.max(v_sim,axis=1)))[0][0],:-1],index=Ydict[crypto].index, name ="Mean volatility "+crypto).plot(ax=   listAx2[crypto])       
    pd.DataFrame(v_sim[np.where(np.max(v_sim,axis=1)>=plot_explode[crypto])[0][:1],:-1].T,index=Ydict[crypto].index).plot(ax=   listAx2[crypto],colormap=sns.color_palette("flare", as_cmap=True),legend=False)       
 
        
    pd.DataFrame(jv_sim[np.where(np.max(v_sim,axis=1)>=plot_explode[crypto])[0][:1],:-1].T,index=Ydict[crypto].index).plot(ax=   listAx2[crypto],legend=False)       
    
    #pd.DataFrame(v_sim[np.where(np.min(v_sim,axis=1)<=plot_explode[crypto])[0][:2],:-1].T,index=Ydict[crypto].index).plot(ax=   listAx2[crypto])       
      
    listAx2[crypto].hlines(np.std(Ydict[crypto]),xmin=-0,xmax=Ydict[crypto].index.size,colors=sns.color_palette("flare"))
    #pd.Series(v_sim[np.where(v_sim==max(np.max(v_sim,axis=1)))[0][0],:-1],index=Ydict[crypto].index, name ="Mean volatility "+crypto).plot(ax=axis11)       
    
    #listAx2[crypto].vlines(np.mean(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],axis=0)\
    #                       ,ymin=0,ymax=0.4,colors=sns.color_palette())
    #sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]]
    listAx2[crypto].set(xlabel=None)
    listAx2[crypto].set(ylabel="Volatility")
    listAx2[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
fig2
    
    #pd.Series(S,index=btc_price.index[:-1].T).plot(figsize=(10,7))
    #btc_price.plot(figsize=(10,7))
    #pd.Series(v_sim[867,:],index=btc_price.index[:-1].T).plot(figsize=(10,7))
    #pd.Series(btc_sim[867,:],index=btc_price.index[:-1].T).plot(figsize=(10,7))
    
    #pd.Series(np.mean(v_sim,axis=0),index=btc_price.index[:-1].T).plot(figsize=(10,7))
# pd.Series(np.mean(v_sim,axis=0)[:-1],index=Ydict[crypto].index.T).plot(figsize=(10,7))
# pd.Series(v_sim[np.where(v_sim==min(np.max(v_sim,axis=1)))[0][0],:-1],index=Ydict[crypto].index).plot(figsize=(10,7))
    
    # plt.hist(np.log(btc_sim[:,-1]),bins=500)
    # plt.show()
    