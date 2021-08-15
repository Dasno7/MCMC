# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:18:15 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns
import scipy.special as special
import scipy.stats as stats
from scipy.interpolate import griddata
from matplotlib import colors

#price data
#sns.set_theme(style='darkgrid')
#cryptos = ['Bitcoin','LINK','ETH','ADA']
cryptos=['Bitcoin']
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
    S = S_0*np.exp((mu-0.5*sigma2)*t+(sigma2)**0.5*W)
    return S


# #initialize RMSE and empirical forecasting density
# RMSE = {}
# densityEnd = {}

# cryptoNames = {'Bitcoin':'Bitcoin','LINK':'Chainlink','ETH':'Ethereum','ADA':'Cardano'}
# fig, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2,figsize=(12,7.5),constrained_layout = True)
# fig2, [[axis12, axis22],[axis32, axis42]] = plt.subplots(2,2,figsize=(12,7.5),constrained_layout = True)
# listAx = {'Bitcoin':axis1,'LINK':axis2,'ETH':axis3,'ADA':axis4}
# listAx2 = {'Bitcoin':axis12,'LINK':axis22,'ETH':axis32,'ADA':axis42}
# for crypto in Y.keys():
#     sim=np.zeros([int(1e5),T[crypto]+1])
#     for i in tqdm(range(int(1e5))):  
#         sim[i,:] = stock_path_sim(T[crypto],T[crypto]/365,P[crypto][0],m[crypto],sigma2_y[crypto])    
    
#     P[crypto].name='Real data'
#     pd.merge(
#     pd.merge(pd.Series(np.mean(sim,axis=0)[:-1],index=Y[crypto].index, name ="Average"),\
#     pd.Series(sim[np.where(sim==max(np.max(sim,axis=1)))[0][0],:-1]\
#               ,index=Y[crypto].index, name ="Maximum"),left_index=True,right_index=True),\
#           P[crypto],left_index=True,right_index=True).plot(figsize=(10,7),loglog=True,ax=listAx[crypto])
#     listAx[crypto].set(xlabel=None)
#     listAx[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
#     densityEnd[crypto] = (pd.DataFrame(sim[:,[-2,-1]],columns = [Y[crypto].index[[-2,-1]]]))
#     sns.histplot(pd.DataFrame(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],columns = [Y[crypto].index[[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]]]),ax=listAx2[crypto]\
#                   ,kde=True,stat="density",log_scale=True,color=sns.color_palette())
   
#     listAx2[crypto].vlines(np.mean(sim[:,[-int(np.round(T[crypto]/3)),-int(np.round(2*T[crypto]/3)),-1]],axis=0)\
#                             ,ymin=0,ymax=0.4,colors=sns.color_palette())
#     listAx2[crypto].set(xlabel=None)
#     listAx2[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
#     RMSE[crypto] =np.sqrt(np.mean((np.mean(sim,axis=0)-P[crypto])**2))
# print(RMSE)
# fig2


# def getParamForecastingDensity(densityDict,cryptos):
#     dictMean={}
#     dictVar={}
#     for crypto in cryptos:
#         score = np.log(np.array(densityEnd[crypto][densityEnd[crypto].columns[1][0]])/np.array(densityEnd[crypto][densityEnd[crypto].columns[0][0]]))
#         dictMean[crypto] =np.mean(score*365) 
#         dictVar[crypto] = np.var(score*np.sqrt(365))
        
#     return dictMean,dictVar

# dictMean,dictVar = getParamForecastingDensity(densityEnd,cryptos)
# #test = np.log(np.array(densityEnd['Bitcoin'][densityEnd['Bitcoin'].columns[1][0]])/np.array(densityEnd['Bitcoin'][densityEnd['Bitcoin'].columns[0][0]]))

#---------------------------------------------------------------------------------------------------------------------------
# Functions for Crude Monte Carlo

def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical

def stock_path_sim(N,TT,S_0,mu,sigma2):
    interval = np.array(range(N+1))/N
    t = interval*TT
    W = np.sqrt(TT/N)*np.append(0,np.cumsum(np.random.normal(0,1,N)))
    #W = np.append(0,np.cumsum(np.random.normal(0,1,N)))
    S = S_0*np.exp((mu-0.5*(sigma2))*t+(sigma2)**0.5*W)
    return S

def vega(S, K, T, r, sigma):
    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    
    vega = S  * np.sqrt(T) * stats.norm._pdf(d1)
    return vega

def black_scholes_call(S, K, T, r, sigma):

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * special.ndtr(d1) -  special.ndtr(d2)* K * np.exp(-r * T)
    return call
    
def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=1e6):
    sigma = 0.5

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = C - black_scholes_call(S, K, T, r, sigma)

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            if i==0:
                return np.nan
            else: return sigma

        ### use newton rapshon to update the estimate
        sigma = sigma + diff / vega(S, K, T, r, sigma)

    return np.nan

#---------------------------------------------------------------------------------------------------------------------------
# Crude Monte Carlo option pricing

# BTC scenario 1: 7 June 2021 S_0=35198

crypto = 'Bitcoin'
S=35198
K_BTC1 = np.array([42000, 37000, 33000, 39000, 40000, 30000, 34000, 36000, 32000,
        38000, 35000, 41000, 45000, 26000, 55000, 20000, 46000, 75000,
        80000, 44000, 48000, 85000, 60000, 56000, 25000, 58000, 50000,
        54000, 31000, 70000, 28000, 52000, 65000, 43000, 24000, 14000,
        16000, 18000, 12000, 64000, 15000, 8000, 68000, 10000, 22000,
        72000, 6000, 76000])
T_BTC1 = np.array([2, 11, 18, 4, 1, 109, 53, 207, 291, 81])

T[crypto]=np.max(T_BTC1)
sim=np.zeros([int(1e5),T[crypto]+1])
for i in tqdm(range(int(1e5))):  
    sim[i,:] =stock_path_sim(T[crypto],T[crypto]/365,S,m[crypto],sigma2_y[crypto])    

pricePathMC = pd.Series(np.mean(sim,axis=0), name ="Average")

i=0; optionDict_BTC1= {}
for K in K_BTC1:
    for t in T_BTC1:
        payOffCall = (sim[:,t]-K).clip(min=0)
        callPrice =np.mean(payOffCall)*np.exp(-m[crypto]*t/365)
        optionDict_BTC1[i]=[callPrice,t,K]
        i+=1
BTC1 = pd.DataFrame(optionDict_BTC1).T
BTC1.columns = ['Call price','T','K']

imp = [implied_volatility_call(BTC1.loc[x]['Call price'],S,BTC1.loc[x]['K'],BTC1.loc[x]['T']/365,m[crypto],max_iterations=int(1e4)) for x in BTC1.index]   
BTC1["IV"]=imp
BTC1["BS analytical"] =  [black_scholes_call(S, BTC1.loc[x]['K'], BTC1.loc[x]['T']/365, m[crypto], sigma2_y[crypto]) for x in BTC1.index]   
test=BTC1

# #BTC scenario 2: 14 March 2021 S_0=57663
# crypto='Bitcoin'
# S=57663
# K_BTC2 = np.array([38000, 32000, 40000, 53000, 36000, 42000, 34000, 66000, 48000,
#         62000, 68000, 44000, 80000, 84000, 58000, 46000, 50000, 70000,
#         64000, 56000, 59000, 54000, 60000, 78000, 55000, 52000, 74000,
#         57000, 76000, 72000, 51000, 63000, 61000, 65000, 67000, 88000,
#         4000, 24000, 14000, 26000, 28000, 16000, 120000, 10000, 18000,
#         140000, 12000, 100000, 7000, 8000, 90000, 85000, 9000, 20000,
#         15000, 22000, 17000, 45000, 75000, 6000, 13000, 82000, 30000,
#         92000, 86000, 5000, 11000])
# T_BTC2 = np.array([ 4, 19, 47, 6, 12, 26, 103, 5, 194, 292, 75])

# T[crypto]=np.max(T_BTC2)
# sim=np.zeros([int(1e5),T[crypto]+1])
# for i in tqdm(range(int(1e5))):  
#     sim[i,:]= stock_path_sim(T[crypto],T[crypto]/365,S,m[crypto],sigma2_y[crypto])  

# pricePathMC = pd.Series(np.mean(sim,axis=0), name ="Average")

# i=0; optionDict_BTC2= {}
# for K in K_BTC2:
#     for t in T_BTC2:
#         payOffCall = (sim[:,t]-K).clip(min=0)
#         callPrice =np.mean(payOffCall)
#         optionDict_BTC2[i]=[callPrice,t,K]
#         i+=1
# BTC2 = pd.DataFrame(optionDict_BTC2).T
# BTC2.columns = ['Call price','T','K']
    
# imp = [implied_volatility_call(BTC2.loc[x]['Call price'],S,BTC2.loc[x]['K'],BTC2.loc[x]['T']/365,m[crypto],max_iterations=int(1e4)) for x in BTC2.index]   
# BTC2["IV"]=imp

# #BTC scenario 3: 16 November 2020 S_0=16052

# crypto = 'Bitcoin'
# S=16053
# K_BTC3 = np.array([15250, 14250, 15750, 15125, 16125, 15500, 14500, 14625, 16625,
#         15375, 15000, 16375, 16500, 13875, 16000, 14125, 14750, 16750,
#         14375, 16250, 14875, 15875, 14000, 15625, 4000, 8000, 6000, 12750,
#         13000, 5000, 9000, 11000, 10000, 10500, 10750, 7000, 11500, 10250,
#         12000, 3000, 13500, 12250, 11750, 12500, 13750, 11250, 13250,
#         16875, 17000, 17250, 18000, 18500, 18250, 17750, 17500, 17375,
#         17125, 24000, 36000, 20000, 19500, 19750, 19000, 40000, 20750,
#         19250, 17625, 18750, 28000, 20500, 32000, 22000, 17875, 20250,
#         21000])
# T_BTC3 = np.array([130, 74, 11, 4, 39, 221,  18,2])

# T[crypto]=np.max(T_BTC3)
# sim=np.zeros([int(1e5),T[crypto]+1])
# for i in tqdm(range(int(1e5))):  
#     sim[i,:]= stock_path_sim(T[crypto],T[crypto]/365,S,m[crypto],sigma2_y[crypto])  

# pricePathMC = pd.Series(np.mean(sim,axis=0), name ="Average")

# i=0; optionDict_BTC3= {}
# for K in K_BTC3:
#     for t in T_BTC3:
#         payOffCall = (sim[:,t]-K).clip(min=0)
#         callPrice =np.mean(payOffCall)
#         optionDict_BTC3[i]=[callPrice,t,K]
#         i+=1
# BTC3 = pd.DataFrame(optionDict_BTC3).T
# BTC3.columns = ['Call price','T','K']
    
# imp = [implied_volatility_call(BTC3.loc[x]['Call price'],S,BTC3.loc[x]['K'],BTC3.loc[x]['T']/365,m[crypto],max_iterations=int(1e4)) for x in BTC3.index]   
# BTC3["IV"]=imp

#--------------------------------------------------------------------------------------------------------------------------------------
#Plot Functions

def make_surf(X,Y,Z):
    XX,YY = np.meshgrid(np.linspace(min(X),max(X),230),np.linspace(min(Y),max(Y),230))
    ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
    return XX,YY,ZZ 

def mesh_plotBTC(fig,ax,title,X,Y,Z):
    XX,YY,ZZ = make_surf(X,Y,Z)
    np.nan_to_num(ZZ[:,135:],copy=False,nan=0)
    ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=1000,limit_direction='both'))
    XX=XX[:,50:200];YY=YY[:,50:200];ZZ=ZZ[:,50:200]
    XX=XX[20:,:];YY=YY[20:,:];ZZ=ZZ[20:,:]   
    my_cmap = plt.get_cmap('plasma') 
    surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
    ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
    return ax

#--------------------------------------------------------------------------------------------------------------------------------------
#Plot BTC
fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

model='IV'

BTC1_plot= BTC1.where((BTC1['K']/35198) <=5).dropna()
axis1 = mesh_plotBTC(fig,axis1,"7 June 2021",BTC1_plot['K']/35198,BTC1_plot['T']/365,BTC1_plot[model])

# BTC2_plot= BTC2.where((BTC2['K']/57663) <=5).dropna()
# axis2 = mesh_plotBTC(fig,axis2,"14 March 2021",BTC2_plot['K']/57663,BTC2_plot['T']/365,BTC2_plot[model])

# BTC3_plot= BTC3.where((BTC3['K']/16053) <=5).dropna()
# axis3 = mesh_plotBTC(fig,axis3,"16 November 2020",BTC3_plot['K']/16053,BTC3_plot['T']/365,BTC3_plot[model])
