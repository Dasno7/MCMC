# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:05:21 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import interpolate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.optimize import curve_fit
import scipy.stats as stats

"""Pricing data for 14-03-2021""" 
priceData120321 = pd.read_csv("Crypto Data Repo/Option Market Data/2021_03_12/price_data_2021_03_12.csv")
startDate = datetime.strptime('2021-03-14','%Y-%m-%d')
## ETH
priceData120321ETH = priceData120321.loc[(priceData120321['instrument'].str.startswith('deribit::ETH')&priceData120321['creation_time'].str.startswith('2021-03'))]
locationList = []
for instrument in priceData120321ETH['instrument'].unique():
    locationList.append(np.where(priceData120321ETH['instrument']==instrument)[0][-1])
    #locationList.append(priceData120321ETH['instrument'].where(priceData120321ETH['instrument']==instrument).last_valid_index())
locationList.sort()
priceData120321ETH = priceData120321ETH.iloc[locationList]
firstDashETH = priceData120321ETH['instrument'].str.find('-')
middleDashETH = priceData120321ETH['instrument'].str.find('-',12+1)
lastDashETH = priceData120321ETH['instrument'].str.rfind('-')

timeToMatETH = priceData120321ETH['instrument'].copy()
timeToMatETH.name = "TimeToMaturity"
strikeETH = priceData120321ETH['instrument'].copy()
strikeETH.name = "Strike"
optionType = priceData120321ETH['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData120321ETH.index:
    if middleDashETH.loc[quote] != -1:
        timeToMatETH.loc[quote] = (datetime.strptime(priceData120321ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
        strikeETH.loc[quote] = int(priceData120321ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
        optionType[quote] = priceData120321ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
    else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData120321ETH[timeToMatETH.name]=timeToMatETH.copy()
priceData120321ETH[strikeETH.name]=strikeETH.copy()
priceData120321ETH[optionType.name]=optionType.copy()
priceData120321ETH.dropna(inplace=True)


## BTC 
priceData120321BTC = priceData120321.loc[(priceData120321['instrument'].str.startswith('deribit::BTC')&priceData120321['creation_time'].str.startswith('2021-03'))]
locationList = []
for instrument in priceData120321BTC['instrument'].unique():
    locationList.append(np.where(priceData120321BTC['instrument']==instrument)[0][-1])
locationList.sort()
priceData120321BTC = priceData120321BTC.iloc[locationList]
firstDashBTC = priceData120321BTC['instrument'].str.find('-')
middleDashBTC = priceData120321BTC['instrument'].str.find('-',12+1)
lastDashBTC = priceData120321BTC['instrument'].str.rfind('-')

timeToMatBTC = priceData120321BTC['instrument'].copy()
timeToMatBTC.name = "TimeToMaturity"
strikeBTC = priceData120321BTC['instrument'].copy()
strikeBTC.name = "Strike"
optionType = priceData120321BTC['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData120321BTC.index:
    if middleDashBTC.loc[quote] != -1:
        timeToMatBTC.loc[quote] = (datetime.strptime(priceData120321BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
        strikeBTC.loc[quote] = int(priceData120321BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
        optionType[quote] = priceData120321BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
    else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData120321BTC[timeToMatBTC.name]=timeToMatBTC.copy()
priceData120321BTC[strikeBTC.name]=strikeBTC.copy()
priceData120321BTC[optionType.name]=optionType.copy()
priceData120321BTC.dropna(inplace=True)


"""Pricing data for 07-06-2021""" 
priceData010621 = pd.read_csv("Crypto Data Repo/Option Market Data/2021_06_01/price_data_2021_06_01.csv")
startDate = datetime.strptime('2021-06-07','%Y-%m-%d')
## ETH
priceData010621ETH = priceData010621.loc[(priceData010621['instrument'].str.startswith('deribit::ETH')&priceData010621['creation_time'].str.startswith('2021-06'))]
locationList = []
for instrument in priceData010621ETH['instrument'].unique():
    locationList.append(np.where(priceData010621ETH['instrument']==instrument)[0][-1])
locationList.sort()
priceData010621ETH = priceData010621ETH.iloc[locationList]
firstDashETH = priceData010621ETH['instrument'].str.find('-')
middleDashETH = priceData010621ETH['instrument'].str.find('-',12+1)
lastDashETH = priceData010621ETH['instrument'].str.rfind('-')

timeToMatETH = priceData010621ETH['instrument'].copy()
timeToMatETH.name = "TimeToMaturity"
strikeETH = priceData010621ETH['instrument'].copy()
strikeETH.name = "Strike"
optionType = priceData010621ETH['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData010621ETH.index:
    if middleDashETH.loc[quote] != -1:
        timeToMatETH.loc[quote] = (datetime.strptime(priceData010621ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
        strikeETH.loc[quote] = int(priceData010621ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
        optionType[quote] = priceData010621ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
    else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData010621ETH[timeToMatETH.name]=timeToMatETH.copy()
priceData010621ETH[strikeETH.name]=strikeETH.copy()
priceData010621ETH[optionType.name]=optionType.copy()
priceData010621ETH.dropna(inplace=True)


## BTC 
priceData010621BTC = priceData010621.loc[(priceData010621['instrument'].str.startswith('deribit::BTC')&priceData010621['creation_time'].str.startswith('2021-06'))]
locationList = []
for instrument in priceData010621BTC['instrument'].unique():
    locationList.append(np.where(priceData010621BTC['instrument']==instrument)[0][-1])
locationList.sort()
priceData010621BTC = priceData010621BTC.iloc[locationList]
firstDashBTC = priceData010621BTC['instrument'].str.find('-')
middleDashBTC = priceData010621BTC['instrument'].str.find('-',12+1)
lastDashBTC = priceData010621BTC['instrument'].str.rfind('-')

timeToMatBTC = priceData010621BTC['instrument'].copy()
timeToMatBTC.name = "TimeToMaturity"
strikeBTC = priceData010621BTC['instrument'].copy()
strikeBTC.name = "Strike"
optionType = priceData010621BTC['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData010621BTC.index:
    if middleDashBTC.loc[quote] != -1:
        timeToMatBTC.loc[quote] = (datetime.strptime(priceData010621BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
        strikeBTC.loc[quote] = int(priceData010621BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
        optionType[quote] = priceData010621BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
    else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData010621BTC[timeToMatBTC.name]=timeToMatBTC.copy()
priceData010621BTC[strikeBTC.name]=strikeBTC.copy()
priceData010621BTC[optionType.name]=optionType.copy()
priceData010621BTC.dropna(inplace=True)

"""Pricing data for 16-11-2020""" 
priceData161120 = pd.read_csv("Crypto Data Repo/Option Market Data/2020_11_10/price_data_2020_11_10.csv")
for file in range(10,17):
    if file !=10:
        add = pd.read_csv("Crypto Data Repo/Option Market Data/2020_11_10/price_data_2020_11_"+str(file)+".csv")
        priceData161120 = priceData161120.append(add)
    add12 = pd.read_csv("Crypto Data Repo/Option Market Data/2020_11_10/price_data_2020_11_"+str(file)+"_12.csv")
    priceData161120 = priceData161120.append(add12)
    
startDate = datetime.strptime('2020-11-16','%Y-%m-%d')
## ETH
priceData161120ETH = priceData161120.loc[priceData161120['instrument'].str.startswith('deribit::ETH')]
locationList = []
locArray = np.array(priceData161120ETH['instrument'])
for instrument in priceData161120ETH['instrument'].unique():
    locationList.append(np.where(locArray==instrument)[0][-1])
locationList.sort()
priceData161120ETH = priceData161120ETH.iloc[locationList]
firstDashETH = priceData161120ETH['instrument'].str.find('-')
middleDashETH = priceData161120ETH['instrument'].str.find('-',12+1)
lastDashETH = priceData161120ETH['instrument'].str.rfind('-')

timeToMatETH = priceData161120ETH['instrument'].copy()
timeToMatETH.name = "TimeToMaturity"
strikeETH = priceData161120ETH['instrument'].copy()
strikeETH.name = "Strike"
optionType = priceData161120ETH['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData161120ETH.index:
    if middleDashETH.loc[quote] != -1:
        timeToMatETH.loc[quote] = (datetime.strptime(priceData161120ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
        strikeETH.loc[quote] = int(priceData161120ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
        optionType[quote] = priceData161120ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
    else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData161120ETH[timeToMatETH.name]=timeToMatETH.copy()
priceData161120ETH[strikeETH.name]=strikeETH.copy()
priceData161120ETH[optionType.name]=optionType.copy()
priceData161120ETH.dropna(inplace=True)

## BTC 
priceData161120BTC = priceData161120.loc[priceData161120['instrument'].str.startswith('deribit::BTC')]
locationList = []
locArray = np.array(priceData161120BTC['instrument'])
for instrument in priceData161120BTC['instrument'].unique():
    locationList.append(np.where(locArray==instrument)[0][-1])
locationList.sort()
priceData161120BTC = priceData161120BTC.iloc[locationList]
firstDashBTC = priceData161120BTC['instrument'].str.find('-')
middleDashBTC = priceData161120BTC['instrument'].str.find('-',12+1)
lastDashBTC = priceData161120BTC['instrument'].str.rfind('-')

timeToMatBTC = priceData161120BTC['instrument'].copy()
timeToMatBTC.name = "TimeToMaturity"
strikeBTC = priceData161120BTC['instrument'].copy()
strikeBTC.name = "Strike"
optionType = priceData161120BTC['instrument'].copy()
optionType.name = "OptionType"
for quote in priceData161120BTC.index:
    if middleDashBTC.loc[quote] != -1:
        timeToMatBTC.loc[quote] = (datetime.strptime(priceData161120BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
        strikeBTC.loc[quote] = int(priceData161120BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
        optionType[quote] = priceData161120BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
    else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
priceData161120BTC[timeToMatBTC.name]=timeToMatBTC.copy()
priceData161120BTC[strikeBTC.name]=strikeBTC.copy()
priceData161120BTC[optionType.name]=optionType.copy()
priceData161120BTC.dropna(inplace=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Plotting functions

def make_surf(X,Y,Z):
   XX,YY = np.meshgrid(np.linspace(min(X),max(X),2300),np.linspace(min(Y),max(Y),2300))
   ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
   return XX,YY,ZZ 

def make_surf_simple(X,Y,Z):
   XX,YY = np.meshgrid(np.linspace(min(X),max(X),230),np.linspace(min(Y),max(Y),230))
   ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
   return XX,YY,ZZ 

def mesh_plotBTC_Call(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf_simple(X,Y,Z)
   np.nan_to_num(ZZ[:,135:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=1000,limit_direction='both'))
   XX=XX[:,50:200];YY=YY[:,50:200];ZZ=ZZ[:,50:200]
   XX=XX[12:,:];YY=YY[12:,:];ZZ=ZZ[12:,:] 
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   return ax

def mesh_plotBTC_Call2(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf_simple(X,Y,Z)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=1000,limit_direction='both'))
   XX=XX[:,40:];YY=YY[:,40:];ZZ=ZZ[:,40:]
   XX=XX[20:,:];YY=YY[20:,:];ZZ=ZZ[20:,:]
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   return surf

def mesh_plotBTC(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,1350:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
   XX=XX[150:,:];YY=YY[150:,:];ZZ=ZZ[150:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230)
   return ax

   
def mesh_plotBTC_P(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,930:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,500:2000];YY=YY[:,500:2000];ZZ=ZZ[:,500:2000]
   XX=XX[170:,:];YY=YY[170:,:];ZZ=ZZ[170:,:] 
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230)
   return surf

   
def mesh_plotBTC_P1(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,930:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
   XX=XX[240:,:];YY=YY[240:,:];ZZ=ZZ[240:,:] 
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230) 
   return surf
   
def mesh_plotETH_Call(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf_simple(X,Y,Z)
   np.nan_to_num(ZZ[:,135:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=100,limit_direction='both'))
   XX=XX[:,50:196];YY=YY[:,50:196];ZZ=ZZ[:,50:196]
   XX=XX[12:,:];YY=YY[12:,:];ZZ=ZZ[12:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   return ax
   
def mesh_plotETH_Call2(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf_simple(X,Y,Z)
   np.nan_to_num(ZZ[:,100:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=100,limit_direction='both'))
   XX=XX[:,40:];YY=YY[:,40:];ZZ=ZZ[:,40:]
   XX=XX[20:,:];YY=YY[20:,:];ZZ=ZZ[20:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   return surf

def mesh_plotETH(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,1350:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
   XX=XX[120:,:];YY=YY[120:,:];ZZ=ZZ[120:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230) 
   return ax

def mesh_plotETH_P(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,1350:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
   XX=XX[200:,:];YY=YY[200:,:];ZZ=ZZ[200:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230) 
   return ax

def mesh_plotETH_P1(fig,ax,title,X,Y,Z):
   XX,YY,ZZ = make_surf(X,Y,Z)
   np.nan_to_num(ZZ[:,1000:],copy=False,nan=0)
   ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
   XX=XX[:,400:1670];YY=YY[:,400:1670];ZZ=ZZ[:,400:1670]
   XX=XX[200:,:];YY=YY[200:,:];ZZ=ZZ[200:,:]   
   my_cmap = plt.get_cmap('plasma') 
   surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
   ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
   ax.view_init(33, 230) 
   return surf
   
# Sorting option chain
def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical

def euro_vanilla_call(S, K, T, r, sigma):
   
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
    
    return call

def getOptionChain(test):  
    S0 = np.mean(test['base_price'])
    optionChain={}
    for T in test[timeToMatBTC.name].unique():
        if T>0:
            index = test[timeToMatBTC.name].index[np.where(test[timeToMatBTC.name]==T)]
            callValues = test['theo_price'].loc[index].copy()
            putValues = test['theo_price'].loc[index].copy()
            callValues = pd.Series([euro_vanilla_call(S0,test[strikeBTC.name].loc[x],T/365,0,test['theo_price'].loc[x]) for x in callValues.index],index=callValues.index,name=callValues.name)
            putValues = pd.Series([putOptionPriceAnalytical(S0,test[strikeBTC.name].loc[x],T/365,0,test['theo_price'].loc[x]) for x in putValues.index],index=putValues.index,name=putValues.name)
            optionChain[T] = pd.concat({'Strike':test[strikeBTC.name].loc[index], 'Call':callValues},axis = 1)
            optionChain[T] = pd.concat({'Strike':test[strikeBTC.name].loc[index], 'Call':callValues, 'Put':putValues},axis = 1)
    return optionChain
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Actual plotting

#BTC CALL
fig2, [axis12, axis22, axis32] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

#1
priceData010621BTC_CALL= priceData010621BTC.where(priceData010621BTC['OptionType']=='C').dropna()
test= priceData010621BTC_CALL.where((priceData010621BTC_CALL[strikeBTC.name]/priceData010621BTC_CALL['base_price']) <=2.5).dropna()
axis12 = mesh_plotBTC_Call(fig2,axis12,"7 June 2021",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

#2
priceData120321BTC_CALL= priceData120321BTC.where(priceData120321BTC['OptionType']=='C').dropna()
test= priceData120321BTC_CALL.where((priceData120321BTC_CALL[strikeBTC.name]/priceData120321BTC_CALL['base_price']) <=2.5).dropna()
axis22 = mesh_plotBTC_Call(fig2,axis22,"14 March 2021",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)


#3
priceData161120BTC_CALL= priceData161120BTC.where(priceData161120BTC['OptionType']=='C').dropna()
test= priceData161120BTC_CALL.where((priceData161120BTC_CALL[strikeBTC.name]/priceData161120BTC_CALL['base_price']) <=2.5).dropna()
surf = mesh_plotBTC_Call2(fig2,axis32,"16 November 2020",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)


fig2.colorbar(surf, ax=[axis12, axis22, axis32], shrink = 0.4, aspect = 7)
fig2.suptitle("Implied volatility surfaces Bitcoin calls",x=0.5,y=0.85,fontsize= 24, fontweight='bold')
plt.rc('axes', labelsize=14)  
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BTC PUT
fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

#1
priceData010621BTC_PUT= priceData010621BTC.where(priceData010621BTC['OptionType']=='P').dropna()
test= priceData010621BTC_PUT.where((priceData010621BTC_PUT[strikeBTC.name]/priceData010621BTC_PUT['base_price']) <=2.5).dropna()
surf = mesh_plotBTC(fig,axis1,"7 June 2021",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)
#2
priceData120321BTC_PUT= priceData120321BTC.where(priceData120321BTC['OptionType']=='P').dropna()
test= priceData120321BTC_PUT.where((priceData120321BTC_PUT[strikeBTC.name]/priceData120321BTC_PUT['base_price']) <=2.5).dropna()
surf = mesh_plotBTC_P(fig,axis2,"14 March 2021",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)
#3
priceData161120BTC_PUT= priceData161120BTC.where(priceData161120BTC['OptionType']=='P').dropna()
test= priceData161120BTC_PUT.where((priceData161120BTC_PUT[strikeBTC.name]/priceData161120BTC_PUT['base_price']) <=2.5).dropna()
surf = mesh_plotBTC_P1(fig,axis3,"16 November 2020",test[strikeBTC.name]/test['base_price'],test[timeToMatBTC.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

fig.colorbar(surf, ax=[axis1, axis2, axis3], shrink = 0.4, aspect = 7)
fig.suptitle("Implied volatility surfaces Bitcoin puts",x=0.5,y=0.85,fontsize= 24, fontweight='bold')
plt.rc('axes', labelsize=14)  
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #ETH CALL
fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

#1
priceData010621ETH_CALL= priceData010621ETH.where(priceData010621ETH['OptionType']=='C').dropna()
test= priceData010621ETH_CALL.where((priceData010621ETH_CALL[strikeETH.name]/priceData010621ETH_CALL['base_price']) <=2.5).dropna()
axis1 = mesh_plotETH_Call(fig,axis1,"7 June 2021",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

#2
priceData120321ETH_CALL= priceData120321ETH.where(priceData120321ETH['OptionType']=='C').dropna()
test= priceData120321ETH_CALL.where((priceData120321ETH_CALL[strikeETH.name]/priceData120321ETH_CALL['base_price']) <=2.5).dropna()
axis2 = mesh_plotETH_Call(fig,axis2,"14 March 2021",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

#3
priceData161120ETH_CALL= priceData161120ETH.where(priceData161120ETH['OptionType']=='C').dropna()
test= priceData161120ETH_CALL.where((priceData161120ETH_CALL[strikeETH.name]/priceData161120ETH_CALL['base_price']) <=2.5).dropna()
surf = mesh_plotETH_Call2(fig,axis3,"16 November 2020",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)


fig.colorbar(surf, ax=[axis1, axis2, axis3], shrink = 0.4, aspect = 7)
fig.suptitle("Implied volatility surfaces Ethereum calls",x=0.5,y=0.85,fontsize= 24, fontweight='bold')
plt.rc('axes', labelsize=14)   
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ETH PUT
fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

#1
priceData010621ETH_PUT= priceData010621ETH.where(priceData010621ETH['OptionType']=='P').dropna()
test= priceData010621ETH_PUT.where((priceData010621ETH_PUT[strikeETH.name]/priceData010621ETH_PUT['base_price']) <=2.5).dropna()
axis1 = mesh_plotETH(fig,axis1,"7 June 2021",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

#2
priceData120321ETH_PUT= priceData120321ETH.where(priceData120321ETH['OptionType']=='P').dropna()
test= priceData120321ETH_PUT.where((priceData120321ETH_PUT[strikeETH.name]/priceData120321ETH_PUT['base_price']) <=2.5).dropna()
axis2 = mesh_plotETH_P(fig,axis2,"14 March 2021",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

#3
priceData161120ETH_PUT= priceData161120ETH.where(priceData161120ETH['OptionType']=='P').dropna()
test= priceData161120ETH_PUT.where((priceData161120ETH_PUT[strikeETH.name]/priceData161120ETH_PUT['base_price']) <=2.5).dropna()
surf = mesh_plotETH_P1(fig,axis3,"16 November 2020",test[strikeETH.name]/test['base_price'],test[timeToMatETH.name]/365,test['theo_price']/np.sqrt(365))

optionChain = getOptionChain(test)

fig.colorbar(surf, ax=[axis1, axis2, axis3], shrink = 0.4, aspect = 7)
fig.suptitle("Implied volatility surfaces Ethereum puts",x=0.5,y=0.85,fontsize= 24, fontweight='bold')
plt.rc('axes', labelsize=14)   
plt.show()


