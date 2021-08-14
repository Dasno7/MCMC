# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:04:22 2021

@author: Dasno7
"""

import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
import scipy.special as special
from scipy.interpolate import griddata
from matplotlib import colors
import matplotlib.pyplot as plt
import statsmodels.api as sm

#------------------------------------------------------------------------------------------------------
# Functions

def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical

def black_scholes_call(S, K, T, r, sigma):

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * special.ndtr(d1) -  special.ndtr(d2)* K * np.exp(-r * T)
    return call


def priceRealWorldOptions(S,realData):
    options=np.zeros(realData.shape[0]);i=0
    for index, row in realData.iterrows():
        if row['OptionType']=='P':
            options[i] = putOptionPriceAnalytical(S,row['Strike'],row['TimeToMaturity']/365,0,row['theo_price'])    
        if row['OptionType']=='C':
            options[i] = black_scholes_call(S,row['Strike'],row['TimeToMaturity']/365,0,row['theo_price'])    
        i+=1

    realData['Option Value'] = options
    return realData

def priceTheoreticalOptions(S,theoData):
    
    for model in ['GBM IV', 'GBMJ IV', 'Heston IV', 'SVCJ IV']:    
        callOptions=np.zeros(theoData.shape[0]);putOptions=np.zeros(theoData.shape[0]);i=0
        for index, row in theoData.iterrows():
            putOptions[i] = putOptionPriceAnalytical(S,row['K'],row['T']/365,0,row[model])    
            callOptions[i] = black_scholes_call(S,row['K'],row['T']/365,0,row[model])    
            i+=1
        theoData[model[:-3]+" Put"] = putOptions
        theoData[model[:-3]+" Call"] = callOptions
    return theoData

def matchRealandTheoreticalOptions(realData,fourierData,typeOption):
    optionGBMFourier = np.zeros(realData.shape[0]);optionGBMJFourier = np.zeros(realData.shape[0]);i=0;
    optionHestonFourier = np.zeros(realData.shape[0]);optionSVCJFourier = np.zeros(realData.shape[0]);
    for index,row in realData.iterrows():
        fourierRow =  fourierData.loc[(fourierData["K"] == row['Strike']) & (fourierData["T"]==row['TimeToMaturity'])]
        if len(fourierRow)!=0:
            optionGBMFourier[i] =fourierRow['GBM '+typeOption].iloc[-1];    optionGBMJFourier[i] =fourierRow['GBMJ '+typeOption].iloc[-1]
            optionHestonFourier[i] =fourierRow['Heston '+typeOption].iloc[-1];    optionSVCJFourier[i] =fourierRow['SVCJ '+typeOption].iloc[-1]
        else:
            optionGBMFourier[i] =np.nan;    optionGBMJFourier[i] =np.nan
            optionHestonFourier[i] =np.nan;    optionSVCJFourier[i] =np.nan 
        i+=1
    realData["GBM Fourier "+typeOption] = optionGBMFourier
    realData["GBMJ Fourier "+typeOption] = optionGBMJFourier
    realData["Heston Fourier "+typeOption] = optionHestonFourier
    realData["SVCJ Fourier "+typeOption] = optionSVCJFourier
    return realData

#------------------------------------------------------------------------------------------------------
# Read real option data

# """Pricing data for 14-03-2021""" 
# priceData120321 = pd.read_csv("C:/Users/Sameer/Documents/Econometrics/Thesis/Crypto Data/thesis_data/2021_03_12/price_data_2021_03_12.csv")
# startDate = datetime.strptime('2021-03-14','%Y-%m-%d')
# ## ETH
# priceData120321ETH = priceData120321.loc[(priceData120321['instrument'].str.startswith('deribit::ETH')&priceData120321['creation_time'].str.startswith('2021-03'))]
# locationList = []
# for instrument in priceData120321ETH['instrument'].unique():
#     locationList.append(np.where(priceData120321ETH['instrument']==instrument)[0][-1])
#     #locationList.append(priceData120321ETH['instrument'].where(priceData120321ETH['instrument']==instrument).last_valid_index())
# locationList.sort()
# priceData120321ETH = priceData120321ETH.iloc[locationList]
# firstDashETH = priceData120321ETH['instrument'].str.find('-')
# middleDashETH = priceData120321ETH['instrument'].str.find('-',12+1)
# lastDashETH = priceData120321ETH['instrument'].str.rfind('-')

# timeToMatETH = priceData120321ETH['instrument'].copy()
# timeToMatETH.name = "TimeToMaturity"
# strikeETH = priceData120321ETH['instrument'].copy()
# strikeETH.name = "Strike"
# optionType = priceData120321ETH['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData120321ETH.index:
#     if middleDashETH.loc[quote] != -1:
#         timeToMatETH.loc[quote] = (datetime.strptime(priceData120321ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
#         strikeETH.loc[quote] = int(priceData120321ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
#         optionType[quote] = priceData120321ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
#     else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData120321ETH[timeToMatETH.name]=timeToMatETH.copy()
# priceData120321ETH[strikeETH.name]=strikeETH.copy()
# priceData120321ETH[optionType.name]=optionType.copy()
# priceData120321ETH.dropna(inplace=True)


# ## BTC 
# priceData120321BTC = priceData120321.loc[(priceData120321['instrument'].str.startswith('deribit::BTC')&priceData120321['creation_time'].str.startswith('2021-03'))]
# locationList = []
# for instrument in priceData120321BTC['instrument'].unique():
#     locationList.append(np.where(priceData120321BTC['instrument']==instrument)[0][-1])
# locationList.sort()
# priceData120321BTC = priceData120321BTC.iloc[locationList]
# firstDashBTC = priceData120321BTC['instrument'].str.find('-')
# middleDashBTC = priceData120321BTC['instrument'].str.find('-',12+1)
# lastDashBTC = priceData120321BTC['instrument'].str.rfind('-')

# timeToMatBTC = priceData120321BTC['instrument'].copy()
# timeToMatBTC.name = "TimeToMaturity"
# strikeBTC = priceData120321BTC['instrument'].copy()
# strikeBTC.name = "Strike"
# optionType = priceData120321BTC['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData120321BTC.index:
#     if middleDashBTC.loc[quote] != -1:
#         timeToMatBTC.loc[quote] = (datetime.strptime(priceData120321BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
#         strikeBTC.loc[quote] = int(priceData120321BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
#         optionType[quote] = priceData120321BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
#     else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData120321BTC[timeToMatBTC.name]=timeToMatBTC.copy()
# priceData120321BTC[strikeBTC.name]=strikeBTC.copy()
# priceData120321BTC[optionType.name]=optionType.copy()
# priceData120321BTC.dropna(inplace=True)


# """Pricing data for 07-06-2021""" 
# priceData010621 = pd.read_csv("C:/Users/Sameer/Documents/Econometrics/Thesis/Crypto Data/thesis_data/2021_06_01/price_data_2021_06_01.csv")
# startDate = datetime.strptime('2021-06-07','%Y-%m-%d')
# ## ETH
# priceData010621ETH = priceData010621.loc[(priceData010621['instrument'].str.startswith('deribit::ETH')&priceData010621['creation_time'].str.startswith('2021-06'))]
# locationList = []
# for instrument in priceData010621ETH['instrument'].unique():
#     locationList.append(np.where(priceData010621ETH['instrument']==instrument)[0][-1])
# locationList.sort()
# priceData010621ETH = priceData010621ETH.iloc[locationList]
# firstDashETH = priceData010621ETH['instrument'].str.find('-')
# middleDashETH = priceData010621ETH['instrument'].str.find('-',12+1)
# lastDashETH = priceData010621ETH['instrument'].str.rfind('-')

# timeToMatETH = priceData010621ETH['instrument'].copy()
# timeToMatETH.name = "TimeToMaturity"
# strikeETH = priceData010621ETH['instrument'].copy()
# strikeETH.name = "Strike"
# optionType = priceData010621ETH['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData010621ETH.index:
#     if middleDashETH.loc[quote] != -1:
#         timeToMatETH.loc[quote] = (datetime.strptime(priceData010621ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
#         strikeETH.loc[quote] = int(priceData010621ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
#         optionType[quote] = priceData010621ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
#     else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData010621ETH[timeToMatETH.name]=timeToMatETH.copy()
# priceData010621ETH[strikeETH.name]=strikeETH.copy()
# priceData010621ETH[optionType.name]=optionType.copy()
# priceData010621ETH.dropna(inplace=True)


# ## BTC 
# priceData010621BTC = priceData010621.loc[(priceData010621['instrument'].str.startswith('deribit::BTC')&priceData010621['creation_time'].str.startswith('2021-06'))]
# locationList = []
# for instrument in priceData010621BTC['instrument'].unique():
#     locationList.append(np.where(priceData010621BTC['instrument']==instrument)[0][-1])
# locationList.sort()
# priceData010621BTC = priceData010621BTC.iloc[locationList]
# firstDashBTC = priceData010621BTC['instrument'].str.find('-')
# middleDashBTC = priceData010621BTC['instrument'].str.find('-',12+1)
# lastDashBTC = priceData010621BTC['instrument'].str.rfind('-')

# timeToMatBTC = priceData010621BTC['instrument'].copy()
# timeToMatBTC.name = "TimeToMaturity"
# strikeBTC = priceData010621BTC['instrument'].copy()
# strikeBTC.name = "Strike"
# optionType = priceData010621BTC['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData010621BTC.index:
#     if middleDashBTC.loc[quote] != -1:
#         timeToMatBTC.loc[quote] = (datetime.strptime(priceData010621BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
#         strikeBTC.loc[quote] = int(priceData010621BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
#         optionType[quote] = priceData010621BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
#     else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData010621BTC[timeToMatBTC.name]=timeToMatBTC.copy()
# priceData010621BTC[strikeBTC.name]=strikeBTC.copy()
# priceData010621BTC[optionType.name]=optionType.copy()
# priceData010621BTC.dropna(inplace=True)

# """Pricing data for 16-11-2020""" 
# priceData161120 = pd.read_csv("C:/Users/Sameer/Documents/Econometrics/Thesis/Crypto Data/thesis_data/2020_11_10/price_data_2020_11_10.csv")
# for file in range(10,17):
#     if file !=10:
#         add = pd.read_csv("C:/Users/Sameer/Documents/Econometrics/Thesis/Crypto Data/thesis_data/2020_11_10/price_data_2020_11_"+str(file)+".csv")
#         priceData161120 = priceData161120.append(add)
#     add12 = pd.read_csv("C:/Users/Sameer/Documents/Econometrics/Thesis/Crypto Data/thesis_data/2020_11_10/price_data_2020_11_"+str(file)+"_12.csv")
#     priceData161120 = priceData161120.append(add12)
    
# startDate = datetime.strptime('2020-11-16','%Y-%m-%d')
# ## ETH
# priceData161120ETH = priceData161120.loc[priceData161120['instrument'].str.startswith('deribit::ETH')]
# locationList = []
# locArray = np.array(priceData161120ETH['instrument'])
# for instrument in priceData161120ETH['instrument'].unique():
#     locationList.append(np.where(locArray==instrument)[0][-1])
# locationList.sort()
# priceData161120ETH = priceData161120ETH.iloc[locationList]
# firstDashETH = priceData161120ETH['instrument'].str.find('-')
# middleDashETH = priceData161120ETH['instrument'].str.find('-',12+1)
# lastDashETH = priceData161120ETH['instrument'].str.rfind('-')

# timeToMatETH = priceData161120ETH['instrument'].copy()
# timeToMatETH.name = "TimeToMaturity"
# strikeETH = priceData161120ETH['instrument'].copy()
# strikeETH.name = "Strike"
# optionType = priceData161120ETH['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData161120ETH.index:
#     if middleDashETH.loc[quote] != -1:
#         timeToMatETH.loc[quote] = (datetime.strptime(priceData161120ETH.loc[quote]['instrument'][firstDashETH.loc[quote]+1:middleDashETH.loc[quote]],'%d%b%y')-startDate).days
#         strikeETH.loc[quote] = int(priceData161120ETH.loc[quote]['instrument'][middleDashETH.loc[quote]+1:lastDashETH.loc[quote]])
#         optionType[quote] = priceData161120ETH.loc[quote]['instrument'][lastDashETH.loc[quote]+1:]
#     else:  timeToMatETH.loc[quote] = np.nan; strikeETH.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData161120ETH[timeToMatETH.name]=timeToMatETH.copy()
# priceData161120ETH[strikeETH.name]=strikeETH.copy()
# priceData161120ETH[optionType.name]=optionType.copy()
# priceData161120ETH.dropna(inplace=True)

# ## BTC 
# priceData161120BTC = priceData161120.loc[priceData161120['instrument'].str.startswith('deribit::BTC')]
# locationList = []
# locArray = np.array(priceData161120BTC['instrument'])
# for instrument in priceData161120BTC['instrument'].unique():
#     locationList.append(np.where(locArray==instrument)[0][-1])
# locationList.sort()
# priceData161120BTC = priceData161120BTC.iloc[locationList]
# firstDashBTC = priceData161120BTC['instrument'].str.find('-')
# middleDashBTC = priceData161120BTC['instrument'].str.find('-',12+1)
# lastDashBTC = priceData161120BTC['instrument'].str.rfind('-')

# timeToMatBTC = priceData161120BTC['instrument'].copy()
# timeToMatBTC.name = "TimeToMaturity"
# strikeBTC = priceData161120BTC['instrument'].copy()
# strikeBTC.name = "Strike"
# optionType = priceData161120BTC['instrument'].copy()
# optionType.name = "OptionType"
# for quote in priceData161120BTC.index:
#     if middleDashBTC.loc[quote] != -1:
#         timeToMatBTC.loc[quote] = (datetime.strptime(priceData161120BTC.loc[quote]['instrument'][firstDashBTC.loc[quote]+1:middleDashBTC.loc[quote]],'%d%b%y')-startDate).days
#         strikeBTC.loc[quote] = int(priceData161120BTC.loc[quote]['instrument'][middleDashBTC.loc[quote]+1:lastDashBTC.loc[quote]])
#         optionType[quote] = priceData161120BTC.loc[quote]['instrument'][lastDashBTC.loc[quote]+1:]
#     else:  timeToMatBTC.loc[quote] = np.nan; strikeBTC.loc[quote]=np.nan; optionType[quote]=np.nan
    
# priceData161120BTC[timeToMatBTC.name]=timeToMatBTC.copy()
# priceData161120BTC[strikeBTC.name]=strikeBTC.copy()
# priceData161120BTC[optionType.name]=optionType.copy()
# priceData161120BTC.dropna(inplace=True)

#------------------------------------------------------------------------------------------------------
# Read fourier results data

#BTC
BTC1_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_BTC6.xlsx',sheet_name='7 June',index_col=0)
#BTC1_Fourier["GBMJ IV"]= BTC1_Fourier["GBMJ IV"]+0.05
S=35198;BTC1_Fourier = priceTheoreticalOptions(S,BTC1_Fourier)

BTC2_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_BTC6.xlsx',sheet_name='14 March',index_col=0)
#BTC2_Fourier["GBMJ IV"]= BTC2_Fourier["GBMJ IV"]+0.05
S=57663;BTC2_Fourier = priceTheoreticalOptions(S,BTC2_Fourier)

BTC3_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_BTC6.xlsx',sheet_name='16 November',index_col=0)
#BTC3_Fourier["GBMJ IV"]= BTC3_Fourier["GBMJ IV"]+0.05
S=16053;BTC3_Fourier = priceTheoreticalOptions(S,BTC3_Fourier)

#ETH
ETH1_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_ETH6.xlsx',sheet_name='7 June',index_col=0)
ETH1_Fourier["GBMJ IV"]= ETH1_Fourier["GBMJ IV"]
S=2647;ETH1_Fourier = priceTheoreticalOptions(S,ETH1_Fourier)

ETH2_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_ETH6.xlsx',sheet_name='14 March',index_col=0)
ETH2_Fourier["GBMJ IV"]= ETH2_Fourier["GBMJ IV"]
S=1795;ETH2_Fourier = priceTheoreticalOptions(S,ETH2_Fourier)

ETH3_Fourier = pd.read_excel('Crypto Data Repo/Fourier_IV_ETH6.xlsx',sheet_name='16 November',index_col=0)
ETH3_Fourier["GBMJ IV"]= ETH3_Fourier["GBMJ IV"]
S=458;ETH3_Fourier = priceTheoreticalOptions(S,ETH3_Fourier)

#-------------------------------------------------------------------------------------------------------
# Sort Real world data

#Puts

#BTC
S=35198;priceData010621BTC=priceRealWorldOptions(S,priceData010621BTC)
priceData010621BTC_PUT= priceData010621BTC.where(priceData010621BTC['OptionType']=='P').dropna()
priceData010621BTC_PUT= priceData010621BTC_PUT.where((priceData010621BTC_PUT[strikeBTC.name]/priceData010621BTC_PUT['base_price']) <=2).dropna()


S=57663;priceData120321BTC=priceRealWorldOptions(S,priceData120321BTC)
priceData120321BTC_PUT= priceData120321BTC.where(priceData120321BTC['OptionType']=='P').dropna()
priceData120321BTC_PUT= priceData120321BTC_PUT.where((priceData120321BTC_PUT[strikeBTC.name]/priceData120321BTC_PUT['base_price']) <=2).dropna()

S=16053;priceData161120BTC=priceRealWorldOptions(S,priceData161120BTC)
priceData161120BTC_PUT= priceData161120BTC.where(priceData161120BTC['OptionType']=='P').dropna()
priceData161120BTC_PUT= priceData161120BTC_PUT.where((priceData161120BTC_PUT[strikeBTC.name]/priceData161120BTC_PUT['base_price']) <=2).dropna()

#ETH
S=2647;priceData010621ETH=priceRealWorldOptions(S,priceData010621ETH)
priceData010621ETH_PUT= priceData010621ETH.where(priceData010621ETH['OptionType']=='P').dropna()
priceData010621ETH_PUT= priceData010621ETH_PUT.where((priceData010621ETH_PUT[strikeETH.name]/priceData010621ETH_PUT['base_price']) <=2).dropna()

S=1795;priceData120321ETH_PUT=priceRealWorldOptions(S,priceData120321ETH)
priceData120321ETH_PUT= priceData120321ETH.where(priceData120321ETH['OptionType']=='P').dropna()
priceData120321ETH_PUT= priceData120321ETH_PUT.where((priceData120321ETH_PUT[strikeETH.name]/priceData120321ETH_PUT['base_price']) <=2).dropna()

S=458;priceData161120ETH=priceRealWorldOptions(S,priceData161120ETH)
priceData161120ETH_PUT= priceData161120ETH.where(priceData161120ETH['OptionType']=='P').dropna()
priceData161120ETH_PUT= priceData161120ETH_PUT.where((priceData161120ETH_PUT[strikeETH.name]/priceData161120ETH_PUT['base_price']) <=2).dropna()


#Calls

#BTC
priceData010621BTC_CALL= priceData010621BTC.where(priceData010621BTC['OptionType']=='C').dropna()
priceData010621BTC_CALL= priceData010621BTC_CALL.where((priceData010621BTC_CALL[strikeBTC.name]/priceData010621BTC_CALL['base_price']) <=2).dropna()

priceData120321BTC_CALL= priceData120321BTC.where(priceData120321BTC['OptionType']=='C').dropna()
priceData120321BTC_CALL= priceData120321BTC_CALL.where((priceData120321BTC_CALL[strikeBTC.name]/priceData120321BTC_CALL['base_price']) <=2).dropna()

priceData161120BTC_CALL= priceData161120BTC.where(priceData161120BTC['OptionType']=='C').dropna()
priceData161120BTC_CALL= priceData161120BTC_CALL.where((priceData161120BTC_CALL[strikeBTC.name]/priceData161120BTC_CALL['base_price']) <=2).dropna()

#ETH
priceData010621ETH_CALL= priceData010621ETH.where(priceData010621ETH['OptionType']=='C').dropna()
priceData010621ETH_CALL= priceData010621ETH_CALL.where((priceData010621ETH_CALL[strikeETH.name]/priceData010621ETH_CALL['base_price']) <=2).dropna()

priceData120321ETH_CALL= priceData120321ETH.where(priceData120321ETH['OptionType']=='C').dropna()
priceData120321ETH_CALL= priceData120321ETH_CALL.where((priceData120321ETH_CALL[strikeETH.name]/priceData120321ETH_CALL['base_price']) <=2).dropna()

priceData161120ETH_CALL= priceData161120ETH.where(priceData161120ETH['OptionType']=='C').dropna()
priceData161120ETH_CALL= priceData161120ETH_CALL.where((priceData161120ETH_CALL[strikeETH.name]/priceData161120ETH_CALL['base_price']) <=2).dropna()

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Match option prices

#BTC
priceData010621BTC_PUT = matchRealandTheoreticalOptions(priceData010621BTC_PUT,BTC1_Fourier,'Put')
priceData120321BTC_PUT = matchRealandTheoreticalOptions(priceData120321BTC_PUT,BTC2_Fourier,'Put')
priceData161120BTC_PUT = matchRealandTheoreticalOptions(priceData161120BTC_PUT,BTC3_Fourier,'Put')

priceData010621BTC_CALL = matchRealandTheoreticalOptions(priceData010621BTC_CALL,BTC1_Fourier,'Call')
priceData120321BTC_CALL = matchRealandTheoreticalOptions(priceData120321BTC_CALL,BTC2_Fourier,'Call')
priceData161120BTC_CALL = matchRealandTheoreticalOptions(priceData161120BTC_CALL,BTC3_Fourier,'Call')

#ETH
priceData010621ETH_PUT = matchRealandTheoreticalOptions(priceData010621ETH_PUT,ETH1_Fourier,'Put')
priceData120321ETH_PUT = matchRealandTheoreticalOptions(priceData120321ETH_PUT,ETH2_Fourier,'Put')
priceData161120ETH_PUT = matchRealandTheoreticalOptions(priceData161120ETH_PUT,ETH3_Fourier,'Put')

priceData010621ETH_CALL = matchRealandTheoreticalOptions(priceData010621ETH_CALL,ETH1_Fourier,'Call')
priceData120321ETH_CALL = matchRealandTheoreticalOptions(priceData120321ETH_CALL,ETH2_Fourier,'Call')
priceData161120ETH_CALL = matchRealandTheoreticalOptions(priceData161120ETH_CALL,ETH3_Fourier,'Call')


#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Regression analysis

def regress1(df,optionType):
    x_list = ['GBM Fourier '+optionType,'GBMJ Fourier '+optionType,'Heston Fourier '+optionType,'SVCJ Fourier '+optionType]
    resArray = np.zeros([len(x_list),8]);i=0
    for x in x_list:
        tempDF = df[[x,'Option Value']].dropna(axis=0)
        res = stats.linregress(tempDF[x], tempDF['Option Value'])
        R=np.identity(2);b_hat = np.array([[res.intercept,res.slope]]).T;r=np.array([[0,1]]).T;
        s=np.linalg.inv(np.array([[res.intercept_stderr, 0], [0, res.stderr]]));wald = (R@b_hat-R@r).T@s@(R@b_hat-R@r)     
        tVal=(res.slope-1)/res.stderr
        
        y = tempDF['Option Value']
        X = tempDF[x];        X = sm.add_constant(X)
        model = sm.OLS(y.astype('float64'),X.astype('float64'))
        resOLS = model.fit()
        waldOLS = resOLS.wald_test((np.identity(2),r),use_f=False)
        
        resArray[i,:] = [res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr,res.intercept_stderr,waldOLS.statistic,tVal];i+=1
    resDF = pd.DataFrame(resArray,columns=['slope', 'intercept', 'rvalue','pvalue', 'stderr','intercept_stderr','Wald','tvalue'],index=x_list)  
    return resDF

def regress3(dfPut,dfCall):
    x_list = ['GBM Fourier ','GBMJ Fourier ','Heston Fourier ','SVCJ Fourier ']
    resArray = np.zeros([len(x_list),8]);i=0
    for x in x_list:
        tempDFPut = dfPut[[(x+'Put'),'Option Value']].dropna(axis=0)
        tempDFCall = dfCall[[(x+'Call'),'Option Value']].dropna(axis=0)
        tempDF = pd.concat([tempDFPut.rename(columns={(x+'Put'):x}),tempDFCall.rename(columns={(x+'Call'):x})], ignore_index=True)
        res = stats.linregress(tempDF[x], tempDF['Option Value'])
        R=np.identity(2);b_hat = np.array([[res.intercept,res.slope]]).T;r=np.array([[0,1]]).T;
        s=np.linalg.inv(np.array([[res.intercept_stderr, 0], [0, res.stderr]]));wald = (R@b_hat-R@r).T@s@(R@b_hat-R@r)     
        tVal=(res.slope-1)/res.stderr
        
        y = tempDF['Option Value']
        X = tempDF[x];        X = sm.add_constant(X)
        model = sm.OLS(y.astype('float64'),X.astype('float64'))
        resOLS = model.fit()
        waldOLS = resOLS.wald_test((np.identity(2),r),use_f=False)
        
        resArray[i,:] = [res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr,res.intercept_stderr,waldOLS.statistic,tVal];i+=1
    resDF = pd.DataFrame(resArray,columns=['slope', 'intercept', 'rvalue','pvalue', 'stderr','intercept_stderr','Wald','tvalue'],index=x_list)  
    return resDF


def regress2(df,optionType):
    x_list = ['GBM Fourier '+optionType,'GBMJ Fourier '+optionType,'Heston Fourier '+optionType,'SVCJ Fourier '+optionType]
    resArray = np.zeros([len(x_list),9]);i=0
    for x in x_list:
        tempDF = df[[x,'Option Value','Strike','TimeToMaturity','base_price']].dropna(axis=0)
        y = np.array(tempDF['Option Value']-tempDF[x])
        X = np.array([(tempDF['base_price']-tempDF['Strike'])/tempDF['Strike'],tempDF['TimeToMaturity']/365]).T
        X = sm.add_constant(X)
        model = sm.OLS(y.astype('float64'),X.astype('float64'))
        res = model.fit()
        fres=res.f_test([[0,1,0],[0,0,1]])
        resArray[i,:] = [res.params[0], res.params[1],res.params[2], res.bse[0],res.bse[1],res.bse[2],fres.fvalue,res.rsquared,fres.pvalue];i+=1
    resDF = pd.DataFrame(resArray,columns=['intercept','slope1','slope2', 'intercept_stderr','stderr1','stderr2','ftest','rvalue','f_pvalue'],index=x_list)  
    return resDF

regress010621BTC_PUT = regress1(priceData010621BTC_PUT,'Put')
regress010621BTC_CALL = regress1(priceData010621BTC_CALL,'Call')
regress010621BTC_PUT_2 = regress2(priceData010621BTC_PUT,'Put')
regress010621BTC_CALL_2 = regress2(priceData010621BTC_CALL,'Call')
regress010621BTC = regress3(priceData010621BTC_PUT,priceData010621BTC_CALL)

regress120321BTC_PUT = regress1(priceData120321BTC_PUT,'Put')
regress120321BTC_CALL = regress1(priceData120321BTC_CALL,'Call')
regress120321BTC_PUT_2 = regress2(priceData120321BTC_PUT,'Put')
regress120321BTC_CALL_2 = regress2(priceData120321BTC_CALL,'Call')
regress120321BTC = regress3(priceData120321BTC_PUT,priceData120321BTC_CALL)

regress161120BTC_PUT = regress1(priceData161120BTC_PUT,'Put')
regress161120BTC_CALL = regress1(priceData161120BTC_CALL,'Call')
regress161120BTC_PUT_2 = regress2(priceData161120BTC_PUT,'Put')
regress161120BTC_CALL_2 = regress2(priceData161120BTC_CALL,'Call')
regress161120BTC = regress3(priceData161120BTC_PUT,priceData161120BTC_CALL)

regress010621ETH_PUT = regress1(priceData010621ETH_PUT,'Put')
regress010621ETH_CALL = regress1(priceData010621ETH_CALL,'Call')
regress010621ETH_PUT_2 = regress2(priceData010621ETH_PUT,'Put')
regress010621ETH_CALL_2 = regress2(priceData010621ETH_CALL,'Call')
regress010621ETH = regress3(priceData010621ETH_PUT,priceData010621ETH_CALL)

regress120321ETH_PUT = regress1(priceData120321ETH_PUT,'Put')
regress120321ETH_CALL = regress1(priceData120321ETH_CALL,'Call')
regress120321ETH_PUT_2 = regress2(priceData120321ETH_PUT,'Put')
regress120321ETH_CALL_2 = regress2(priceData120321ETH_CALL,'Call')
regress120321ETH = regress3(priceData120321ETH_PUT,priceData120321ETH_CALL)

regress161120ETH_PUT = regress1(priceData161120ETH_PUT,'Put')
regress161120ETH_CALL = regress1(priceData161120ETH_CALL,'Call')
regress161120ETH_PUT_2 = regress2(priceData161120ETH_PUT,'Put')
regress161120ETH_CALL_2 = regress2(priceData161120ETH_CALL,'Call')
regress161120ETH = regress3(priceData161120ETH_PUT,priceData161120ETH_CALL)


def output(df1,df2,df3):
    string = '&'+ str(round(df1['intercept'])) +'&'+  str(round(df1['slope1']))  +'&'+ str(round(df1['slope2']))+\
            '&'+ str(round(df1['ftest'],2))  +'&'+  str(round(df1['rvalue'],2)) +'&'+  str(round(df2['intercept'])) +'&'+ str(round(df2['slope1']))\
                +'&'+ str(round(df2['slope2']))+'&'+  str(round(df2['ftest'],2))+'&'+ str(round(df2['rvalue'],2))\
                    +'&'+ str(round(df3['intercept']))+'&'+ str(round(df3['slope1']))+'&'+ str(round(df3['slope2']))+'&'+ str(round(df3['ftest'],2))+'&'+ str(round(df3['rvalue'],2))+'\\' \
                    +'&('+str(round(df1['intercept_stderr']))+')&('+str(round(df1['stderr1']))+')&('+str(round(df1['stderr2']))+')&&&('+str(round(df2['intercept_stderr']))+')&('+str(round(df2['stderr1']))\
                        +')&('+str(round(df2['stderr2']))+')&&&('+str(round(df3['intercept_stderr']))+')&('+str(round(df3['stderr1']))+')&('+str(round(df3['stderr2']))+')&&'+'\\'
    return string

model = 'SVCJ'
output(regress010621BTC_PUT_2.loc[model+' Fourier Put'],regress120321BTC_PUT_2.loc[model+' Fourier Put'],regress161120BTC_PUT_2.loc[model+' Fourier Put'])


#--------------------------------------------------------------------------------------------------------------------------------------
#Plot Functions

def make_surf(X,Y,Z):
    XX,YY = np.meshgrid(np.linspace(min(X),max(X),2300),np.linspace(min(Y),max(Y),2300))
    ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear',rescale=True)
    return XX,YY,ZZ 

def mesh_plotBTC(fig,ax,title,X,Y,Z):
    XX,YY,ZZ = make_surf(X,Y,Z)
    #np.nan_to_num(ZZ[:,1350:],copy=False,nan=0)
    ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
    XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
    XX=XX[700:,:];YY=YY[700:,:];ZZ=ZZ[700:,:]   
    my_cmap = plt.get_cmap('plasma') 
    surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0.83, vcenter=0.8325, vmax=0.835))
    ax.set_title(title, fontdict= { 'fontsize': 18},style='italic',x=0.5,y=0.95)
    #ax.view_init(33, 230) 
    ax.view_init(20, 60) 
    return surf

def mesh_plotETH(fig,ax,title,X,Y,Z):
    XX,YY,ZZ = make_surf(X,Y,Z)
    #np.nan_to_num(ZZ[:,1350:],copy=False,nan=0)
    ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=10000,limit_direction='both'))
    XX=XX[:,500:1700];YY=YY[:,500:1700];ZZ=ZZ[:,500:1700]
    XX=XX[500:,:];YY=YY[500:,:];ZZ=ZZ[500:,:]   
    my_cmap = plt.get_cmap('plasma') 
    surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=250, cstride=160,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=1.05, vcenter=1.055, vmax=1.06))
    ax.set_title(title, fontdict= { 'fontsize': 18},style='italic',x=0.5,y=0.95)
    #ax.view_init(33, 230) 
    ax.view_init(20, 60) 
    return surf



model='GBMJ IV'

fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)


BTC1_plot= BTC1_Fourier.where((BTC1_Fourier['K']/35198) <=2.5).dropna()
axis12 = mesh_plotBTC(fig,axis1,"7 June 2021",BTC1_plot['K']/35198,BTC1_plot['T']/365,BTC1_plot[model])#/np.log(np.sqrt(365)))

BTC2_plot= BTC2_Fourier.where((BTC2_Fourier['K']/57663) <=2.5).dropna()
axis22 = mesh_plotBTC(fig,axis2,"14 March 2021",BTC2_plot['K']/57663,BTC2_plot['T']/365,BTC2_plot[model])

BTC3_plot= BTC3_Fourier.where((BTC3_Fourier['K']/16053) <=2.5).dropna()
surf = mesh_plotBTC(fig,axis3,"16 November 2020",BTC3_plot['K']/16053,BTC3_plot['T']/365,BTC3_plot[model])

fig.colorbar(surf, ax=[axis1, axis2, axis3], shrink = 0.4, aspect = 7)
fig.suptitle("Implied volatility surfaces Bitcoin "+model[:-2]+"model",x=0.5,y=0.785,fontsize= 24, fontweight='bold')


model='GBMJ IV'

fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

ETH1_plot= ETH1_Fourier.where((ETH1_Fourier['K']/2647) <=2.5).dropna()
axis12 = mesh_plotETH(fig,axis1,"7 June 2021",ETH1_plot['K']/1795,ETH1_plot['T']/365,ETH1_plot[model])

ETH2_plot= ETH2_Fourier.where((ETH2_Fourier['K']/1795) <=5).dropna()
axis22 = mesh_plotETH(fig,axis2,"14 March 2021",ETH2_plot['K']/2647,ETH2_plot['T']/365,ETH2_plot[model])

ETH3_plot= ETH3_Fourier.where((ETH3_Fourier['K']/458) <=2.5).dropna()
surf = mesh_plotETH(fig,axis3,"16 November 2020",ETH3_plot['K']/458,ETH3_plot['T']/365,ETH3_plot[model])

fig.colorbar(surf, ax=[axis1, axis2, axis3], shrink = 0.4, aspect = 7)
fig.suptitle("Implied volatility surfaces Ethereum "+model[:-2]+"model",x=0.5,y=0.785,fontsize= 24, fontweight='bold')

