# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:02:25 2021

@author: Dasno7
"""

from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
from tqdm import tqdm


def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical


put7JuneBTC = pd.read_excel("BTC_PUT_GRID.xlsx", sheet_name='7 June')

S = 35198
putOptions = np.zeros(put7JuneBTC.shape[0])
for index, row in put7JuneBTC.iterrows():
    putOptions[index] = putOptionPriceAnalytical(
        S, row['Strike'], row['TimeToMaturity']/365, 0, row['theo_price'])
put7JuneBTC['Put Value'] = putOptions

T_BTC1 = np.array([2, 11, 18, 4, 1, 109, 53, 207, 291, 81])
T_BTC1.sort()


put7JuneBTC_sorted = put7JuneBTC[put7JuneBTC['TimeToMaturity'].isin(T_BTC1)]
IVs_real_7June_BTC = put7JuneBTC_sorted.pivot_table(columns='Strike', index='TimeToMaturity',
                                                    values='theo_price', fill_value=np.nan)
putVals_real_7June_BTC = put7JuneBTC_sorted.pivot_table(columns='Strike', index='TimeToMaturity',
                                                        values='Put Value', fill_value=np.nan)
K_BTC1 = putVals_real_7June_BTC.columns


put14MarchBTC = pd.read_excel(
    "BTC_PUT_GRID.xlsx", sheet_name='7 June')
put16NovemberBTC = pd.read_excel(
    "BTC_PUT_GRID.xlsx", sheet_name='7 June')
# ---------------------------------------------------------------------------------------------------------------------------------
# Param
mu_GBM = {'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642,
          'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}
sigma2_y_GBM = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903,
                'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}
mu_GBMJ = {'Bitcoin': 0.027147875324741517, 'LINK': 0.053533829570666804,
           'ETH': 0.039328550620623515, 'ADA': 0.01790843001866173}
mu_heston = {'Bitcoin': 0.02029249494235486, 'LINK': 0.04698867505564754,
             'ETH':  0.01583345207967061, 'ADA': -0.003431383092799574}
mu_svcj = {'Bitcoin': 0.021644682489966826, 'LINK': 0.03645939802343439,
           'ETH':  0.015283256840018638, 'ADA': -0.007316876526135481}

# -----------------------------------------------------------------------------------------------------
# Functions


def fi_GBM(x, param_dict, T, S):
    m_bs = np.log(S)+(param_dict['m']-0.5*param_dict['sigma2'])*T

    characFunc = np.exp(1j*x*m_bs-0.5*x**2*param_dict['sigma2']*T)
    return characFunc


def fi_heston(x, v, param_dict, T, S):
    # Heston model parameters
    # mu ={'Bitcoin': 0.01729249494235486, 'LINK':0.04698867505564754, 'ETH':  0.0073345207967061, 'ADA':-0.003431383092799574 }
    # s2V = {'Bitcoin': 0.00033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.00823061919130012, 'ADA':0.13299513034831426 }
    # alpha ={'Bitcoin':  0.057300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0527768090812799, 'ADA': 0.1500259628758315}
    # beta ={'Bitcoin':0.9078989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9414176612486856, 'ADA': 0.8795055872798027}
    # rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.03260865056494854 , 'ADA':-0.05034955922479641 }
    # kappa ={'Bitcoin':(1-beta['Bitcoin']) , 'LINK':(1-beta['LINK']), 'ETH':(1- beta['ETH']), 'ADA': (1-beta['ADA'])}
    # theta ={'Bitcoin':  alpha['Bitcoin']/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': alpha['ETH']/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}

    # mu ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }
    # s2V = {'Bitcoin': 0.04033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012, 'ADA':0.13299513034831426 }
    # alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
    # beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
    # rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
    # kappa ={'Bitcoin':(1-beta['Bitcoin']) , 'LINK':(1-beta['LINK']), 'ETH':(1- beta['ETH']), 'ADA': (1-beta['ADA'])}
    # theta ={'Bitcoin':  alpha['Bitcoin']/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': alpha['ETH']/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}

    d = np.sqrt(param_dict['s2V']*(1j*x+x**2)+(param_dict['rho']
                * param_dict['s2V']**(0.5)*1j*x-param_dict['kappa'])**2)

    c = (param_dict['kappa']-param_dict['s2V']**(0.5)*param_dict['rho']*1j*x-d) / \
        (param_dict['kappa']-param_dict['s2V']**(0.5)*param_dict['rho']*1j*x+d)
    beta = (param_dict['kappa']-param_dict['s2V']**(0.5)*param_dict['rho']
            * 1j*x-d)*(1-np.exp(-d*T))/(param_dict['s2V']*(1-c*np.exp(-d*T)))
    alpha = param_dict['kappa']*param_dict['theta']/param_dict['s2V'] * \
        ((param_dict['kappa']-param_dict['s2V']**(0.5) *
         param_dict['rho']*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
    m = np.log(S)+param_dict['mu']*T

    characFunc = np.exp(1j*x*m+alpha+beta*v)
    return characFunc


def gn(n, a, b, K):
    hn = n*np.pi/(b-a)
    g = (np.exp(a)-K/hn*np.sin(hn*(a-np.log(K))) -
         K*np.cos(hn*(a-np.log(K))))/(1+(hn)**2)

    return g


def f(M, a, b, T, fi, param_dict, S, K, **v0):
    g0 = K*(np.log(K)-a-1)+np.exp(a)
    if v0:
        putEst = g0 + np.array([2*gn(i, a, b, K)*np.exp(-np.pi*a*i*1j/(b-a))*fi(
            np.pi*i/(b-a), v0['v0'], param_dict, T, S) for i in range(1, M+1)]).sum()
    else:
        putEst = g0 + np.array([2*gn(i, a, b, K)*np.exp(-np.pi*a*i*1j/(b-a))*fi(
            np.pi*i/(b-a), param_dict, T, S) for i in range(1, M+1)]).sum()
    return putEst.real


def vega(S, K, T, r, sigma):
    # calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    vega = S * np.sqrt(T) * stats.norm._pdf(d1)
    return vega


def implied_volatility_put(C, S, K, T, r, tol=0.0001,
                           max_iterations=1e6):
    sigma = 0.5

    for i in range(max_iterations):

        # calculate difference between blackscholes price and market price with
        # iteratively updated volality estimate
        # +S-K*np.exp(-r*T))
        diff = C-(putOptionPriceAnalytical(S, K, T, r, sigma))

        # break if difference is less than specified tolerance level
        if abs(diff) < tol:
            #print(f'found on {i}th iteration')
            #print(f'difference is equal to {diff}')
            if i == 0:
                return np.nan
            else:
                return sigma
            # return sigma

        # use newton raphson to update the estimate
        sigma = sigma + diff / vega(S, K, T, r, sigma)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Perform Grid Search
param_grid = {'m': np.arange(0.65, 0.7, 0.001), 'sigma2': np.arange(
    0.225, 0.23, 0.001)}  # GBM results {'m': 0.6970000000000001, 'sigma2': 0.226}

param_grid = {'mu': np.arange(0.1, 0.6, 0.1), 's2V': np.arange(0.1, 0.5, 0.1),
              'rho': [-0.06], 'kappa': np.arange(0.9, 1.3, 0.1), 'theta': np.arange(0.0, 0.3, 0.1)}


paramTrials = ParameterGrid(param_grid)

# put7JuneBTC_sorted = put7JuneBTC[put7JuneBTC['Strike'].isin(K_BTC1)]
# put7JuneBTC_sorted = put7JuneBTC_sorted[put7JuneBTC['TimeToMaturity'].isin(T_BTC1)]
# put7JuneBTC_sorted.pivot_table(columns='Strike', index='TimeToMaturity',
#                           values='theo_price', fill_value=np.nan)


# K_BTC1 = np.array([42000, 37000, 33000, 39000, 40000, 30000, 34000, 36000, 32000,
#         38000, 35000, 41000, 45000, 26000, 55000, 20000, 46000, 75000,
#         80000, 44000, 48000, 85000, 60000, 56000, 25000, 58000, 50000,
#         54000, 31000, 70000, 28000, 52000, 65000, 43000, 24000, 14000,
#         16000, 18000, 12000, 64000, 15000, 8000, 68000, 10000, 22000,
#         72000, 6000, 76000])
# K_BTC1.sort()


RMSE_array = np.zeros(len(list(paramTrials)))
params = list(paramTrials)
for i in tqdm(range(len(params))):
    param_iter = params[i]
    crypto = 'Bitcoin'
    S = 35198

    putVall_array = np.zeros([T_BTC1.shape[0], K_BTC1.shape[0]])
    putVall_array[:] = np.nan

    # v0_svcj=0.99
    v0_sv = sigma2_y_GBM[crypto]
    v0_svcj = sigma2_y_GBM[crypto]
    #v0_svcj= sigma2_y_GBM[crypto]/np.log(np.sqrt(365))
    #v0_svcj= 0.043193377631366804*10
    M = int(2**4)

    for k in range(K_BTC1.shape[0]):
        for t in range(T_BTC1.shape[0]):
            if putVals_real_7June_BTC.iloc[t, k] != np.nan:
                T_days = T_BTC1[t]/365
                c1 = mu_GBM[crypto]*T_days
                c2 = sigma2_y_GBM[crypto]*T_days
                L = 300

                a = c1-L*np.sqrt(c2)
                b = c1+L*np.sqrt(c2)
                # callVal_BS = putOptionPriceAnalytical(S,K,T_days,mu_GBM[crypto],sigma2_y_GBM[crypto]**0.5)#+S-K*np.exp(-mu_GBM[crypto]*T_days)
                # callVal_GBMJ=f(M,a,b,T_days,fi_GBMJ,crypto)/(b-a)*np.exp(-mu_GBMJ[crypto]*T_days)#+S-K*np.exp(-mu_GBMJ[crypto]*T_days)
                # putVal_GBM=f(M,a,b,T_days,fi_GBM,param_iter,S,K_BTC1[k])/(b-a)#+S-K*np.exp(-mu_GBM[crypto]*T_days)
                #iv_GBM = implied_volatility_put(putVal_GBM,S,K_BTC1[k],T_BTC1[t],mu_GBM[crypto],max_iterations=int(1e2))
                # +S-K*np.exp(-mu_heston[crypto]*T_days)
                putVal_Heston = f(M, a, b, T_days, fi_heston,
                                  param_iter, S, K_BTC1[k], v0=v0_sv)/(b-a)
                # callVal_SVCJ=f(M,a,b,T_days,fi_svcj,crypto,v0=v0_svcj)/(b-a)*np.exp(-mu_svcj[crypto]*T_days)#+S-K*np.exp(-mu_svcj[crypto]*T_days)

                # optionDict_BTC1[i]=[callVal_BS,callVal_GBM,callVal_GBMJ,callVal_Heston,callVal_SVCJ,T,K]

                putVall_array[t, k] = putVal_Heston

    RMSE = np.sqrt(np.nanmean((putVall_array-putVals_real_7June_BTC)**2))
    RMSE_array[i] = RMSE
print(np.argpartition(RMSE_array, 10)[:10])
print(params[np.argpartition(RMSE_array, 10)[:10][-1]],
      params[np.argpartition(RMSE_array, 10)[:10][-2]])

# for model,drift in {'GBM':mu_GBM}.items():
#     imp = [implied_volatility_call(BTC1.loc[x][model],S,BTC1.loc[x]['K'],BTC1.loc[x]['T']/365,drift[crypto],max_iterations=int(1e2)) for x in BTC1.index]
#     BTC1[model+" IV"]=imp

# for model,drift in {'Analytical BS':mu_GBM,'GBM':mu_GBM,'GBMJ':mu_GBMJ,'Heston':mu_heston,'SVCJ':mu_svcj}.items():
#     imp = [implied_volatility_call(BTC1.loc[x][model],S,BTC1.loc[x]['K'],BTC1.loc[x]['T']/365,drift[crypto],max_iterations=int(1e2)) for x in BTC1.index]
#     BTC1[model+" IV"]=imp
