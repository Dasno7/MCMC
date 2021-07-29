# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:51:02 2021

@author: Dasno7
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt

cryptos = ['Bitcoin']#,'LINK','ETH','ADA']

# # Geometric Brownian Motion
# m ={'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642, 'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}#/np.sqrt(365)
# sigma2_y = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903, 'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}#/np.sqrt(365)

# S=57663
# K=38000
# T=1/365

# def putOptionPriceAnalytical(S0, K, T, r, sigma):
#     d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     N_d1 = stats.norm.cdf(-d1)
#     N_d2 = stats.norm.cdf(-d2)

#     europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
#     return europePutAnalytical

# for crypto in cryptos:
#     price = putOptionPriceAnalytical(S,K,T,m[crypto],sigma2_y[crypto]**0.5)
#     #price = putOptionPriceAnalytical(S,K,T,0,sigma2_y[crypto]**0.5)
#     print(price)

# # # Heston model
# # mu ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }
# # s2V = {'Bitcoin': 0.04033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012, 'ADA':0.13299513034831426 }
# # alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
# # beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
# # rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
# # kappa ={'Bitcoin':(1-0.9578989881256078) , 'LINK':(1-beta['LINK']), 'ETH':(1- 0.9114176612486856), 'ADA': (1-beta['ADA'])}
# # theta ={'Bitcoin':  0.027300605314312377/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': 0.0927768090812799/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}


# # SVCJ model

# # mu ={'Bitcoin': 0.021644682489966826, 'LINK':0.03645939802343439, 'ETH':  0.015283256840018638, 'ADA':-0.007316876526135481 }#/np.sqrt(365)
# # s2V ={'Bitcoin': 0.06756870157684675, 'LINK':0.40446812600230947, 'ETH':  0.17931239389025172, 'ADA':0.26116042635375053 }#/np.sqrt(365)
# # alpha ={'Bitcoin': 0.043193377631366804, 'LINK': 0.3855999310971581, 'ETH':   0.16119100120638954, 'ADA':0.2662085060789083 }
# # beta ={'Bitcoin': 0.9263876734631863 , 'LINK':0.7873614865856806 , 'ETH':  0.8491499506560084, 'ADA':0.7995479771756661 }
# # rho ={'Bitcoin': -0.09119104913669938, 'LINK':-0.10825275611533464, 'ETH':  -0.12041305370319762, 'ADA':-0.09317621524732624  }
# # kappa ={'Bitcoin':(1-beta['Bitcoin']) , 'LINK':(1-beta['LINK']), 'ETH':(1- beta['ETH']), 'ADA': (1-beta['ADA'])}
# # theta ={'Bitcoin':  alpha['Bitcoin']/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': alpha['ETH']/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}
# # mJ = {'Bitcoin': 0.9406651072820391, 'LINK':3.098086827567313 , 'ETH':  2.520408386900445, 'ADA':2.068795155113863 }
# # s2J = {'Bitcoin': 8.42181066157726 , 'LINK':27.90186571275088, 'ETH':  13.099952280133733, 'ADA':17.140749486332407 }
# # lambd = {'Bitcoin': 0.0053092423361834245, 'LINK':0.0025708713027121845, 'ETH':  0.002263724926372133, 'ADA':0.002458274763338694 }
# # mV = {'Bitcoin': 2.30966833400468, 'LINK': 2.002095925724665, 'ETH':  2.192366611567432 , 'ADA':1.986970474497482 }
# # rhoJ = {'Bitcoin': -0.05277382444932916, 'LINK':-0.0850455673163998 , 'ETH':  -0.16674148085665919, 'ADA':-0.16623534607963417 }


# # GBMJ model
# # m ={'Bitcoin': 0.027147875324741517, 'LINK': 0.053533829570666804, 'ETH': 0.039328550620623515, 'ADA': 0.01790843001866173}#/np.sqrt(365)
# # sigma2_y = {'Bitcoin':0.5640435894769665, 'LINK': 1.3009875076748425, 'ETH': 1.019470509187793, 'ADA': 0.8920809742512347}#/np.sqrt(365)
# # lamb = {'Bitcoin':0.05986714696291274, 'LINK': 0.07302822526492724, 'ETH': 0.030574931256999466, 'ADA': 0.11611984301428516}
# # m_j = {'Bitcoin':0.06587865926909922, 'LINK': 0.03090872955154435, 'ETH': -0.059820561646782816, 'ADA': -0.030874433257736224}
# # sigma2_j = {'Bitcoin': 1.9446523435616887, 'LINK':5.614355345655559, 'ETH': 3.1867898469787788, 'ADA': 3.4154570830278237}



# v=0.6945836313141901#/np.sqrt(365)

# # BS ranges
# c1=m[crypto]*T
# c2 = sigma2_y[crypto]*T
# L=300

# a=c1-L*np.sqrt(c2)
# b=c1+L*np.sqrt(c2)

# # Other ranges
# # a=-2
# # b=2
# #lamb['ETH']=0

# M=int(1e3)
# for crypto in cryptos:
#     # def fi_heston(x):
#     #     d = np.sqrt(s2V[crypto]*(1j*x+(x)**2)+(rho[crypto]*s2V[crypto]**(0.5)*1j*x-kappa[crypto])**2)
#     #     c=(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)/(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x+d)
#     #     beta = (kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*(1-np.exp(-d*T))/(s2V[crypto]*(1-c*np.exp(-d*T)))
#     #     alpha = kappa[crypto]*theta[crypto]/s2V[crypto]*((kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
#     #     m = np.log(S)+mu[crypto]*T
        
#     #     characFunc = np.exp(1j*x*m+alpha+beta*v)
#     #     return characFunc
    
#     # def fi_svcj(x):
#     #     f=rho[crypto]*s2V[crypto]**(0.5)*1j*x-kappa[crypto];g=1-rhoJ[crypto]*mV[crypto]*1j*x;h=(1j*x+(x)**2)
        
#     #     d = np.sqrt(s2V[crypto]*h+f**2)
#     #     c=(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)/(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x+d)
#     #     beta = (kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*(1-np.exp(-d*T))/(s2V[crypto]*(1-c*np.exp(-d*T)))
#     #     alpha = kappa[crypto]*theta[crypto]/s2V[crypto]*((kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
#     #     m = np.log(S)+mu[crypto]*T
#     #     mu_bar = np.exp(mJ[crypto]/365+0.5*s2J[crypto]/np.sqrt(365))/(1-rhoJ[crypto]*mV[crypto])-1
#     #     #mu_bar = np.exp(mJ[crypto]+0.5*s2J[crypto])/(1-rhoJ[crypto]*mV[crypto])-1
        
#     #     dzeta = (d-f)*T/((d-f)*g+mV[crypto]*h)-(2*mV[crypto]*h)/((d*g)**2-(f*g-mV[crypto]*h)**2)*np.log(1-((d+f)*g-mV[crypto]*h)*(1-np.exp(-d*T))/(2*d*g))
#     #     Theta =  np.exp(1j*x*mJ[crypto]+(x*1j)**2*0.5*s2J[crypto])*dzeta
        
#     #     characFunc = np.exp(1j*x*m+alpha+beta*v-lambd[crypto]*T*(1+mu_bar*1j*x)+lambd[crypto]*Theta)
#     #     return characFunc    
  
    
#     def fi_BS(x):
#         m_bs = np.log(S)+(m[crypto]-0.5*sigma2_y[crypto])*T
        
#         characFunc = np.exp(1j*x*m_bs-0.5*x**2*sigma2_y[crypto]*T)
#         return characFunc
    
#     # def fi_GBMJ(x):
#     #     m_bs = np.log(S)+(m[crypto]-0.5*sigma2_y[crypto])*T
#     #     mu_bar = np.exp(m_j[crypto]+0.5*sigma2_j[crypto])-1
#     #     Theta = np.exp(m_j[crypto]*x*1j+0.5*sigma2_j[crypto]*(x*1j)**2)*T
        
#     #     characFunc = np.exp(1j*x*m_bs-0.5*x**2*sigma2_y[crypto]*T-lamb[crypto]*T*(1+mu_bar*1j*x)+lamb[crypto]*Theta)
#     #     return characFunc
    
#     def gn(n):
#         hn= n*np.pi/(b-a)
#         g = (np.exp(a)-K/hn*np.sin(hn*(a-np.log(K)))-K*np.cos(hn*(a-np.log(K))))/(1+(hn)**2)
        
#         return g
    
#     def f(M):
#         g0 = K*(np.log(K)-a-1)+np.exp(a)
#         putEst = g0+ np.array([2*gn(i)*np.exp(-np.pi*a*i*1j/(b-a))*fi_BS(np.pi*i/(b-a)) for i in range(1,M+1)]).sum()
#         return putEst.real/(b-a)*np.exp(-m[crypto]*T) #change mu depending on model
    
#     putValue=f(M)
#     print(putValue)
#     print('Call option BS:',price+S-K*np.exp(-m[crypto]*T))
#     print('Call option non-BS:',putValue+S-K*np.exp(-m[crypto]*T))
#     test = np.array([f(t) for t in range(1,M)])
#     plt.plot(test)
#     plt.xlabel('Number of series expansions')
#     plt.ylabel('Put value in thousands dollars')
#     plt.title('Fourier series convergence')
#-----------------------------------------------------------------------------------------------------    
# Param
mu_GBM ={'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642, 'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}
sigma2_y_GBM = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903, 'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}
mu_GBMJ ={'Bitcoin': 0.027147875324741517, 'LINK': 0.053533829570666804, 'ETH': 0.039328550620623515, 'ADA': 0.01790843001866173}
mu_heston ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }
mu_svcj ={'Bitcoin': 0.021644682489966826, 'LINK':0.03645939802343439, 'ETH':  0.015283256840018638, 'ADA':-0.007316876526135481 } 
    
#-----------------------------------------------------------------------------------------------------
# Functions

def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical

def fi_GBM(x,crypto,T):
    # GBM model parameters
    m ={'Bitcoin': 0.02817135116188635, 'LINK': 0.06792776181895642, 'ETH': 0.028163787262985164, 'ADA': 0.026269044494565446}
    sigma2_y = {'Bitcoin': 0.6945836313141901, 'LINK': 1.8381271048871903, 'ETH': 1.123005248647937, 'ADA': 1.3440624591240018}

    m_bs = np.log(S)+(m[crypto]-0.5*sigma2_y[crypto])*T
    
    characFunc = np.exp(1j*x*m_bs-0.5*x**2*sigma2_y[crypto]*T)
    return characFunc

def fi_GBMJ(x,crypto,T):
    # GBMJ model parameters
    m ={'Bitcoin': 0.027147875324741517, 'LINK': 0.053533829570666804, 'ETH': 0.039328550620623515, 'ADA': 0.01790843001866173}#/np.sqrt(365)
    sigma2_y = {'Bitcoin':0.5640435894769665, 'LINK': 1.3009875076748425, 'ETH': 1.019470509187793, 'ADA': 0.8920809742512347}#/np.sqrt(365)
    lamb = {'Bitcoin':0.05986714696291274, 'LINK': 0.07302822526492724, 'ETH': 0.030574931256999466, 'ADA': 0.11611984301428516}
    m_j = {'Bitcoin':0.06587865926909922, 'LINK': 0.03090872955154435, 'ETH': -0.059820561646782816, 'ADA': -0.030874433257736224}
    sigma2_j = {'Bitcoin': 1.9446523435616887, 'LINK':5.614355345655559, 'ETH': 3.1867898469787788, 'ADA': 3.4154570830278237}

    m_bs = np.log(S)+(m[crypto]-0.5*sigma2_y[crypto])*T
    mu_bar = np.exp(m_j[crypto]/365+0.5*sigma2_j[crypto]/np.sqrt(365))-1
    Theta = np.exp(m_j[crypto]*x*1j+0.5*sigma2_j[crypto]*(x*1j)**2)*T
    
    characFunc = np.exp(1j*x*m_bs-0.5*x**2*sigma2_y[crypto]*T-lamb[crypto]*T*(1+mu_bar*1j*x)+lamb[crypto]*Theta)
    return characFunc

def fi_heston(x,v,crypto,T):
    # Heston model parameters
    mu ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }
    s2V = {'Bitcoin': 0.04033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012, 'ADA':0.13299513034831426 }
    alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
    beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
    rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
    kappa ={'Bitcoin':(1-0.9578989881256078) , 'LINK':(1-beta['LINK']), 'ETH':(1- 0.9114176612486856), 'ADA': (1-beta['ADA'])}
    theta ={'Bitcoin':  0.027300605314312377/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': 0.0927768090812799/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}


    d = np.sqrt(s2V[crypto]*(1j*x+(x)**2)+(rho[crypto]*s2V[crypto]**(0.5)*1j*x-kappa[crypto])**2)
    c=(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)/(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x+d)
    beta = (kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*(1-np.exp(-d*T))/(s2V[crypto]*(1-c*np.exp(-d*T)))
    alpha = kappa[crypto]*theta[crypto]/s2V[crypto]*((kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
    m = np.log(S)+mu[crypto]*T
    
    characFunc = np.exp(1j*x*m+alpha+beta*v)
    return characFunc 

def fi_svcj(x,v,crypto,T):
    # SVCJ model parameters
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

    f=rho[crypto]*s2V[crypto]**(0.5)*1j*x-kappa[crypto];g=1-rhoJ[crypto]*mV[crypto]*1j*x;h=(1j*x+(x)**2)
    
    d = np.sqrt(s2V[crypto]*h+f**2)
    c=(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)/(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x+d)
    beta = (kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*(1-np.exp(-d*T))/(s2V[crypto]*(1-c*np.exp(-d*T)))
    alpha = kappa[crypto]*theta[crypto]/s2V[crypto]*((kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
    m = np.log(S)+mu[crypto]*T
    mu_bar = np.exp(mJ[crypto]/365+0.5*s2J[crypto]/np.sqrt(365))/(1-rhoJ[crypto]*mV[crypto])-1
    #mu_bar = np.exp(mJ[crypto]+0.5*s2J[crypto])/(1-rhoJ[crypto]*mV[crypto])-1
    
    dzeta = (d-f)*T/((d-f)*g+mV[crypto]*h)-(2*mV[crypto]*h)/((d*g)**2-(f*g-mV[crypto]*h)**2)*np.log(1-((d+f)*g-mV[crypto]*h)*(1-np.exp(-d*T))/(2*d*g))
    Theta =  np.exp(1j*x*mJ[crypto]+(x*1j)**2*0.5*s2J[crypto])*dzeta
    
    characFunc = np.exp(1j*x*m+alpha+beta*v-lambd[crypto]*T*(1+mu_bar*1j*x)+lambd[crypto]*Theta)
    return characFunc 


def gn(n,a,b):
    hn= n*np.pi/(b-a)
    g = (np.exp(a)-K/hn*np.sin(hn*(a-np.log(K)))-K*np.cos(hn*(a-np.log(K))))/(1+(hn)**2)
    
    return g

def f(M,a,b,T,fi,crypto,**v0):
    g0 = K*(np.log(K)-a-1)+np.exp(a)
    if v0:
        putEst = g0+ np.array([2*gn(i,a,b)*np.exp(-np.pi*a*i*1j/(b-a))*fi(np.pi*i/(b-a),v0['v0'],crypto,T) for i in range(1,M+1)]).sum()
    else:
        putEst = g0+ np.array([2*gn(i,a,b)*np.exp(-np.pi*a*i*1j/(b-a))*fi(np.pi*i/(b-a),crypto,T) for i in range(1,M+1)]).sum()
    return putEst.real #change mu depending on model

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
#-----------------------------------------------------------------------------------------------------

#BTC scenario 1: 7 June 2021 S_0=57663

crypto = 'Bitcoin'
S=57663
K_BTC1 = np.array([38000, 32000, 40000, 53000, 36000, 42000, 34000, 66000, 48000,
        62000, 68000, 44000, 80000, 84000, 58000, 46000, 50000, 70000,
        64000, 56000, 59000, 54000, 60000, 78000, 55000, 52000, 74000,
        57000, 76000, 72000, 51000, 63000, 61000, 65000, 67000, 88000,
        4000, 24000, 14000, 26000, 28000, 16000, 120000, 10000, 18000,
        140000, 12000, 100000, 7000, 8000, 90000, 85000, 9000, 20000,
        15000, 22000, 17000, 45000, 75000, 6000, 13000, 82000, 30000,
        92000, 86000, 5000, 11000])
T_BTC1 = np.array([ 4, 19, 47, 6, 12, 26, 103, 5, 194, 292, 75])

v0=0.6945836313141901
M=int(2**10)

i=0; optionDict_BTC1= {}
for K in K_BTC1:
    for T in T_BTC1:
        T_days=T/365
        c1=mu_GBM[crypto]*T_days
        c2 = sigma2_y_GBM[crypto]*T_days
        L=300

        a=c1-L*np.sqrt(c2)  
        b=c1+L*np.sqrt(c2)
        callVal_BS = putOptionPriceAnalytical(S,K,T_days,mu_GBM[crypto],sigma2_y_GBM[crypto]**0.5)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_GBMJ=f(M,a,b,T_days,fi_GBMJ,crypto)/(b-a)*np.exp(-mu_GBMJ[crypto]*T_days)+S-K*np.exp(-mu_GBMJ[crypto]*T_days)
        callVal_GBM=f(M,a,b,T_days,fi_GBM,crypto)/(b-a)*np.exp(-mu_GBM[crypto]*T_days)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_Heston=f(M,a,b,T_days,fi_heston,crypto,v0=v0)/(b-a)*np.exp(-mu_heston[crypto]*T_days)+S-K*np.exp(-mu_heston[crypto]*T_days)
        callVal_SVCJ=f(M,a,b,T_days,fi_svcj,crypto,v0=v0)/(b-a)*np.exp(-mu_svcj[crypto]*T_days)+S-K*np.exp(-mu_svcj[crypto]*T_days)
                
        optionDict_BTC1[i]=[callVal_BS,callVal_GBM,callVal_GBMJ,callVal_Heston,callVal_SVCJ,T,K]
        i+=1

BTC1 = pd.DataFrame(optionDict_BTC1).T
BTC1.columns = ['Analytical BS','GBM','GBMJ','Heston','SVCJ','T','K']

for model,drift in {'Analytical BS':mu_GBM,'GBM':mu_GBM,'GBMJ':mu_GBMJ,'Heston':mu_heston,'SVCJ':mu_svcj}.items():
    imp = [implied_volatility_call(BTC1.loc[x][model],S,BTC1.loc[x]['K'],BTC1.loc[x]['T']/365,drift[crypto],max_iterations=int(1e4)) for x in BTC1.index]   
    BTC1[model+" IV"]=imp
    

#BTC scenario 2: 14 March 2021 S_0=35198

crypto = 'Bitcoin'
S=35198
K_BTC2 = np.array([42000, 37000, 33000, 39000, 40000, 30000, 34000, 36000, 32000,
        38000, 35000, 41000, 45000, 26000, 55000, 20000, 46000, 75000,
        80000, 44000, 48000, 85000, 60000, 56000, 25000, 58000, 50000,
        54000, 31000, 70000, 28000, 52000, 65000, 43000, 24000, 14000,
        16000, 18000, 12000, 64000, 15000, 8000, 68000, 10000, 22000,
        72000, 6000, 76000])
T_BTC2 = np.array([2, 11, 18, 4, 1, 109, 53, 207, 291, 81])

v0=0.6945836313141901
M=int(2**6)

i=0; optionDict_BTC2= {}
for K in K_BTC2:
    for T in T_BTC2:
        T_days=T/365
        c1=mu_GBM[crypto]*T_days
        c2 = sigma2_y_GBM[crypto]*T_days
        L=300

        a=c1-L*np.sqrt(c2)  
        b=c1+L*np.sqrt(c2)
        callVal_BS = putOptionPriceAnalytical(S,K,T_days,mu_GBM[crypto],sigma2_y_GBM[crypto]**0.5)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_GBMJ=f(M,a,b,T_days,fi_GBMJ,crypto)/(b-a)*np.exp(-mu_GBMJ[crypto]*T_days)+S-K*np.exp(-mu_GBMJ[crypto]*T_days)
        callVal_GBM=f(M,a,b,T_days,fi_GBM,crypto)/(b-a)*np.exp(-mu_GBM[crypto]*T_days)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_Heston=f(M,a,b,T_days,fi_heston,crypto,v0=v0)/(b-a)*np.exp(-mu_heston[crypto]*T_days)+S-K*np.exp(-mu_heston[crypto]*T_days)
        callVal_SVCJ=f(M,a,b,T_days,fi_svcj,crypto,v0=v0)/(b-a)*np.exp(-mu_svcj[crypto]*T_days)+S-K*np.exp(-mu_svcj[crypto]*T_days)
                
        optionDict_BTC2[i]=[callVal_BS,callVal_GBM,callVal_GBMJ,callVal_Heston,callVal_SVCJ,T,K]
        i+=1

BTC2 = pd.DataFrame(optionDict_BTC2).T
BTC2.columns = ['Analytical BS','GBM','GBMJ','Heston','SVCJ','T','K']

for model,drift in {'Analytical BS':mu_GBM,'GBM':mu_GBM,'GBMJ':mu_GBMJ,'Heston':mu_heston,'SVCJ':mu_svcj}.items():
    imp = [implied_volatility_call(BTC2.loc[x][model],S,BTC2.loc[x]['K'],BTC2.loc[x]['T']/365,drift[crypto],max_iterations=int(1e4)) for x in BTC2.index]   
    BTC2[model+" IV"]=imp


#BTC scenario 3: 16 November 2020 S_0=16052

crypto = 'Bitcoin'
S=16053
K_BTC3 = np.array([15250, 14250, 15750, 15125, 16125, 15500, 14500, 14625, 16625,
       15375, 15000, 16375, 16500, 13875, 16000, 14125, 14750, 16750,
       14375, 16250, 14875, 15875, 14000, 15625, 4000, 8000, 6000, 12750,
       13000, 5000, 9000, 11000, 10000, 10500, 10750, 7000, 11500, 10250,
       12000, 3000, 13500, 12250, 11750, 12500, 13750, 11250, 13250,
       16875, 17000, 17250, 18000, 18500, 18250, 17750, 17500, 17375,
       17125, 24000, 36000, 20000, 19500, 19750, 19000, 40000, 20750,
       19250, 17625, 18750, 28000, 20500, 32000, 22000, 17875, 20250,
       21000])
T_BTC3 = np.array([130, 74, 11, 4, 39, 221,  18,2])

v0=0.6945836313141901
M=int(2**6)

i=0; optionDict_BTC3= {}
for K in K_BTC3:
    for T in T_BTC3:
        T_days=T/365
        c1=mu_GBM[crypto]*T_days
        c2 = sigma2_y_GBM[crypto]*T_days
        L=300

        a=c1-L*np.sqrt(c2)  
        b=c1+L*np.sqrt(c2)
        callVal_BS = putOptionPriceAnalytical(S,K,T_days,mu_GBM[crypto],sigma2_y_GBM[crypto]**0.5)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_GBMJ=f(M,a,b,T_days,fi_GBMJ,crypto)/(b-a)*np.exp(-mu_GBMJ[crypto]*T_days)+S-K*np.exp(-mu_GBMJ[crypto]*T_days)
        callVal_GBM=f(M,a,b,T_days,fi_GBM,crypto)/(b-a)*np.exp(-mu_GBM[crypto]*T_days)+S-K*np.exp(-mu_GBM[crypto]*T_days)
        callVal_Heston=f(M,a,b,T_days,fi_heston,crypto,v0=v0)/(b-a)*np.exp(-mu_heston[crypto]*T_days)+S-K*np.exp(-mu_heston[crypto]*T_days)
        callVal_SVCJ=f(M,a,b,T_days,fi_svcj,crypto,v0=v0)/(b-a)*np.exp(-mu_svcj[crypto]*T_days)+S-K*np.exp(-mu_svcj[crypto]*T_days)
                
        optionDict_BTC3[i]=[callVal_BS,callVal_GBM,callVal_GBMJ,callVal_Heston,callVal_SVCJ,T,K]
        i+=1

BTC3 = pd.DataFrame(optionDict_BTC3).T
BTC3.columns = ['Analytical BS','GBM','GBMJ','Heston','SVCJ','T','K']

for model,drift in {'Analytical BS':mu_GBM,'GBM':mu_GBM,'GBMJ':mu_GBMJ,'Heston':mu_heston,'SVCJ':mu_svcj}.items():
    imp = [implied_volatility_call(BTC3.loc[x][model],S,BTC3.loc[x]['K'],BTC3.loc[x]['T']/365,drift[crypto],max_iterations=int(1e4)) for x in BTC3.index]   
    BTC3[model+" IV"]=imp
    
#ETH scenario 1: 7 June 2021 S_0=1795
crypto = 'ETH'
S=1795

K_ETH1=np.array([2150, 2400, 2550, 2600, 2300, 2900, 2500, 2750, 2450, 2350, 2700,
       2000, 2200, 2650, 2250, 2100, 2800, 2850, 3000, 5200, 3500, 5800,
       5400, 4500, 4800, 4000, 3750, 4100, 6000, 1200, 3200, 3850, 5000,
       1900, 3100, 4050, 1400, 3550, 3300, 6200, 4400, 3700, 1600, 1800,
       4300, 4200, 3650, 1700, 3900, 4600, 5600, 3950, 4150, 3800, 3600,
       3400, 3450, 2950, 640, 1280, 800, 2880, 600, 560, 380, 960, 2240,
       3840, 1920, 360, 280, 1000, 3680, 3520, 2720, 1120, 1440, 340, 480,
       2560, 1500, 320, 1040, 2080, 200, 400, 240, 520, 440, 4480, 5500,
       720, 4320, 4160])

T_ETH1=np.array([2, 11, 109, 18, 53, 207, 4, 81, 1, 291])

#ETH scenario 2: 14 March 2021 S_0=2647
crypto = 'ETH'
S=2647

K_ETH2=np.array([1120, 1740, 1240, 1200, 2040, 1640, 1400, 1520, 2240, 2640, 2120,
       1840, 1600, 1440, 1860, 1960, 2200, 1800, 2320, 1680, 1360, 2080,
       2400, 2560, 2000, 1320, 1900, 1760, 1780, 1280, 1920, 1880, 2280,
       2480, 1720, 2800, 2720, 2880, 2160, 1560, 1820, 1480, 1660, 1700,
       1620, 1940, 1980, 2060, 2020, 440, 180, 640, 720, 300, 800, 600,
       3200, 3000, 560, 380, 960, 3520, 1950, 3840, 420, 360, 280, 4160,
       1850, 340, 320, 2100, 160, 520, 480, 2600, 140, 2500, 460, 220,
       1040, 200, 1000, 400, 260, 240, 1750, 4000, 1500, 880])

T_ETH2=np.array([2, 3, 4, 6, 5, 19, 75, 47, 12, 194, 103, 292, 26])
#ETH scenario 3: 16 November 2020 S_0=458
crypto = 'ETH'
S=458

K_ETH3=np.array([470, 500, 430, 460, 390, 480, 420, 410, 400, 490, 450, 440, 180,
       320, 300, 340, 240, 330, 80, 380, 280, 220, 120, 160, 360, 140,
       290, 100, 310, 40, 200, 260, 350, 370, 60, 530, 550, 510, 540, 560,
       520, 580, 800, 720, 640, 630, 960, 600, 1120, 590, 620, 570, 880,
       610])

T_ETH3=np.array([130, 39, 11, 4, 221, 74, 18, 1, 0, 2])
