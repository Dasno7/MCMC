# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:51:02 2021

@author: Dasno7
"""
import numpy as np
cryptos = ['Bitcoin']#,'LINK','ETH','ADA']
mu ={'Bitcoin': 0.02029249494235486, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061, 'ADA':-0.003431383092799574 }
s2V = {'Bitcoin': 0.04033712521749577, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012, 'ADA':0.13299513034831426 }
alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
kappa ={'Bitcoin':(1-0.9578989881256078) , 'LINK':(1-beta['LINK']), 'ETH':(1- 0.9114176612486856), 'ADA': (1-beta['ADA'])}
theta ={'Bitcoin':  0.027300605314312377/kappa['Bitcoin'], 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': 0.0927768090812799/kappa['ETH'], 'ADA': alpha['ADA']/kappa['ADA']}

T=100
S=34
v=1.5/np.sqrt(365)

a=0.01
b=10000
K=36

M=int(5e5)
for crypto in cryptos:
    def fi_heston(x):
        d = np.sqrt(s2V[crypto]*(1j*x+x**2)+(rho[crypto]*s2V[crypto]**(0.5)*1j*x-kappa[crypto])**2)
        c=(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)/(kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x+d)
        beta = (kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*(1-np.exp(-d*T))/(s2V[crypto]*(1-c*np.exp(-d*T)))
        alpha = kappa[crypto]*theta[crypto]/s2V[crypto]*((kappa[crypto]-s2V[crypto]**(0.5)*rho[crypto]*1j*x-d)*T-2*np.log((1-c*np.exp(-d*T))/(1-c)))
        m = np.log(S)+mu[crypto]*T
        
        characFunc = np.exp(1j*x*m+alpha+beta*v)
        return characFunc
    
    def gn(n):
        hn= n*np.pi/(b-a)
        g = (np.exp(a)-K/hn*np.sin(hn*(a-np.log(K)))-K*np.cos(hn*(a-np.log(K))))/(1+(hn)**2)
        
        return g
    
    def f(M):
        g0 = K*(np.log(K)-a-1)+np.exp(a)
        putEst = g0+ np.array([2*gn(i)*np.exp(-np.pi*a*i*1j/(b-a))*fi_heston(np.pi*i/(b-a)) for i in range(1,M+1)]).sum()
        return putEst.real/(b-a)
    
    test = np.array([f(t) for t in range(1,M)])
    # def cn(n):
    #     c = y*np.exp(-1j*2*n*np.pi*time/period)
    #     return c.sum()/c.size
    
    # def f(x, Nh):
    #     f = np.array([2*cn(i)*np.exp(1j*2*i*np.pi*x/period) for i in range(1,Nh+1)])
    #     return f.sum()
    
    #y2 = np.array([f(t,50).real for t in time])
    
    # plot(time, y)
    # plot(time, y2)