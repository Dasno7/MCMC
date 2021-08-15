# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:46:55 2021

@author: Dasno7
"""

import pandas as pd
import numpy as np
import scipy.stats as stats 
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from matplotlib import rc

#BTC price data
btc_data = pd.read_csv("Crypto Data Repo/Bitcoin Historical Data.csv")
btc_price = np.flip(pd.Series(btc_data['Price'].str.replace(',','').astype(float)))
btc_price.index = np.flip(btc_data['Date'])
Y = np.log(btc_price/btc_price.shift(1))[1:]*np.sqrt(365) #return process Y
T = Y.shape[0] #T of process
#format_str = "%b %d, %Y"
#Y.index = [datetime.datetime.strptime(Y.index[j],format_str) for j in range(T)]
J = (abs(Y)-np.mean(Y)>2*np.std(Y)).astype(int) # starting value of jumps

def get_hyperGamma(annualized_returns, x):
    Dayx_vol = np.zeros(annualized_returns.shape[0])    
    Day1_vol = np.zeros(annualized_returns.shape[0]) 
    for i in range(annualized_returns[x:].shape[0]):
        Dayx_vol[i] = np.std(annualized_returns[i:(x-1+i)])
        Day1_vol[i] = Dayx_vol[i]/np.sqrt(x)
    mu_gamma = np.var(annualized_returns)
    var_gamma = np.var(Day1_vol)*np.sqrt(365)
    a= mu_gamma**2/var_gamma+2
    b= (a-1)*mu_gamma
    return a,b

def get_hyperBeta(annualized_returns):
    J = (abs(annualized_returns)-np.mean(annualized_returns)>2*np.std(annualized_returns)).astype(int) 
    mean = np.sum(J)/J.shape[0]
    var = 2*40/((42**2)*43) #approx var used for lambda
    a = ((1-mean)*mean**2)/(var*(mean+1))
    b = a*(1-mean)/mean
    return a,b
    

#Prior distribution hyperparameter values
a=np.mean(Y);A=np.var(Y);
f,F = get_hyperGamma(Y,30);k,K = get_hyperBeta(Y)
e = np.mean(Y[np.where(J==1)[0]]); E = np.var(Y[np.where(J==1)[0]])
h,H = get_hyperGamma(Y[np.where(J==1)[0]],int(np.round(np.where(J==1)[0].shape[0]/5)))

#starting values parameters
sigma2 =F/(f-2) #initial values jumps
sigma2_j = E
xi = np.random.normal(e,E**0.5,T)*J

#Initialize saving parameters for MC
mtot=0;mtot2=0 #drawing mu
sigma2tot=0;sigma2tot2=0 #drawing sigma2_y
Jtot=0 #drawing J's
xitot=0;lambtot=0;lambtot2=0 #drawing lambda
m_jtot=0;m_jtot2=0;sigma2_jtot=0;sigma2_jtot2=0

N=100000;burn=30000 #Number of draws and burn period
sig2_save = np.zeros(N)
mu_save = np.zeros(N)
lamb_save = np.zeros(N)
muJ_save = np.zeros(N)
sigma2J_save = np.zeros(N)
for i in tqdm(range(N)):
    #Draw mu (derive A_star)
    A_star = 1/(T/sigma2+1/A)
    a_star = A_star*(np.sum(Y-xi*J)/sigma2+a/A)
    m=np.random.normal(a_star,A_star**0.5)
    
    if i>burn:
        mtot += m
        mtot2 =+ m**2
    mu_save[i]=m
    
    #Draw sigma2
    f_star = T +f
    F_star = F+np.sum((Y-m-xi*J)**2)
    sigma2 = stats.invgamma.rvs(f_star, scale=F_star)
    if i > burn:
        sigma2tot += sigma2
        sigma2tot2 += sigma2**2
    sig2_save[i]=sigma2
    
    #Draw mu_J
    E_star = 1/(T/sigma2_j+1/E)
    e_star = E_star*(np.sum(xi)/sigma2_j+e/E)
    m_j = np.random.normal(e_star,E_star**0.5)
    if i>burn:
        m_jtot += m_j
        m_jtot2 += m_j**2
    muJ_save[i]=m_j
    
    #Draw sigma2_j
    h_star = h +T
    H_star = H+np.sum((xi-m_j)**2)
    sigma2_j = stats.invgamma.rvs(h_star, scale=H_star)
    if i>burn:
        sigma2_jtot += sigma2_j
        sigma2_jtot2 += sigma2_j**2 
    sigma2J_save[i] = sigma2_j    
    
    #lambda
    k_star = k + np.sum(J)
    K_star = K+T-np.sum(J)
    lamb = np.random.beta(k_star,K_star)
    if i>burn:
        lambtot += lamb
        lambtot2 += lamb**2
    lamb_save[i] = lamb
    
    #Jumps
    eY0 = Y - m
    eY1 = eY0 - xi
    p1 = lamb*np.exp(-(eY1)**2/(2*sigma2))
    p2 = (1-lamb)*np.exp(-(eY0)**2/(2*sigma2))
    p = p1/(p1+p2)
    J = np.random.binomial(1,p,T)
    if i >burn:
        Jtot += J
        
    #Jump size
    Jindex = np.where(J==1)[0]
    Jnot = np.where(J==0)[0]
    xi[Jnot] = np.random.normal(m_j,sigma2_j**0.5,xi[Jnot].size)
    if Jindex.size != 0:
        for j in Jindex:
            Z_star = 1/(1/sigma2_j+1/sigma2)
            z_star = Z_star*((Y[j]-m)/sigma2+m_j/sigma2_j)
            xi[j] = np.random.normal(z_star,Z_star**0.5) 
    if i > burn:
        xitot += xi
    
    
#Monte carlo estimates
m=mtot/(N-burn)        
sigma2=sigma2tot/(N-burn)
m_j = m_jtot/(N-burn)
sigma2_j = sigma2_jtot/(N-burn)
lamb = lambtot/(N-burn)
J_result = np.round(Jtot/(N-burn))
xi_result = xitot/(N-burn)
print(m,sigma2,lamb,m_j,sigma2_j,np.sum(J),np.mean(xi))
        

sns.set_theme(style='darkgrid')
ax = pd.Series(J_result*xi_result,index=Y.index).plot(figsize=(12,3.75)\
                                                 ,ylabel="Annualized log return")
ax.set(xlabel=None)
ax.set_title('Bitcoin return jumps',fontdict= { 'fontsize': 18, 'fontweight':'bold'})

fig, ax = plt.subplots(1,2,figsize=(12,3.75),constrained_layout=True)
sns.lineplot(data=sigma2J_save,ax=ax[0])
ax[0].set_title(r'$\sigma_J^2$',fontdict= { 'fontsize': 18, 'fontweight':'bold'})
ax[0].set(xlabel='iters')
sns.lineplot(data=lamb_save,ax=ax[1])  
ax[1].set(xlabel='iters')  
ax[1].set_title(r'$\lambda$',fontdict= { 'fontsize': 18, 'fontweight':'bold'})  

        

    
