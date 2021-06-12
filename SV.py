# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:37:37 2021

@author: Dasno7
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
# from math import gamma
#from multiprocess import Pool


#BTC price data
btc_data = pd.read_csv("Crypto Data Repo/Bitcoin Historical Data.csv")
btc_price = np.flip(pd.Series(btc_data['Price'].str.replace(',','').astype(float)))
btc_price.index = np.flip(btc_data['Date'])
Y = np.log(btc_price/btc_price.shift(1))[1:]*np.sqrt(365) #return process Y
T = Y.shape[0] #T of process

def get_hyperGamma(annualized_returns, x):
    Dayx_vol = np.zeros(annualized_returns.shape[0])    
    Day1_vol = np.zeros(annualized_returns.shape[0]) 
    for i in range(annualized_returns[x:].shape[0]):
        Dayx_vol[i] = np.std(annualized_returns[i:(x-1+i)])
        Day1_vol[i] = Dayx_vol[i]/np.sqrt(x)
    mu_gamma = np.var(annualized_returns)
    var_gamma = np.var(Day1_vol)
    a= mu_gamma**2/var_gamma+2
    b= (a-1)*mu_gamma
    return a,b

def getLongTermVar(annualized_returns,x):
    Dayx_vol = np.zeros(annualized_returns.shape[0])    
    Day1_vol = np.zeros(annualized_returns.shape[0]) 
    for i in range(annualized_returns[x:].shape[0]):
        Dayx_vol[i] = np.std(annualized_returns[i:(x-1+i)])
        Day1_vol[i] = Dayx_vol[i]/np.sqrt(x)
    mu_theta = np.var(annualized_returns)
    var_theta = np.var(Day1_vol)
    return mu_theta,var_theta

#Prior distribution hyperparameter values
a=np.mean(Y);A=np.var(Y);#b=0;B=1;c=0;C=1;
d,D = get_hyperGamma(Y,30)
theta_star,theta_starVar = getLongTermVar(Y,30)
kappa_star = 0.5;kappa_starVar=0.1
b=kappa_star*theta_star;B=theta_starVar*kappa_starVar
c=1-kappa_star;C = kappa_starVar

#starting values parameters
V = 0.1*(Y-np.mean(Y))**2+0.9*(np.var(Y)) #initial values for variance_t
#V0=V[0]
#V_min1 = pd.Series(np.append(V[0],np.array(V.shift(1)[1:])),index= V.index)
sigma2_v=D/(d-1)
alpha=b;beta=c;m=a;rho=0;


#Initialize saving parameters for MC
mtot=0;mtot2=0 #drawing mu
alphatot=0;alphatot2=0
betatot=0;betatot2=0
dfSigV = 3; stdV=0.9;acceptPropSig=0;sigma2_vtot=0; sigma2_vtot2=0
stdRho = 0.005; dfRho = 6.5;acceptPropRho = 0;rhotot=0;rhotot2=0 #drawing rho
dfV = 4.5; stdV=0.9; Vtot=0;Vtot2=0;acceptPropV=np.zeros(T)


N=10000;burn=3000 #Number of draws and burn period
sig2V_save = np.zeros(N)
rho_save = np.zeros(N)
for i in tqdm(range(N)):
    
    #resest V_min1
    V_min1 = pd.Series(np.append(V[0],np.array(V.shift(1)[1:])),index= V.index)
    
    #Draw mu
    A_star = 1/(1/(1-rho**2)*np.sum(1/V_min1)+1/A)
    a_star = A_star*(1/(1-rho**2)*np.sum((Y-rho/sigma2_v**(0.5)*(V-alpha-beta*V_min1))/V_min1)+a/A)
    m=np.random.normal(a_star,A_star**0.5)
    
    if i>burn:
        mtot += m
        mtot2 =+ m**2
        
    #Draw alpha
    B_star = 1/(1/((1-rho**2)*sigma2_v)*np.sum(1/V_min1)+1/B)
    b_star = B_star*(1/((1-rho**2)*sigma2_v)*np.sum((V- beta*V_min1-rho*sigma2_v**(0.5)*(Y-m))/V_min1)+b/B)
    alpha=np.random.normal(b_star,B_star**0.5)
    
    if i>burn:
        alphatot += alpha
        alphatot2 =+ alpha**2
        
    #Draw beta
    C_star = 1/(1/((1-rho**2)*sigma2_v)*np.sum(V_min1)+1/C)
    c_star = C_star*(1/((1-rho**2)*sigma2_v)*np.sum(V- alpha-rho*sigma2_v**(0.5)*(Y-m))+c/C)
    beta=np.random.normal(c_star,C_star**0.5)
    
    if i>burn:
        betatot += m
        betatot2 =+ m**2
        
    #Draw sigma2_v
    d_star = d+T
    D_star = D+np.sum((V-alpha-beta*V_min1)**2/(V_min1))

    # def inv_gamma(x,a,b):
    #     y = (1/x)**(a+1)*np.exp(-b/x)
    #     b/(a-1)
    #     b**2/((a-1)**2*(a-2))
    #     return (y)
    
    # x=np.arange(100,8000,10000)
    # plt.plot(x,inv_gamma(x,85000,818800))
    # plt.show()
    sigma2_v_prop = stats.invwishart.rvs(d_star,D_star)
    #sigma2_v_prop = sigma2_v + stdRho*np.random.standard_t(dfSigV)
    e_v = V-alpha-V_min1*beta
    q = np.exp(-0.5*np.sum((e_v)**2/(sigma2_v_prop*V_min1) - (e_v)**2/(sigma2_v*V_min1)))
    p = np.exp(-0.5*np.sum(((e_v)**2-2*rho*sigma2_v_prop**0.5*(Y-m))/((1-rho**2)*sigma2_v_prop*V_min1)-((e_v)**2-2*rho*sigma2_v**0.5*(Y-m))/((1-rho**2)*sigma2_v*V_min1)))
      
    #if q!=np.inf and q!=0.0:
    x = min(p/q,1)
    #else:
    #    x=1
    u = np.random.uniform(0,1)
    if x>u:
        sigma2_v = sigma2_v_prop
        if i>burn:
            acceptPropSig +=1
    if i > burn:
        sigma2_vtot += sigma2_v
        sigma2_vtot2 += sigma2_v**2
        
    sig2V_save[i] = sigma2_v
    
    #Draw rho
    rho_prop = rho + stdRho*np.random.standard_t(dfRho)
    if np.abs(rho_prop)<1:
        e_y=Y-m
        e_v= V-alpha-beta*V_min1
        p = (np.sqrt(1-rho**2)/np.sqrt(1-rho_prop**2))**T*np.exp(-1/(2*(1-rho_prop**2))*np.sum((e_y**2-2*rho_prop/(sigma2_v**0.5)*e_v*e_y +e_v**2/sigma2_v)/V_min1)+1/(2*(1-rho**2))*np.sum((e_y**2-2*rho/sigma2_v**(0.5)*e_y*e_v+e_v**2/sigma2_v)/V_min1))
        u = np.random.uniform(0,1)
        if not np.isnan(p):
            x=min(p,1)
        else:
            x=1
        if x>u:
            rho=rho_prop
            if i>burn:
                acceptPropRho +=1
    if i>burn:
        rhotot += rho
        rhotot2 += rho**2
    rho_save[i]=rho
        
    #Draw V
    eps =  np.random.standard_t(dfV,size=T)
    eps = stdV/np.sqrt((dfV/(dfV-2)))*eps
    if i == np.floor(burn/2):
         Vpercentiles = np.percentile(Vtot2,[2.5,25,75,92.5])
         Vindex1 = np.where(Vtot2>=Vpercentiles[3])
         Vindex2 = np.where((Vpercentiles[2]<=Vtot2)&(Vtot2<Vpercentiles[3]))
         Vindex3 = np.where((Vpercentiles[0]<Vtot2)&(Vtot2<=Vpercentiles[1]))
         Vindex4 = np.where(Vtot2<=Vpercentiles[0])
    
    if i> np.floor(burn/2)-1:
         eps[Vindex1] = 1.35*eps[Vindex1]
         eps[Vindex2] = 1.25*eps[Vindex2]
         eps[Vindex3] = 0.75*eps[Vindex3]
         eps[Vindex4] = 0.65*eps[Vindex4]
    Vprop = V+eps
    
    #j==0
    p1 = max(0,Vprop[0]**(-1)*np.exp(-0.5*((Y[1]-m-rho/sigma2_v**0.5*(V[1]-alpha-beta*Vprop[0]))**2/(Vprop[0]*(1-rho**2) )+ (Y[0]-m-rho/sigma2_v**0.5*(Vprop[0]-alpha-beta*V[0]))**2/(V[0]*(1-rho**2))+
        (V[1]-alpha-beta*Vprop[0])**2/(sigma2_v*Vprop[0])+(Vprop[0]-alpha-beta*V[0])**2/(sigma2_v*V[0]))))
    p2 = max(0,V[0]**(-1)*np.exp(-0.5*((Y[1]-m-rho/sigma2_v**0.5*(V[1]-alpha-beta*V[0]))**2/(V[0]*(1-rho**2) )+ (Y[0]-m-rho/sigma2_v**0.5*(V[0]-alpha-beta*V[0]))**2/(V[0]*(1-rho**2))+
        (V[1]-alpha-beta*V[0])**2/(sigma2_v*V[0])+(V[0]-alpha-beta*V[0])**2/(sigma2_v*V[0])))    )

    acceptV= min(p1/p2,1) if p2!=0 else 1 if p1>0 else 0 
    u = np.random.uniform(0,1,T)
    if u[0]< acceptV:
        V[0]=Vprop[0]
        if i>burn: acceptPropV[0]+=acceptPropV[0]+1
    
    #j==(1,...,T-2)
    for j in range(1,T-1):
        p1 = max(0,Vprop[j]**(-1)*np.exp(-0.5*((Y[j+1]-m-rho/sigma2_v**0.5*(V[j+1]-alpha-beta*Vprop[j]))**2/(Vprop[j]*(1-rho**2) )+ (Y[j]-m-rho/sigma2_v**0.5*(Vprop[j]-alpha-beta*V[j-1]))**2/(V[j-1]*(1-rho**2))+
            (V[j+1]-alpha-beta*Vprop[j])**2/(sigma2_v*Vprop[j])+(Vprop[j]-alpha-beta*V[j-1])**2/(sigma2_v*V[j-1]))))
        p2 = max(0,V[j]**(-1)*np.exp(-0.5*((Y[j+1]-m-rho/sigma2_v**0.5*(V[j+1]-alpha-beta*V[j]))**2/(V[j]*(1-rho**2) )+ (Y[j]-m-rho/sigma2_v**0.5*(V[j]-alpha-beta*V[j-1]))**2/(V[j-1]*(1-rho**2))+
            (V[j+1]-alpha-beta*V[j])**2/(sigma2_v*V[j])+(V[j]-alpha-beta*V[j-1])**2/(sigma2_v*V[j-1])))    )
 
        acceptV=min(p1/p2,1) if p2!=0 else 1 if p1>0 else 0 
        if u[j]< acceptV:
            V[j]=Vprop[j]
            if i>burn: acceptPropV[j]+=acceptPropV[j]+1
            
    #j==(T-1)
    p1 = max(0,Vprop[T-1]**(-0.5)*np.exp(-0.5*((Y[T-1]-m-rho/sigma2_v**0.5*(Vprop[T-1]-alpha-beta*V[T-2]))**2/(V[T-2]*(1-rho**2)) + (Vprop[T-1]-alpha-beta*V[T-2])**2/(V[T-2]*sigma2_v))))
    p2 = max(0,V[T-1]**(-0.5)*np.exp(-0.5*((Y[T-1]-m-rho/sigma2_v**0.5*(V[T-1]-alpha-beta*V[T-2]))**2/(V[T-2]*(1-rho**2)) + (V[T-1]-alpha-beta*V[T-2])**2/(V[T-2]*sigma2_v))))

    acceptV=min(p1/p2,1) if p2!=0 else 1 if p1>0 else 0 
    if u[T-1]< acceptV:
        V[T-1]=Vprop[T-1]
        if i>burn: acceptPropV[T-1]+=acceptPropV[T-1]+1
    
    if i>burn: Vtot+= V
    if i>np.floor(burn/2)-100 or i < np.floor(burn/2):   Vtot2 += V
    
    
# #speed up calculations
# pool4 = Pool(processes=4)
# result_list = list(tqdm(pool4.imap_unordered(Bootstrap, arglist), total=REP))
# pool4.close()
# pool4.join()    
    
    
#Monte carlo estimates
m=mtot/(N-burn)
sigma2_v=sigma2_vtot/(N-burn)
rho = rhotot/(N-burn)
alpha = alphatot/(N-burn)
beta = betatot/(N-burn)
V_result = Vtot/(N-burn)
print(m,sigma2_v,alpha,beta,rho)
        
#Y.plot()
plt.plot(sig2V_save)
plt.show()
        

    