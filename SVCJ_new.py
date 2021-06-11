# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:06:06 2021

@author: Dasno7
"""
# -*- coding: utf-8 -*-
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
J = (abs(Y)-np.mean(Y)>2*np.std(Y)).astype(int) # starting value of jumps

def get_hyperBeta(annualized_returns):
    J = (abs(annualized_returns)-np.mean(annualized_returns)>2*np.std(annualized_returns)).astype(int) 
    mean = np.sum(J)/J.shape[0]
    var = 2*40/((42**2)*43) #approx var used for lambda
    a = ((1-mean)*mean**2)/(var*(mean+1))
    b = a*(1-mean)/mean
    return a,b


def trunc_norm(mu,sigma2,left,right):
    sigma = np.sqrt(sigma2)
    a = stats.norm.cdf((left-mu)/sigma)
    b =  stats.norm.cdf((right-mu)/sigma)
    unif = np.random.uniform(0,1,1) #mu.shape[0] for MV
    p = a+unif*(b-a)
    
    return mu + sigma*stats.norm.ppf(p)


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
kappa_star = 0.5;kappa_starVar=0.01
b=kappa_star*theta_star;B=theta_starVar*kappa_starVar
c=1-kappa_star;C = kappa_starVar
e = np.mean(Y[np.where(J==1)[0]]); E = np.var(Y[np.where(J==1)[0]])
k,K = get_hyperBeta(Y) #hyper lambda
g=0;G=4 #rho_j

#starting values parameters
V = 0.1*(Y-np.mean(Y))**2+0.9*(np.var(Y)) #initial values for variance_t
#V0=V[0]
#V_min1 = pd.Series(np.append(V[0],np.array(V.shift(1)[1:])),index= V.index)
sigma2_v=D/(d-1)
alpha=b;beta=c;m=a;rho=0;
r,R = get_hyperGamma(Y[np.where(J==1)[0]],5) #hyper of mu_v
mu_v = (R/(r-1))**0.5
xi_v = np.random.exponential(mu_v,T)
rho_j=0
sigma2_s = E
mu_s = e
xi_s = np.random.normal(e+rho_j*xi_v,E**0.5)*J
h,H = get_hyperGamma(Y[np.where(J==1)[0]],int(np.round(np.where(J==1)[0].shape[0]/5)))



#Initialize saving parameters for MC
mtot=0;mtot2=0 #drawing mu
alphatot=0;alphatot2=0
betatot=0;betatot2=0
lambtot=0;lambtot2=0 #drawing lambda
m_stot=0;m_stot2=0;sigma2_stot=0;sigma2_stot2=0 #xi_s
mVtot = 0; mVtot2=0 #muV
rho_jtot=0;rho_jtot2=0 #rho_j
Jtot=0
xi_s_tot=0;xi_v_tot=0
dfSigV = 3; stdV=0.9;acceptPropSig=0;sigma2_vtot=0; sigma2_vtot2=0
stdRho = 0.005; dfRho = 6.5;acceptPropRho = 0;rhotot=0;rhotot2=0 #drawing rho
dfV = 4.5; stdV=0.9; Vtot=0;Vtot2=0;acceptPropV=np.zeros(T)


N=10000;burn=3000 #Number of draws and burn period
sig2V_save = np.zeros(N)
rho_save = np.zeros(N)
muS_save = np.zeros(N)
sigma2S_save = np.zeros(N)
for i in tqdm(range(N)):
    
    #resest V_min1
    V_min1 = pd.Series(np.append(V[0],np.array(V.shift(1)[1:])),index= V.index)
    
    #Draw mu
    A_star = 1/(1/(1-rho**2)*np.sum(1/V_min1)+1/A)
    a_star = A_star*(1/(1-rho**2)*np.sum(((Y-xi_s*J)-rho/sigma2_v**(0.5)*(V-alpha-beta*V_min1-xi_v*J))/V_min1)+a/A)
    m=np.random.normal(a_star,A_star**0.5)
    
    if i>burn:
        mtot += m
        mtot2 =+ m**2
        
    #Draw alpha
    B_star = 1/(1/((1-rho**2)*sigma2_v)*np.sum(1/V_min1)+1/B)
    b_star = B_star*(1/((1-rho**2)*sigma2_v)*np.sum((V- beta*V_min1-xi_v*J-rho*sigma2_v**(0.5)*(Y-m-xi_s*J))/V_min1)+b/B)
    alpha=np.random.normal(b_star,B_star**0.5)
    
    if i>burn:
        alphatot += alpha
        alphatot2 =+ alpha**2
        
    #Draw beta
    C_star = 1/(1/((1-rho**2)*sigma2_v)*np.sum(V_min1)+1/C)
    c_star = C_star*(1/((1-rho**2)*sigma2_v)*np.sum(V- alpha-xi_v*J-rho*sigma2_v**(0.5)*(Y-m-xi_s*J))+c/C)
    beta=np.random.normal(c_star,C_star**0.5)
    
    if i>burn:
        betatot += m
        betatot2 =+ m**2
        
    #Draw lambda   
    k_star = k + np.sum(J)
    K_star = K+T-np.sum(J)
    lamb = np.random.beta(k_star,K_star)
    if i>burn:
        lambtot += lamb
        lambtot2 += lamb**2
        
    #Draw sigma2_v
    d_star = d+T
    D_star = D+np.sum((V-alpha-beta*V_min1-xi_v*J)**2/(V_min1))
    sigma2_v_prop = stats.invwishart.rvs(d_star,D_star)
    e_v = V-alpha-V_min1*beta-xi_v*J
    q = np.exp(-0.5*np.sum((e_v)**2/(sigma2_v_prop*V_min1) - (e_v)**2/(sigma2_v*V_min1)))
    p = np.exp(-0.5*np.sum(((e_v)**2-2*rho*sigma2_v_prop**0.5*(Y-m-xi_s*J))/((1-rho**2)*sigma2_v_prop*V_min1)-((e_v)**2-2*rho*sigma2_v**0.5*(Y-m-xi_s*J))/((1-rho**2)*sigma2_v*V_min1)))

    x = min(p/q,1)
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
        e_y=Y-m-xi_s*J
        e_v= V-alpha-beta*V_min1-xi_v*J
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
        
    #mu_s
    E_star = 1/(T/sigma2_s+1/E)
    e_star = E_star*(np.sum(xi_s-rho_j*xi_v)/sigma2_s+e/E)
    mu_s = np.random.normal(e_star,E_star**0.5)
    if i>burn:
        m_stot += mu_s
        m_stot += mu_s**2
    muS_save[i]=mu_s
    
    #Draw sigma2_s
    h_star = h +T
    H_star = H+np.sum((xi_s-mu_s-rho_j*xi_v)**2)
    sigma2_s = stats.invgamma.rvs(h_star, scale=H_star)
    if i>burn:
        sigma2_stot += sigma2_s
        sigma2_stot2 += sigma2_s**2 
    sigma2S_save[i] = sigma2_s
    
    #Draw mu_v
    r_star = r+2*T
    R_star = R+2*np.sum(xi_v)
    mu_v = stats.invwishart.rvs(r_star,R_star)
    if i > burn:
        mVtot += mu_v
        mVtot2 += mu_v**2
        
    #Draw rho_J
    G_star=1/(np.sum(xi_v**2)/sigma2_s+1/G)
    g_star=G_star*(np.sum((xi_s-mu_s)*xi_v)/sigma2_s+g/G)
    rho_j = np.random.normal(g_star,G_star**0.5)
    if i>burn:
        rho_jtot += rho_j
        rho_jtot2 += rho_j**2
        
    #Jumps
    eY0 = Y - m
    eY1 = eY0 - xi_s
    eV0 = V - alpha - beta*V_min1
    eV1 = eV0 - xi_v
    p1 = lamb*np.exp(-(eY1-rho*sigma2_v**(-0.5)*eV1)**2/(2*(1-rho**2)*V_min1)-eV1**2/(2*sigma2_v*V_min1))
    p2 = (1-lamb)*np.exp(-(eY0-rho*sigma2_v**(-0.5)*eV0)**2/(2*(1-rho**2)*V_min1)-eV0**2/(2*sigma2_v*V_min1))
    p = p1/(p1+p2)
    J = np.random.binomial(1,p,T)
    if i >burn:
        Jtot += J
        
    #xi_v     (improve by vectorizing?)
    Jindex = np.where(J==1)[0] 
    if Jindex.size != 0:
        for j in Jindex:
         eY = Y[j]-m-xi_s[j]
         eV  = V[j]-alpha-V_min1[j]*beta
         sigma2_v_star = 1/(1/(sigma2_v*(1-rho**2)*V_min1[j])+rho_j**2/sigma2_s**2)
         mu_v_star = sigma2_v_star*((eV-rho*sigma2_v**0.5*eY)/(sigma2_v*(1-rho**2)*V_min1[j])+rho_j*(xi_s[j]-mu_s)/(sigma2_s)-1/mu_v)
         upper_bound = mu_v_star+5*sigma2_v_star**0.5
         if upper_bound>0: xi_v[j] = trunc_norm(mu_v_star,  sigma2_v_star, 0, upper_bound)
    if i>burn:
        xi_v_tot += xi_v 
             
         
    #xi_s (improve by vectorizing?)
    if Jindex.size != 0:
        for j in Jindex:
            sigma2_s_star = 1/(1/sigma2_s+1/((1-rho**2)*V_min1[j]))
            eY = Y[j]-m
            eV = V[j]-alpha-V_min1[j]*beta-xi_v[j]
            mu_s_star = sigma2_s_star*((mu_v+rho_j*xi_v[j])/(sigma2_s)+(eY-rho/sigma2_v**0.5*eV)/((1-rho**2)*V_min1[j]))
            xi_s[j] = np.random.normal(mu_s_star+rho_j*xi_s[j],sigma2_s_star)
    if i>burn:
        xi_s_tot += xi_s
    
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
    p1 = max(0,Vprop[0]**(-1)*np.exp(-0.5*((Y[1]-m-xi_s[1]*J[1]-rho/sigma2_v**0.5*(V[1]-alpha-beta*Vprop[0])-xi_v[1]*J[1])**2/(Vprop[0]*(1-rho**2) )+ (Y[0]-m-xi_s[0]*J[0]-rho/sigma2_v**0.5*(Vprop[0]-alpha-beta*V[0]-xi_v[0]*J[0]))**2/(V[0]*(1-rho**2))+
        (V[1]-alpha-beta*Vprop[0]-xi_v[1]*J[1])**2/(sigma2_v*Vprop[0])+(Vprop[0]-alpha-beta*V[0]-xi_v[0]*J[0])**2/(sigma2_v*V[0]))))
    p2 = max(0,V[0]**(-1)*np.exp(-0.5*((Y[1]-m-xi_s[1]*J[1]-rho/sigma2_v**0.5*(V[1]-alpha-beta*V[0]-xi_v[1]*J[1]))**2/(V[0]*(1-rho**2) )+ (Y[0]-m-xi_s[0]*J[0]-rho/sigma2_v**0.5*(V[0]-alpha-beta*V[0]-xi_v[0]*J[0]))**2/(V[0]*(1-rho**2))+
        (V[1]-alpha-beta*V[0]-xi_v[1]*J[1])**2/(sigma2_v*V[0])+(V[0]-alpha-beta*V[0]-xi_v[0]*J[0])**2/(sigma2_v*V[0])))    )

    acceptV= min(p1/p2,1) if p2!=0 else 1 if p1>0 else 0 
    u = np.random.uniform(0,1,T)
    if u[0]< acceptV:
        V[0]=Vprop[0]
        if i>burn: acceptPropV[0]+=acceptPropV[0]+1
    
    #j==(1,...,T-2)
    for j in range(1,T-1):
        p1 = max(0,Vprop[j]**(-1)*np.exp(-0.5*((Y[j+1]-m-xi_s[j+1]*J[j+1]-rho/sigma2_v**0.5*(V[j+1]-alpha-beta*Vprop[j]-xi_v[j+1]*J[j+1]))**2/(Vprop[j]*(1-rho**2) )+ (Y[j]-m-xi_s[j]*J[j]-rho/sigma2_v**0.5*(Vprop[j]-alpha-beta*V[j-1]-xi_v[j]*J[j]))**2/(V[j-1]*(1-rho**2))+
            (V[j+1]-alpha-beta*Vprop[j]-xi_v[j+1]*J[j+1])**2/(sigma2_v*Vprop[j])+(Vprop[j]-alpha-beta*V[j-1]-xi_v[j]*J[j])**2/(sigma2_v*V[j-1]))))
        p2 = max(0,V[j]**(-1)*np.exp(-0.5*((Y[j+1]-m-xi_s[j+1]*J[j+1]-rho/sigma2_v**0.5*(V[j+1]-alpha-beta*V[j]-xi_v[j+1]*J[j+1]))**2/(V[j]*(1-rho**2) )+ (Y[j]-m-xi_s[j]*J[j]-rho/sigma2_v**0.5*(V[j]-alpha-beta*V[j-1]-xi_v[j]*J[j]))**2/(V[j-1]*(1-rho**2))+
            (V[j+1]-alpha-beta*V[j]-xi_v[j+1]*J[j+1])**2/(sigma2_v*V[j])+(V[j]-alpha-beta*V[j-1]-xi_v[j]*J[j])**2/(sigma2_v*V[j-1])))    )
 
        acceptV=min(p1/p2,1) if p2!=0 else 1 if p1>0 else 0 
        if u[j]< acceptV:
            V[j]=Vprop[j]
            if i>burn: acceptPropV[j]+=acceptPropV[j]+1
            
    #j==(T-1)
    p1 = max(0,Vprop[T-1]**(-0.5)*np.exp(-0.5*((Y[T-1]-m-xi_s[T-1]*J[T-1]-rho/sigma2_v**0.5*(Vprop[T-1]-alpha-beta*V[T-2]-xi_v[T-1]*J[T-1]))**2/(V[T-2]*(1-rho**2)) + (Vprop[T-1]-alpha-beta*V[T-2]-xi_v[T-1]*J[T-1])**2/(V[T-2]*sigma2_v))))
    p2 = max(0,V[T-1]**(-0.5)*np.exp(-0.5*((Y[T-1]-m-xi_s[T-1]*J[T-1]-rho/sigma2_v**0.5*(V[T-1]-alpha-beta*V[T-2]-xi_v[T-1]*J[T-1]))**2/(V[T-2]*(1-rho**2)) + (V[T-1]-alpha-beta*V[T-2]-xi_v[T-1]*J[T-1])**2/(V[T-2]*sigma2_v))))

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
    
#MC estimates
def MC_results(N, burn, tot, tot2):
     param_est = tot/(N-burn)   
     error_est = (tot2/(N-burn)-(param_est)**2)**0.5
     return param_est,error_est    
    
#Monte carlo estimates
m, m_se = MC_results(N, burn,mtot,mtot2)
alpha, alpha_se = MC_results(N, burn,alphatot,alphatot2)
beta, beta_se = MC_results(N, burn,betatot,betatot2)
rho, rho_se = MC_results(N, burn,rhotot,rhotot2)
mV, mV_se = MC_results(N, burn,mVtot,mVtot2)
mS, mS_se = MC_results(N, burn,m_stot,m_stot2)
sigma2_s, sigma2_s_se = MC_results(N, burn,sigma2_stot,sigma2_stot2)
sigma2_v, sigma2_v_se = MC_results(N, burn,sigma2_vtot,sigma2_vtot2)
rho_j, rho_j_se = MC_results(N, burn,rho_jtot,rho_jtot2)
lamb, lamb_se = MC_results(N, burn,lambtot,lambtot2)

xi_s = xi_s_tot/(N-burn)
xi_v = xi_v_tot/(N-burn)
J_test = (Jtot/(N-burn)>=0.003).astype(int)
V_result = Vtot/(N-burn)
print(m,sigma2_v,alpha,beta,rho)
        
#Y.plot()
plt.plot(sig2V_save)
plt.show()
        

    