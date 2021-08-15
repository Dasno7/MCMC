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
import scipy.special as special
import scipy.stats as stats
from scipy.interpolate import griddata
from matplotlib import colors

#price data
sns.set_theme(style='darkgrid')
cryptos = ['Bitcoin','LINK','ETH','ADA']
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


# FINAL PARAMETERS
mu ={'Bitcoin': 0.02029249494235486/365, 'LINK':0.04698867505564754, 'ETH':  0.01583345207967061/365, 'ADA':-0.003431383092799574 }#/np.sqrt(365)
s2V = {'Bitcoin': 0.04033712521749577/365, 'LINK':0.18696787354715597 , 'ETH': 0.09823061919130012/365, 'ADA':0.13299513034831426 }#/np.sqrt(365)
alpha ={'Bitcoin':  0.027300605314312377, 'LINK':0.1939941460600538, 'ETH': 0.0927768090812799, 'ADA': 0.1500259628758315}
beta ={'Bitcoin':0.9578989881256078 , 'LINK':0.8845069692442109, 'ETH': 0.9114176612486856, 'ADA': 0.8795055872798027}
rho ={'Bitcoin': -0.06637348251947067, 'LINK':-0.04645815936821193, 'ETH':-0.07260865056494854 , 'ADA':-0.05034955922479641 }
kappa ={'Bitcoin':(1-0.9578989881256078)/365, 'LINK':(1-beta['LINK']), 'ETH':(1- 0.9114176612486856)/365, 'ADA': (1-beta['ADA'])}
theta ={'Bitcoin':  0.027300605314312377/kappa['Bitcoin']/365, 'LINK':alpha['LINK']/kappa['LINK'], 'ETH': 0.0927768090812799/kappa['ETH']/365, 'ADA': alpha['ADA']/kappa['ADA']}

N=T

explode_corr = {'Bitcoin': 2, 'LINK':80, 'ETH':  3.5, 'ADA':20 }
plot_explode = {'Bitcoin': 0, 'LINK':0 , 'ETH':  0, 'ADA':0 }

# #initialize RMSE and empirical forecasting density
RMSE = {}
densityEnd = {}

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
    Y[0] = Ydict[crypto][0]
    v[0] =  np.sqrt((0.1*(Ydict[crypto]-np.mean(Ydict[crypto]))**2+0.9*(np.var(Ydict[crypto])))[0]) # Initial value of volatility = mean of volatilty
    S[0] = P[crypto][0]
    
    sim=np.zeros([int(1e5),T[crypto]+1])
    v_sim=np.zeros([int(1e5),T[crypto]+1])
    iters=0
    while iters <int(1e5): 
        # Run the simulation T times and save the calculated values
        interval = np.array(range(N[crypto]+1))/N[crypto] #tweak
        TT = N[crypto]/365
        dt = np.diff(interval*(TT))[0]
        a=(2*(2*rho[crypto]*s2V[crypto]**0.5-kappa[crypto])/s2V[crypto])**2
        i=1
        while i <N[crypto]+1:
            Zy        = np.random.normal(0,1)  # Standard normal random value
            Z_interim       = np.random.normal(0,1)  # Standard normal random value
            Zv       = rho[crypto]*Zy + np.sqrt(1-rho[crypto]**2)*Z_interim
              
                      
            #V[i]  = kappa * theta + (1 - kappa) * V[i - 1] + s2V**0.5* np.sqrt(max(V[i - 1],0))* Zv
            #Y[i]  =  Y[i-1]+ mu+ np.sqrt(max(V[i - 1],0)) * Zy
            
            #transformed volatility scheme
            v[i] = v[i-1]+0.5*(kappa[crypto]*(theta[crypto]/v[i-1]-v[i-1])+s2V[crypto]/(4*v[i-1])+s2V[crypto]**0.5*Zv)
            Y[i] = mu[crypto]*dt +np.sqrt(TT/N[crypto])* v[i-1]*Zy
            
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
    
    #Return forecasting density 
    densityEnd[crypto] = (pd.DataFrame(sim[:,[-2,-1]],columns = [Ydict[crypto].index[[-2,-1]]]))
   
    #RMSE
    RMSE[crypto] =np.sqrt(np.mean((np.mean(sim,axis=0)-P[crypto])**2))
 
    pd.DataFrame(v_sim[np.where(np.max(v_sim,axis=1)>=plot_explode[crypto])[0][:5],:-1].T,index=Ydict[crypto].index).plot(ax=   listAx2[crypto],colormap=sns.color_palette("flare", as_cmap=True),legend=False)       

    listAx2[crypto].hlines(np.std(Ydict[crypto]),xmin=-0,xmax=Ydict[crypto].index.size)
    listAx2[crypto].set(xlabel=None)
    listAx2[crypto].set(ylabel="Volatility")
    listAx2[crypto].set_title(cryptoNames[crypto],fontdict= { 'fontsize': 18, 'fontweight':'bold'})
fig2

def getParamForecastingDensity(densityDict,cryptos):
    dictMean={}
    dictVar={}
    for crypto in cryptos:
        score = np.log(np.array(densityEnd[crypto][densityEnd[crypto].columns[1][0]])/np.array(densityEnd[crypto][densityEnd[crypto].columns[0][0]]))
        dictMean[crypto] =np.mean(score*365) 
        dictVar[crypto] = np.var(score*np.sqrt(365))
        
    return dictMean,dictVar

dictMean,dictVar = getParamForecastingDensity(densityEnd,cryptos)

#---------------------------------------------------------------------------------------------------------------------------
# Functions for Crude Monte Carlo

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
    
def putOptionPriceAnalytical(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(-d1)
    N_d2 = stats.norm.cdf(-d2)

    europePutAnalytical = K * np.exp(-r * T) * N_d2 - S0 * N_d1
    return europePutAnalytical

def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=1e6):
    sigma = 0.5

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = C - putOptionPriceAnalytical(S, K, T, r, sigma)

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            if i==0:
                return np.nan
            else: return sigma

        ### use newton raphson to update the estimate
        sigma = sigma + diff / vega(S, K, T, r, sigma)

    return np.nan



def simulSV(crypto,stockPrice,horiz):
    V    = np.zeros(horiz+1)  # Volatility of log return
    v    = np.zeros(horiz+1)  # sqrt Volatility of log return
    Y    = np.zeros(horiz+1)  # Log return
    S = np.zeros(horiz+1)
    Jv   = np.zeros(horiz+1)  # Jumps in volatility
    Jy   = np.zeros(horiz+1) # Jumps in log return
    Y[0] = Ydict[crypto][0]
    v[0] =  np.sqrt((0.1*(Ydict[crypto]-np.mean(Ydict[crypto]))**2+0.9*(np.var(Ydict[crypto])))[0]) # Initial value of volatility = mean of volatilty
    S[0] = stockPrice
    
    sim=np.zeros([int(1e4),horiz+1])
    v_sim=np.zeros([int(1e4),horiz+1])
    jv_sim = np.zeros([int(1e4),horiz+1])
    iters=0
    while iters <int(1e4): 
        # Run the simulation T times and save the calculated values
        interval = np.array(range(horiz+1))/horiz #tweak
        TT = horiz/365
        dt = np.diff(interval*(TT))[0]
        i=1
        while i <horiz+1:
            Zy        = np.random.normal(0,1)  # Standard normal random value
            Z_interim       = np.random.normal(0,1)  # Standard normal random value
            Zv       = rho[crypto]*Zy + np.sqrt(1-rho[crypto]**2)*Z_interim
            
            #transformed volatility scheme
            v[i] = v[i-1]+0.5*(kappa[crypto]*(theta[crypto]/v[i-1]-v[i-1])+s2V[crypto]/(4*v[i-1])+s2V[crypto]**0.5*Zv)
            Y[i] = (mu[crypto]-0.5*v[i-1])*dt +np.sqrt(TT/horiz)* v[i-1]*Zy
            
            S[i] = np.exp(Y[i])*S[i-1]
            
            #if a+explode_corr[crypto]< abs(4*v[i-1]):
            if explode_corr[crypto]< abs(v[i-1]):
                print('out' +str(v[i-1])+"  "+str(i)+"   "+crypto)
                break
            else:
                i+=1
        if i ==horiz+1:
                sim[iters,:] = S
                v_sim[iters,:] = v
                if iters%1000==0: print(crypto+":"+str(iters))
                iters+=1
        else:
          continue
    return sim

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

sim = simulSV(crypto,S,np.max(T_BTC1))

i=0; optionDict_BTC1= {}
for K in K_BTC1:
    for t in T_BTC1:
        payOffCall = (K-sim[:,t]).clip(min=0)
        callPrice =np.mean(payOffCall)
        optionDict_BTC1[i]=[callPrice,t,K]
        i+=1
BTC1 = pd.DataFrame(optionDict_BTC1).T
BTC1.columns = ['Call price','T','K']
imp = [implied_volatility_call(BTC1.loc[x]['Call price'],S,BTC1.loc[x]['K'],BTC1.loc[x]['T']/365,mu[crypto],max_iterations=int(1e2)) for x in BTC1.index]   
BTC1["IV"]=imp


#BTC scenario 2: 14 March 2021 S_0=57663
crypto='Bitcoin'
S=57663
K_BTC2 = np.array([38000, 32000, 40000, 53000, 36000, 42000, 34000, 66000, 48000,
        62000, 68000, 44000, 80000, 84000, 58000, 46000, 50000, 70000,
        64000, 56000, 59000, 54000, 60000, 78000, 55000, 52000, 74000,
        57000, 76000, 72000, 51000, 63000, 61000, 65000, 67000, 88000,
        4000, 24000, 14000, 26000, 28000, 16000, 120000, 10000, 18000,
        140000, 12000, 100000, 7000, 8000, 90000, 85000, 9000, 20000,
        15000, 22000, 17000, 45000, 75000, 6000, 13000, 82000, 30000,
        92000, 86000, 5000, 11000])
T_BTC2 = np.array([ 4, 19, 47, 6, 12, 26, 103, 5, 194, 292, 75])

sim = simulSV(crypto,S,np.max(T_BTC2))

i=0; optionDict_BTC2= {}
for K in K_BTC2:
    for t in T_BTC2:
        payOffCall = (K-sim[:,t]).clip(min=0)
        callPrice =np.mean(payOffCall)
        optionDict_BTC2[i]=[callPrice,t,K]
        i+=1
BTC2 = pd.DataFrame(optionDict_BTC2).T
BTC2.columns = ['Call price','T','K']
    
imp = [implied_volatility_call(BTC2.loc[x]['Call price'],S,BTC2.loc[x]['K'],BTC2.loc[x]['T']/365,0,max_iterations=int(1e4)) for x in BTC2.index]   
BTC2["IV"]=imp

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
sim = simulSV(crypto,S,np.max(T_BTC3))

i=0; optionDict_BTC3= {}
for K in K_BTC3:
    for t in T_BTC3:
        payOffCall = (K-sim[:,t]).clip(min=0)
        callPrice =np.mean(payOffCall)
        optionDict_BTC3[i]=[callPrice,t,K]
        i+=1
BTC3 = pd.DataFrame(optionDict_BTC3).T
BTC3.columns = ['Call price','T','K']
    
imp = [implied_volatility_call(BTC3.loc[x]['Call price'],S,BTC3.loc[x]['K'],BTC3.loc[x]['T']/365,0,max_iterations=int(1e4)) for x in BTC3.index]   
BTC3["IV"]=imp

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

sim = simulSV(crypto,S,np.max(T_ETH1))

i=0; optionDict_ETH1= {}
for K in K_ETH1:
    for t in T_ETH1:
        payOffCall = (sim[:,t]-K).clip(min=0)
        callPrice =np.mean(payOffCall)*np.exp(-mu[crypto]*t/365)
        optionDict_ETH1[i]=[callPrice,t,K]
        i+=1
ETH1 = pd.DataFrame(optionDict_ETH1).T
ETH1.columns = ['Call price','T','K']
    
imp = [implied_volatility_call(ETH1.loc[x]['Call price'],S,ETH1.loc[x]['K'],ETH1.loc[x]['T']/365,0,max_iterations=int(1e2)) for x in ETH1.index]   
ETH1["IV"]=imp

#ETH scenario 2: 14 March 2021 S_0=2647
crypto = 'ETH'
S=2647

K_ETH2=np.array([1120, 1740, 1240, 1200, 2040, 1640, 1400, 1520, 2240, 2640, 2120,
        1840, 1600, 1440, 1860, 1960, 2200, 1800, 2320, 1680, 1360, 2080,
        2400, 2560, 2000, 1320, 1900, 1760, 1780, 1280, 1920, 1880, 2280,
        2900, 3000, 4000, 4080, 4380, 4520, 3240, 3680, 4800, 5240, 3960,
        2480, 1720, 2800, 2720, 2880, 2160, 1560, 1820, 1480, 1660, 1700,
        1620, 1940, 1980, 2060, 2020, 440, 180, 640, 720, 300, 800, 600,
        3200, 3000, 560, 380, 960, 3520, 1950, 3840, 420, 360, 280, 4160,
        1850, 340, 320, 2100, 160, 520, 480, 2600, 140, 2500, 460, 220,
        1040, 200, 1000, 400, 260, 240, 1750, 4000, 1500, 880])

T_ETH2=np.array([2, 3, 4, 6, 5, 19, 75, 47, 12, 194, 103, 292, 26])

sim = simulSV(crypto,S,np.max(T_ETH2))

i=0; optionDict_ETH2= {}
for K in K_ETH2:
    for t in T_ETH2:
        payOffCall = (K-sim[:,t]).clip(min=0)
        callPrice =np.mean(payOffCall)
        optionDict_ETH2[i]=[callPrice,t,K]
        i+=1
ETH2 = pd.DataFrame(optionDict_ETH2).T
ETH2.columns = ['Call price','T','K']
    
imp = [implied_volatility_call(ETH2.loc[x]['Call price'],S,ETH2.loc[x]['K'],ETH2.loc[x]['T']/365,0,max_iterations=int(1e2)) for x in ETH2.index]   
ETH2["IV"]=imp

#ETH scenario 3: 16 November 2020 S_0=458
crypto = 'ETH'
S=458

K_ETH3=np.array([470, 500, 430, 460, 390, 480, 420, 410, 400, 490, 450, 440, 180,
        320, 300, 340, 240, 330, 80, 380, 280, 220, 120, 160, 360, 140,
        290, 100, 310, 40, 200, 260, 350, 370, 60, 530, 550, 510, 540, 560,
        520, 580, 800, 720, 640, 630, 960, 600, 1120, 590, 620, 570, 880,
        610])

T_ETH3=np.array([130, 39, 11, 4, 221, 74, 18, 1, 0, 2])

sim = simulSV(crypto,S,np.max(T_ETH3))

i=0; optionDict_ETH3= {}
for K in K_ETH3:
    for t in T_ETH3:
        payOffCall = (K-sim[:,t]).clip(min=0)
        callPrice =np.mean(payOffCall)
        optionDict_ETH3[i]=[callPrice,t,K]
        i+=1
ETH3 = pd.DataFrame(optionDict_ETH3).T
ETH3.columns = ['Call price','T','K']
    
imp = [implied_volatility_call(ETH3.loc[x]['Call price'],S,ETH3.loc[x]['K'],ETH3.loc[x]['T']/365,0,max_iterations=int(1e2)) for x in ETH3.index]   
ETH3["IV"]=imp

#--------------------------------------------------------------------------------------------------------------------------------------
#Plot Functions

def make_surf(X,Y,Z):
    XX,YY = np.meshgrid(np.linspace(min(X),max(X),230),np.linspace(min(Y),max(Y),230))
    ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
    return XX,YY,ZZ 

def mesh_plotBTC(fig,ax,title,X,Y,Z):
    XX,YY,ZZ = make_surf(X,Y,Z)
    ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=1000,limit_direction='both'))
    XX=XX[:,50:200];YY=YY[:,50:200];ZZ=ZZ[:,50:200]
    XX=XX[120:,:];YY=YY[120:,:];ZZ=ZZ[120:,:]   
    my_cmap = plt.get_cmap('plasma') 
    surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
    ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
    ax.view_init(33, 60) 
    return ax

def mesh_plotBTC_3(fig,ax,title,X,Y,Z):
    XX,YY,ZZ = make_surf(X,Y,Z)
    np.nan_to_num(ZZ[:,93:],copy=False,nan=0)
    ZZ=np.array(pd.DataFrame(ZZ).interpolate(method='linear',axis=0,limit=1000,limit_direction='both'))
    XX=XX[:,40:];YY=YY[:,40:];ZZ=ZZ[:,40:]
    XX=XX[40:,:];YY=YY[40:,:];ZZ=ZZ[40:,:]  
    my_cmap = plt.get_cmap('plasma') 
    surf = ax.plot_surface(XX,YY,ZZ, cmap=my_cmap, rstride=25, cstride=16,
                        edgecolors='k', lw=0.6, antialiased=True,norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5))
    ax.set_title(title, fontdict= { 'fontsize': 18},style='italic')
    return surf

#--------------------------------------------------------------------------------------------------------------------------------------
#Plot BTC
fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

model='IV'

BTC1_plot= BTC1.where((BTC1['K']/35198) <=2.5).dropna()
axis1 = mesh_plotBTC(fig,axis1,"7 June 2021",BTC1_plot['K']/35198,BTC1_plot['T']/365,BTC1_plot[model])

BTC2_plot= BTC2.where((BTC2['K']/57663) <=2.5).dropna()
axis2 = mesh_plotBTC(fig,axis2,"14 March 2021",BTC2_plot['K']/57663,BTC2_plot['T']/365,(BTC2_plot[model]-np.min(BTC2_plot[model])-0.1)/2)

BTC3_plot= BTC3.where((BTC3['K']/16053) <=2.5).dropna()
axis3 = mesh_plotBTC(fig,axis3,"16 November 2020",BTC3_plot['K']/16053,BTC3_plot['T']/365,(BTC3_plot[model]-np.min(BTC3_plot[model])-0.1)/2)


fig, [axis1, axis2, axis3] = plt.subplots(1,3,figsize=(22.5,12), subplot_kw=dict(projection="3d",xlabel=r'$K/S$',ylabel=r'$T$',zlabel=r'$\sigma_{imp}$'),constrained_layout = True)

ETH1_plot= ETH1.where((ETH1['K']/1795) <=2.5).dropna()
axis1 = mesh_plotBTC(fig,axis1,"7 June 2021",ETH1_plot['K']/1795,ETH1_plot['T']/365,ETH1_plot[model])

ETH2_plot= ETH2.where((ETH2['K']/2647) <=2.5).dropna()
axis2 = mesh_plotBTC(fig,axis2,"14 March 2021",ETH2_plot['K']/2647,ETH2_plot['T']/365,(ETH2_plot[model]-np.min(ETH2_plot[model]))/2)

ETH3_plot= ETH3.where((ETH3['K']/458) <=2.5).dropna()
axis3 = mesh_plotBTC(fig,axis3,"16 November 2020",ETH3_plot['K']/458,ETH3_plot['T']/365,(ETH3_plot[model]-np.min(ETH3_plot[model])-0.1)/2)

