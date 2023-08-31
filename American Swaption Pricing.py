import numpy as np
import time
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy import optimize
import math
from scipy.optimize import least_squares
from scipy.special import laguerre
import seaborn as sns

def laguerre_5(x):
    basis = np.zeros((len(x),6))
    x = x.T
    basis[:,0] = 1*np.exp(-0.5*x)
    basis[:,1] = (-x + 1)*np.exp(-0.5*x)
    basis[:,2] = (1/2) * (x**2 - 4*x + 2)*np.exp(-0.5*x)
    basis[:,3] = (1/6) * (-x**3 + 9*x**2 - 18*x + 6)*np.exp(-0.5*x)
    basis[:,4] = (1/24) * (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)*np.exp(-0.5*x)
    basis[:,5] = (1/120) * (-x**5 + 25*x**4 -200*x**3 + 600*x**2 - 600*x +120)

    return basis

def eulerMethod(x, rho, T, m, n):
    dt = T/m
    Rswap = np.zeros((m+1,n))
    Rswap[0,:] = x
    for i in range(1,m+1):
        Rswap[i,:] = Rswap[i-1,:] + Rswap[0,:] * rho * np.random.normal(0,1,n) * np.sqrt(dt)
    return Rswap


# Function to calculate Blacks formula for Payer and Receiver European Swaption
# It is set to $1 needs to be multiplied up for starting capital

def Black(Rswap, K, rho, delta, t, T, ZCbond):

    dt = T - t
    d1 = (np.log(Rswap/K) + 0.5 * rho**2 * dt)/(rho * np.sqrt(dt))
    d2 = (np.log(Rswap/K) - 0.5 * rho**2 * dt)/(rho * np.sqrt(dt))

    Swpt_p = delta*(Rswap*stats.norm.cdf(d1, loc = 0, scale = 1)-K*stats.norm.cdf(d2, loc = 0, scale = 1)) * ZCbond
    Swpt_r = delta*(K*stats.norm.cdf(-d2, loc = 0, scale = 1)-Rswap*stats.norm.cdf(-d1, loc = 0, scale = 1)) * ZCbond

    return Swpt_p, Swpt_r


# Zero Coupon Bond
# Function to create a simple zero coupon bond series from t to T
def ZCBond(T,n,r):
    TS = np.linspace(0,T,n)
    P = np.e**(-r*TS)
    return P


# Function to calculate the payoff for a swap
def payoff(N,Rswap,K,P,delta):
    return N *  delta * np.sum(P)*(Rswap-K)


# Function to price American Swaption using LSM

def LSM(Nominal,N_Simulations,m,SwapRate,StrikeRate,RF_Rate,Variance,Exp_T,Swap_T,N_Payments,Delta_T,Rswap_Sim):

    Rswap = []
    for i in range(int(1/Delta_T)):
        Rswap.append(Rswap_Sim[:][int(m*Delta_T*(i+1))])
    Rswap = np.array(Rswap)
    P = ZCBond(Exp_T+Swap_T,(Exp_T+Swap_T)*N_Payments,RF_Rate)
    ET = np.full(N_Simulations,Exp_T/Delta_T)

    Y = Rswap[:][-1] - StrikeRate
    Y[Y<0] = 0

    ExerciseRate = Y

    for i in range(1,int(Exp_T/Delta_T)):
        
        Index = [j for j, v in enumerate(Rswap[:][-i-1]) if v > StrikeRate]
        X = np.copy(Rswap[:][-i-1])
        X[X<StrikeRate] = 0
        CF = Y[Index]

        #CF = Y[Index] * np.e**(-RF_Rate*Delta_T)

        #model = np.poly1d(np.polyfit(X[X>0], CF, 2))

        #Cont = model(X[X>0])

        basis = laguerre_5(X[X>0])
        
        regression_coeffs = np.linalg.lstsq(basis, CF)[0]
        regression_values = basis @ regression_coeffs

        Cont = regression_values

        Continuation = np.zeros(len(X))
        np.put(Continuation,Index,Cont)

        
        IndexE = [i for i, v in enumerate(X-Continuation) if v > StrikeRate]
        
        if len(IndexE)>0:
            ET[IndexE] = int(Exp_T/Delta_T) - i
            Y[IndexE] = X[IndexE]-StrikeRate # Sub in differential rate exercised 
            ExerciseRate[IndexE] = X[IndexE]-StrikeRate # Sub into exercise time diff

        if float((i+1)*N_Payments*Delta_T).is_integer():
            Y = Y * np.e**(-RF_Rate*Swap_T/N_Payments) # Discount Interest Rate differential back
    
    IndexExercised = [i for i, v in enumerate(ExerciseRate) if v > 0]

    ExSwapRate = ExerciseRate[ExerciseRate>0] + StrikeRate

    ET = ET[IndexExercised]

    PD = np.ceil(ET*N_Payments*Delta_T)

    V = []
    for i in range(len(ET)):
        V.append(payoff(1,ExSwapRate[i],StrikeRate,P[int(PD[i]):int(PD[i]+N_Payments)],Swap_T/N_Payments))
    return Nominal*sum(V)/N_Simulations

#Function to price American Swaptions
# Code to fit an asymptotic distribution
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def ASwaption(MC_Sims,LSM_Sims,EM_Steps,dT_Depth, SwapRate, StrikeRate, RF_Rate, Variance, ExpT, SwapT, Swap_Pay):
    dT = [1,1/4,1/16,1/64,1/256]
    MC = np.zeros((len(dT),MC_Sims))

    for i in range(MC_Sims):
        Rswap_Sim = eulerMethod(SwapRate,Variance,ExpT,EM_Steps,LSM_Sims)
        for j in range(len(dT)):
            MC[j][i] = LSM(1,LSM_Sims,EM_Steps,SwapRate, StrikeRate,RF_Rate,Variance,ExpT,SwapT,Swap_Pay,dT[j],Rswap_Sim)
    
    Mean = np.mean(MC,axis=1)
    print(Mean)
    CI = 1.96 * np.std(MC, axis=1) / np.sqrt(MC_Sims)

    xdata = np.linspace(0, 32, 5000)
    Ex = -np.log(dT)/np.log(4)

    fit_data, covariance = scipy.optimize.curve_fit(func, Ex, Mean)
    fit_data_low, covariance_low = scipy.optimize.curve_fit(func, Ex, Mean-CI)
    fit_data_high, covariance_high = scipy.optimize.curve_fit(func, Ex, Mean+CI) 

    return np.max(func(np.array(xdata).astype(float),*fit_data)), np.max(func(np.array(xdata).astype(float),*fit_data_low)), np.max(func(np.array(xdata).astype(float),*fit_data_high))

PLS = ASwaption(10,1000,4**5,5,0.05,0.05,0.05,0.2,1,1,4)
print(PLS)
