#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:39:14 2024

@author: zhangjiyu
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.stats import norm
from scipy.special import binom
import random

#Set seed sothat the result is reproducible.
seed_value = 42
random.seed(seed_value)
#1 Inputting data 
def input_data():
    """
    This is a function responsible for inputting data and it can deal with 
    situations when an inputted parameter is clearly wrong.

    Returns
    -------
    tuple
        The tuple contains all of the parameter values.
    """
    #Ask user to input the value of parameters.
    S_0 = input("Please input the initial price of the stock:")
    r = input("Please input the risk-free interest rate r(per annum):")
    sigma = input("Please input the the volatility of the stock price σ(per annum):")
    T = input("Please in input the expiry time T of the option:")
    K = input("Please in input the strike price K of the option:")
    M = input("Please in input number of periods in the binomial tree, M:")
    
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    #Verify the correctness of parameters.
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    return S_0, r, sigma, T, K, M

#2 The binomial model
def compute_binomial_algorithm1_2(S_0,r,sigma,T,K,M):
    """
    This is a function that implement the pricing of a vanilla European put 
    option on a recombining binomial tree based on Algorithm 1.2 using 
    approximate CRR method. In this part, we use matrix to store the option 
    prices. This function performs calculations using for loops.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.
    M : float or int
        Number of periods in the binomial tree.

    Returns
    -------
    float
        The price of the vanilla European put option at time 0.
    """
    #Verify the correctness of the parameter values.
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    
    #Calculate the time of each periods.
    dt=T/M
    
    #Use approximate CRR method to estimate parameters u and d.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-1 * sigma * math.sqrt(dt))
    
    #Calculate the risk neutral measure.
    ps=(np.exp(r*dt)-d)/(u-d)
    
    #Check if it is arbitrage free.
    assert 0<=ps<=1, 'Error: not arbitrage free, ensure T/M<(sigma/r)^2'
        
    #Set an empty matrix to store the option prices.
    V=np.zeros((M+1,M+1))
    
    #Calculate and store the payoffs at maturity.
    for j in range(M+1):
        S=S_0*u**j*d**(M-j)
        V[j,M]=np.maximum(K-S,0)
        
    #Recurse calculation of option prices by r.n.m. and terminal payoffs.
    for i in range(M-1,-1,-1):
        for j in range(i+1):
            V[j,i]=np.exp(-r*dt)*(ps*V[j+1,i+1]+(1-ps)*V[j,i+1])
    return V[0,0]


#3 Verification
def Black_Scholes(S_0,r,sigma,T,K):
    '''
    This is a function that implement the pricing of a vanilla European put 
    option by Black Scholes model.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.

    Returns
    -------
    put_price : float
        The price of the vanilla European put option at time 0.
    '''
    #Transform the Normal distribution into standard Normal distribution.
    d1=(np.log(S_0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    
    #Calculate the option price at time 0.
    put_price=K*np.exp(-r*T)*norm.cdf(-d2)-S_0*norm.cdf(-d1)
    return put_price

#Use the two algorithms to calculate option prices under the same condition.
A = compute_binomial_algorithm1_2(80, 0.025, 0.3, 1.5, 90, 50)
B = Black_Scholes(80, 0.025, 0.3, 1.5, 90)
print(A,B)


#4 Computational complexity
#Set the minimum value and the maximum value of M.
start = 2
stop = 500

#Set values of M.
M_values=np.linspace(start,stop,stop-start)

#Calculate the time to show the complexity of algorithm 1.2 
Time = []
for i in range(start, stop):
    time_start = time.perf_counter()
    compute_binomial_algorithm1_2(80, 0.025, 0.3, 1.5, 90, i)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time.append(totaltime)

#Calculate the slope.
coeffs_1 = np.polyfit(np.log(M_values), np.log(Time), 1)
slope_1 = coeffs_1[0]
print("The loglog plot slope of algerithm1.2 is", slope_1)

M_values = np.arange(start,stop,1)
reference_line1 =2 *(M_values /M_values[0])**2 * Time[1]

#Plot to show the complexity for different M values.
plt.figure()
plt.loglog(M_values, Time, markersize=2, label="Algorithm1.2")
plt.loglog(M_values, reference_line1, linestyle='--', 
           label='Reference Line (slope 2)')
plt.title('European put Option pricing times')
plt.xlabel('M')
plt.ylabel('Time')
plt.legend()


#5 Faster algorithms
def compute_binomial_single_array(S_0,r,sigma,T,K,M):
    """
    This is a function that implement the pricing of a vanilla European put 
    option on a recombining binomial tree based on Algorithm 1.2 using 
    approximate CRR method. In this part, we use a single list to store the 
    option prices. This function performs calculations using for loops.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.
    M : float or int
        Number of periods in the binomial tree.

    Returns
    -------
    float
        The price of the vanilla European put option at time 0.
    """
    #Verify the correctness of the parameter values.
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    
    #Calculate the time of each periods.
    dt=T/M
    
    #Use approximate CRR method to estimate parameters u and d.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-1 * sigma * math.sqrt(dt))
    
    #Calculate the risk neutral measure.
    ps=(np.exp(r*dt)-d)/(u-d)
    
    #Check if it is arbitrage free.
    assert 0<=ps<=1, 'Error: not arbitrage free, ensure T/M<(sigma/r)^2'
    
    #Set an empty array to store option prices.
    V=np.zeros(M+1)
    
    #Calculate and store the payoffs at maturity.
    for j in range(M+1):
        S=S_0*u**j*d**(M-j)
        V[j]=np.maximum(K-S,0)
    
    #Recurse calculation of option prices by r.n.m. and terminal payoffs.
    for i in range(M-1,-1,-1):
        for j in range(i+1):
            V[j]=np.exp(-r*dt)*(ps*V[j+1]+(1-ps)*V[j])
    return V[0]


def compute_binomial_vectorised(S_0,r,sigma,T,K,M):
    """
    This is a function that implement the pricing of a vanilla European put 
    option on a recombining binomial tree based on Algorithm 1.2 using 
    approximate CRR method. In this part, we use a single list to store the 
    option prices. This function performs vectorised calculations.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.
    M : float or int
        Number of periods in the binomial tree.

    Returns
    -------
    float
        The price of the vanilla European put option at time 0.
    """
    #Verify the correctness of the parameter values.
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    
    #Calculate the time of each periods.
    dt=T/M
    
    #Use approximate CRR method to estimate parameters u and d.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-1 * sigma * math.sqrt(dt))
    
    #Calculate the risk neutral measure.
    ps=(np.exp(r*dt)-d)/(u-d)
    
    #Check if it is arbitrage free.
    assert 0<=ps<=1, 'Error: not arbitrage free, ensure T/M<(sigma/r)^2'
    
    #Set empty arrays to store values of parameters.
    V=np.zeros(M+1)
    ST=np.zeros(M+1)
    j=np.arange(M+1)
    
    #Calculate the terminal stock prices.
    ST=S_0*u**j*d**(M-j)
    
    #Store the terminal payoffs into vector.
    V=np.maximum(K-ST,0)
    
    #Calculate the option price at time 0.
    for i in range(M):
        V=np.exp(-r*dt)*(ps*V[1:]+(1-ps)*V[:-1])
    
    return V[0]


def compute_binomial_binomial(S_0,r,sigma,T,K,M):
    """
    This is a function that implement the pricing of a vanilla European put 
    option on a recombining binomial tree based on Algorithm 1.2 using 
    approximate CRR method. In this part, we use a single list to store the 
    option prices. This function performs vectorised calculations and 
    binomial coefficients.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.
    M : float or int
        Number of periods in the binomial tree.

    Returns
    -------
    float
        The price of the vanilla European put option at time 0.
    """
    #Verify the correctness of the parameter values.
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    
    #Calculate the time of each periods.
    dt=T/M
    
    #Use approximate CRR method to estimate parameters u and d.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-1 * sigma * math.sqrt(dt))
    
    #Calculate the risk neutral measure.
    ps=(np.exp(r*dt)-d)/(u-d)
    
    #Check if it is arbitrage free.
    assert 0<=ps<=1, 'Error: not arbitrage free, ensure T/M<(sigma/r)^2'
    
    #Set empty arrays to store values of parameters.
    V=np.zeros(M+1)
    ST=np.zeros(M+1)
    P=np.zeros(M+1)
    prob=np.zeros(M+1)
    j=np.arange(M+1)

    #Calculate the terminal stock prices.
    ST=S_0*u**j*d**(M-j)
    
    #Store the terminal payoffs into vector.
    V=np.maximum(K-ST,0)
    
    P=ps**(j)*(1-ps)**(M-j)
    prob=binom(M, j)
    #Calculate the option price at time 0.
    value=np.exp(-r*T)*np.sum(P * prob * V)
    return value


#Calculate the time and slopes to show the complexities of algorithms.
Time_array = []
for i in range(start, stop):
    time_start = time.perf_counter()
    compute_binomial_single_array(80, 0.025, 0.3, 1.5, 90, i)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_array.append(totaltime)
    
coeffs_2 = np.polyfit(np.log(M_values), np.log(Time_array), 1)
slope_2 = coeffs_2[0]

Time_array_vectorised = []
for i in range(start, stop):
    time_start = time.perf_counter()
    compute_binomial_vectorised(80, 0.025, 0.3, 1.5, 90, i)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_array_vectorised.append(totaltime)
    
coeffs_3 = np.polyfit(np.log(M_values), np.log(Time_array_vectorised), 1)
slope_3 = coeffs_3[0]

Time_array_vectorised_bin = []
for i in range(start, stop):
    time_start = time.perf_counter()
    compute_binomial_binomial(80, 0.025, 0.3, 1.5, 90, i)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_array_vectorised_bin.append(totaltime)
    
coeffs_4 = np.polyfit(np.log(M_values), np.log(Time_array_vectorised_bin), 1)
slope_4 = coeffs_4[0]

    

print("The loglog plot slope of algerithm1.2(single array) is", slope_2)
print("The loglog plot slope of algerithm1.2(vectorised) is", slope_3)
print("The loglog plot slope of algerithm1.2(vectorised+binom) is", slope_4)

reference_line2 =(M_values /M_values[0]) * Time_array_vectorised[1]
reference_line3 =0.5 *(M_values /M_values[0])**0.5 * Time_array_vectorised_bin[10]
#Show the loglog plots to compare the comlexities.
plt.figure()
plt.loglog(M_values, Time, markersize=1, label="Algarithm 1.2")
plt.loglog(M_values, Time_array, markersize=1, label="Array algarithm")
plt.loglog(M_values, Time_array_vectorised, markersize=1,
           label="Vectorised algarithm")
plt.loglog(M_values, Time_array_vectorised_bin, markersize=1,
           label="Vectorised algarithm by binomial coefficients")
plt.loglog(M_values, reference_line1, linestyle='--', 
           label='Reference Line (slope 2)')
plt.loglog(M_values, reference_line2, linestyle='--',
           label='Reference Line (slope 1)')
plt.loglog(M_values, reference_line3, linestyle='--',
           label='Reference Line (slope 0.5)')
plt.title('Comparison of the runtime between two algorithms')
plt.xlabel('M')
plt.ylabel('Time')
plt.legend(fontsize='7')



#6 Tilted tree and Richardson Extrapolation
prices_exact = []
prices_vectorised_odd = []
prices_vectorised_even = []
for i in range(start, stop):
    price_vectorised = compute_binomial_vectorised(80, 0.025, 0.3, 1.5, 90, i)
    if i%2!=0:
        price_exact=Black_Scholes(80, 0.025, 0.3, 1.5, 90)
        prices_exact.append(price_exact)
        prices_vectorised_odd.append(price_vectorised)
    if i%2==0:
        prices_vectorised_even.append(price_vectorised)

plt.figure()
plt.plot(range(len(prices_vectorised_odd)), prices_vectorised_odd, 
         label="The option prices of odd M")
plt.plot(range(len(prices_vectorised_even)), prices_vectorised_even, 
         label="The option prices of even M")
plt.plot(range(len(prices_exact )), prices_exact , 
         label=" The exact option price as a function of M")
plt.title('The option prices without tilted tree')
plt.xlabel('M')
plt.ylabel('Prices')
plt.legend(fontsize='7')
plt.show()

def compute_binomial_tilted_tree(S_0,r,sigma,T,K,M):
    """
    This is a function that implement the pricing of a vanilla European put 
    option on a recombining binomial tree applied tilted tree method. In this 
    part, we use a single list to store the option prices. This function 
    performs vectorised calculations.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.
    M : float or int
        Number of periods in the binomial tree.

    Returns
    -------
    float
        The price of the vanilla European put option at time 0.
    """
    #Verify the correctness of the parameter values.
    try:
        S_0 = float(S_0)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
        K = float(K)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    for param_name, param_value in [('S_0', S_0), ('r', r), ('sigma', sigma), 
                                    ('T', T), ('K', K), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
    
    #Calculate the time of each periods.
    dt=T/M
    
    #Use tilted tree method to estimate parameters u and d.
    x = np.log(K / S_0) / M
    u = np.exp(sigma * math.sqrt(dt)+x)
    d = np.exp(-1 * sigma * math.sqrt(dt)+x)
    
    #Calculate the risk neutral measure.
    ps=(np.exp(r*dt)-d)/(u-d)
    
    #Check if it is arbitrage free.
    assert 0<=ps<=1, 'Error: not arbitrage free, ensure T/M<(sigma/r)^2'
   
    #Set empty arrays to store values of parameters.
    V=np.zeros(M+1)
    ST=np.zeros(M+1)
    j=np.arange(M+1)
    
    #Calculate the terminal stock prices.
    ST=S_0*u**j*d**(M-j)
    
    #Store the terminal payoffs into vector.
    V=np.maximum(K-ST,0)
    
    #Calculate the option price at time 0.
    for i in range(M):
        V=np.exp(-r*dt)*(ps*V[1:]+(1-ps)*V[:-1])
    
    return V[0]


#Create empty lists to store option prices for different scenarios.
prices_exact = []
prices_odd = []
prices_even = []
prices_even_Rich = []
prices_odd_Rich = []

#Empty lists to store absolute errors for different Richardson extrapolation.
Abs_error_Rich_even = []
Abs_error_Rich_odd = []


for i in range(start, stop):
    #The original price.
    price = compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, i)
    
    #The price of Black Sholes model.
    price_exact=Black_Scholes(80, 0.025, 0.3, 1.5, 90)
    
    #Richardson extrapolation when M is odd.
    if i%2!=0:
        #Approach 1.
        prices_odd.append(price)
        prices_exact.append(price_exact)
        #Calculate the price of (3M).
        b=int(i*3)
        #Calculate the price of (3M).
        price_3=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, b)
        price_Rich_odd = (3*price_3 - price)/2
        prices_odd_Rich.append(price_Rich_odd)
        
        #Absolute error of these two approaches when odd M.
        Abs_error_Rich_odd.append(abs(price_Rich_odd-price_exact))
    
    #Richardson extrapolation when M is even.
    if i%2==0:
        a=int(i*2)
        prices_even.append(price)
        #Calculate the price of (2M).
        price_2 = compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, a)
        price_Rich_even = 2*price_2 - price
        
        #Absolute error when even M.
        Abs_error_Rich_even.append(abs(price_Rich_even-price_exact))

#Plot to compare the prices.
plt.figure()
plt.plot(range(len(prices_odd)), prices_odd, 
         label=" The option price as a function of odd M")
plt.plot(range(len(prices_even)), prices_even, 
         label=" The option price as a function of even M")
plt.plot(range(len(prices_even_Rich)), prices_even_Rich, 
         label=" The option price as a function of even M with Richardson extrapolation")
plt.plot(range(len(prices_odd_Rich)), prices_odd_Rich, 
         label=" The option price as a function of odd M with Richardson extrapolation")
plt.plot(range(len(prices_exact )), prices_exact , 
         label=" The exact option price as a function of M")
plt.title('The option price as a function of M')
plt.xlabel('M')
plt.ylabel('Prices')
plt.legend(fontsize='7')
plt.show()

#Plot to compare the absolute errors.
plt.figure()
plt.plot(range(len(Abs_error_Rich_even)), Abs_error_Rich_even, 
         label=" The option price as a function of even M with Richardson extrapolation")
plt.plot(range(len(Abs_error_Rich_odd)), Abs_error_Rich_odd, 
         label=" The option price as a function of odd M with Richardson extrapolation")
plt.title('The option price absolute error as a function of M')
plt.xlabel('M')
plt.ylabel('Absolute Errors of Prices')
plt.legend(fontsize='8')
plt.show()


Aberror_Rich_odd_1 = []
Aberror_Rich_odd_2 = []
Aberror_Rich_odd_3 = []

for i in range(start, stop):
    price = compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, i)
    if i%2!=0:
        a=int(i*3)
        prices_odd.append(price)
        price_exact=Black_Scholes(80, 0.025, 0.3, 1.5, 90)
        prices_exact.append(price_exact)
        price_2=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, a)
        price_Rich_odd = (3*price_2 - price)/2
        
        b=int(i*5)
        price_3=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, b)
        price_Rich_odd_new = (5*price_3 - price)/4
        
        c=int(i*11)
        price_4=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, c)
        price_Rich_odd_new_new = (11*price_4 - price)/10
        
        
        Aberror_Rich_odd_1.append(abs(price_Rich_odd-price_exact))
        Aberror_Rich_odd_2.append(abs(price_Rich_odd_new-price_exact))
        Aberror_Rich_odd_3.append(abs(price_Rich_odd_new_new-price_exact))

plt.figure()
plt.plot(range(len(Aberror_Rich_odd_1)), Aberror_Rich_odd_1, 
         label=" The option price as n=1")
plt.plot(range(len(Aberror_Rich_odd_2)), Aberror_Rich_odd_2 , 
         label=" The option price as n=2")
plt.plot(range(len(Aberror_Rich_odd_3)), Aberror_Rich_odd_3 , 
         label=" The option price as n=5")
plt.title('The option price absolute error with R.E. as a function of odd M')
plt.xlabel('M')
plt.ylabel('Absolute Errors of Prices')
plt.legend(fontsize='8')
plt.show()


Aberror_Rich_even_1 = []
Aberror_Rich_even_2 = []
Aberror_Rich_even_3 = []

for i in range(start, stop):
    price = compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, i)
    if i%2!=0:
        a=int(i*2)
        prices_even.append(price)
        price_exact=Black_Scholes(80, 0.025, 0.3, 1.5, 90)
        prices_exact.append(price_exact)
        price_2=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, a)
        price_Rich_even = 2*price_2 - price
        
        b=int(i*6)
        price_3=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, b)
        price_Rich_even_new = (6*price_3 - price)/5
        
        c=int(i*12)
        price_4=compute_binomial_tilted_tree(80, 0.025, 0.3, 1.5, 90, c)
        price_Rich_even_new_new = (12*price_4 - price)/11
        
        
        Aberror_Rich_even_1.append(abs(price_Rich_even-price_exact))
        Aberror_Rich_even_2.append(abs(price_Rich_even_new-price_exact))
        Aberror_Rich_even_3.append(abs(price_Rich_even_new_new-price_exact))

plt.figure()
plt.plot(range(len(Aberror_Rich_even_1)), Aberror_Rich_even_1, 
         label=" The option price as n=1")
plt.plot(range(len(Aberror_Rich_even_2)), Aberror_Rich_even_2 , 
         label=" The option price as n=3")
plt.plot(range(len(Aberror_Rich_even_3)), Aberror_Rich_even_3 , 
         label=" The option price as n=6")
plt.title('The option price absolute error with R.E. as a function of even M')
plt.xlabel('M')
plt.ylabel('Absolute Errors of Prices')
plt.legend(fontsize='8')
plt.show()