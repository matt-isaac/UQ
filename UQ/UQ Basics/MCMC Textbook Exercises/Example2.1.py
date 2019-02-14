# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:17:49 2019

MCMC Textbook, Example 2.1 (p.44)

@author: misaa
"""

import numpy as np
import matplotlib.pyplot as plt

def calc_tau(n, sig2, t0):
    return(((n/sig2) + (1/t0))**-1)
    
def calc_mu(n, xbar, sig2, mu0, t0, t1):
    return(t1 * (((n*xbar)/sig2) + (mu0/t0)))
    
def mu_post(xs, sig2, mu0, t0):
    n = len(xs)
    xbar = np.mean(xs)
    
    t1 = calc_tau(n = n, sig2 = sig2, t0 = t0)
    mu1 = calc_mu(n = n, xbar = xbar, sig2 = sig2, mu0 = mu0, t0 = t0, t1 = t1)
    
    t1 = np.round(t1, 4)
    mu1 = np.round(mu1, 4)
    
    return((mu1, t1))
    

var = 2
mean = 5
n_lst = np.arange(100, 10000, 100)

mu0 = 8
tau0 = 0.5

mus = np.zeros(len(n_lst))
taus = np.zeros(len(n_lst))



counter = 0
for n in n_lst:
    x = np.random.normal(mean, np.sqrt(var), size = n)
    pd = mu_post(xs = x, sig2 = var, mu0 = mu0, t0 = tau0)
    mus[counter] = pd[0]
    taus[counter] = pd[1]
    counter = counter + 1
    print(str(counter) + ". n = " + str(n) + ", mu: " + str(pd[0]) + ", tau2: " + str(pd[1]))
    

plt.plot(n_lst, mus)
plt.show()