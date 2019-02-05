cl# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:17:49 2019

Accessed 2/5

MCMC Textbook, Example 2.1 (p.44)

@author: misaa
"""

import numpy as np

def likelihood_func(x_array, mu, sigma2):
    terms = np.zeros(len(x_array))
    t1 = 1/np.sqrt(2 * np.pi * sigma2)
    counter = 0
    for x in x_array:
        t2 = -0.5 * ((x - mu)**2 / sigma2)
        terms[counter] = t1 * np.exp(t2)
        counter = counter + 1
    lik = np.prod(terms)
    return(lik)
    
def update_tau2(n, sigma2, tau2_prev):
    return(n * (1/sigma2) + (1/tau2_prev))
    
def update_mu(n, tau2_1, sigma2, xbar, mu_prev):
    return(tau2_1 * (n * (1/sigma2) * xbar + (1/tau2_1) * mu_prev))
    
x = np.random.normal(5, 0.5, size = 100)
        
update_mu(n = 100, tau2_1 = 2, sigma2 = 0.5, xbar = np.mean(x), mu_prev = 2) 

       
    