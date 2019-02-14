# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:32:38 2019

@author: misaa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def test_mh(pop_size, samp_size):
    
    # generate population and select sample
    population = np.random.normal(15, 2, size = pop_size)
    sample = population[np.random.randint(0, pop_size, samp_size)]
    
    plt.hist(sample, bins = 35)
    plt.xlabel = ("Value")
    plt.ylabel = ("Frequency")
    plt.title = ("Distribution of Sample")
    plt.show()
    
    accepted, rejected = run_mh(likelihood_computer = log_like_normal, 
                            prior = prior, 
                            transition_model = transition_model,
                            param_init = [np.mean(sample),0.1], 
                            iterations = 50000, 
                            data = sample, 
                            acceptance_rule = acceptance)
    return(accepted, rejected)

# transition model - defines how to move from sigma_current to sigma_new
def transition_model(x):
    return([x[0], np.random.normal(x[1], 0.5, (1,))])

def prior(x):
    if(x[1] <= 0):
        return(0)
    return(1)
    
def log_like_normal(x, data):
    return np.sum(-np.log(x[1] * np.sqrt(2 + np.pi)) - ((data - x[0])**2) / (2 * x[1]**2))

def acceptance(x, x_new):
    if(x_new > x):
        return(True)
    else:
        accept = np.random.uniform(0,1)
        return(accept < (np.exp(x_new - x)))

def run_mh(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    
    x = param_init
    accepted = []
    rejected = []
    
    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        if (acceptance(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))): 
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)
            
    return(np.array(accepted), np.array(rejected))



accept, reject = test_mh(30000, 1000)
