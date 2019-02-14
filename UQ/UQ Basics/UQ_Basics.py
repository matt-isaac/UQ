# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:29:23 2018

@author: misaa
"""
# %matplotlib auto

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

def srq1(a1, a2, a3, e1, e2, e3):
    """
    Return the System Response Quantity (SRQ) for the given input paramters
    
    Parameters
    ----------
    input_vec : dictionary
        dictionary of input values
        
    Returns
    -------
    srq : numeric
        The SRQ resulting from the given parameters
    """

    srq = ((e1**e2) * (a3 * a2**a1)) + e3
    
    return(srq)
    
def required_keys(ref_dict, test_dict):
    """
    Checks to verify all items in key_list are keys in dictionary
    
    Parameters
    ----------
    key_list : list
        list of keys that are expected to be in dictionary
        
    dictionary : dict
        dictionary that is expected to contain keys in key_list
        
    Returns
    -------
    result : logical
        Returns True if all keys in key_list exist in dictionary, returns 
        False otherwise. 
    """
    for k,v in ref_dict:
        if k not in test_dict:
            return False
    return True

def sample_inputs(dist_dict):
    """
    Draws a random value from each distribution in dist_dict
    
    Parameters
    ----------
    dist_dict : dict
        dictionary of distributions. Key should correspond to the name of the 
        variable the distribution is associated with.
        
    Returns
    -------
    sample_dict : dict
        dictionary containing random values from distributions specified in 
        dist_dict. Keys of dictionary are the same values as the keys in 
        dist_dict. 
    """
    sample_dict = {} 
    for k, v in dist_dict.items():
	    sample_dict[k] = v.rvs()
    return(sample_dict)    

def calc_cdf(arr, resolu, num_cdfs, num_tosses = None):
    """
    Calculates cdf for an array of SRQ values. Used in 2D UQ. 
    
    Parameters
    ----------
    arr : numpy array
        array of srqs. Row should be the results from one inner loop in 2D UQ. 
    
    resolu : int
        specify the number of values over which cdf should be evaluated (and 
        eventually plotted)
        
    num_cdfs : int
        number of outer loops for a 2D UQ analysis, 
        number of outer loops * number of 'tosses' for 3D UQ analysis
        
    num_tosses : int
        optional argument, to be used for '3D' UQ
    
    Returns
    -------
    A tuple containing an array of cdfs, and x values over which cdfs were 
    evaluated. Can be used for plotting. 
    
    """
    srq_max = np.max(arr) # max value in entire array
    srq_min = np.min(arr) # min value in entire array
    
    x = np.linspace(srq_min, srq_max, num = resolu) # x grid over which to evaluate cdf 
    cdf_arr = np.zeros((num_cdfs,resolu)) # empty array to store results
    counter = 0    
    for row in arr:
        ecdf = sm.distributions.ECDF(row) # calculate emperical cdf
        y = ecdf(x) # evaluate cdf over grid of xs. 
        cdf_arr[counter] = y # store results
        counter = counter + 1
    return(cdf_arr, x) 

def plot_cdfs(cdf_arr, xs, dim, pbox_upper = 0.95, pbox_lower = 0.05, title = None):
    """
    Creates plot of cdfs in cdf_arr.
    
    Parameters
    ----------
    cdf_arr : numpy array
         CDF's should be rows (0th axis) of array
         
    xs : numpy array
        Create using np.linspace(). Should be same length as rows in cdf_arr
        
    dim : integer
        Must be either 1 or 2. Indicates whether the results of 1-D or 2-D UQ 
        results are being plotted.
        
    pbox_upper : float
        Must be between 0 and 1. Indicates quantile for which to draw upper
        pbox bound.
        
    pbox_lower : float
        Must be between 0 and 1. Indicates quantile for which to draw lower
        pbox bound.       
    """
    fig = plt.figure()
#    p1 = fig.add_subplot(1,3,1)
#    p2 = fig.add_subplot(1,3,2)
#    p3 = fig.add_subplot(1,3,3)
    p3 = fig.add_subplot(1,1,1)
    
    if dim == 1:
        p3.plot(xs, cdf_arr, color = 'tab:blue', alpha=1)
        plt.show()
    
    else:
        for cdf in cdf_arr:
    #        p1.plot(cdf[0], cdf[1], color = 'tab:blue', alpha=1)
    #        p2.plot(cdf[0], cdf[1], color = 'tab:blue', alpha=0.5)
            p3.plot(xs, cdf, color = 'tab:blue', alpha=0.2)
            plt.title(title)
        
        # plot p-boxes
        
        ql = np.zeros_like(cdf_arr[0]) # container for lower quantile
        qu = np.zeros_like(cdf_arr[0]) # container for upper quantile
        counter = 0 # counter for computing quantiles
        for row in np.transpose(cdf_arr): # loop over transposed matrix
            qlower = np.quantile(row, q = pbox_lower) # calculate lower quantile
            qupper = np.quantile(row, q = pbox_upper) # calculate upper quantile
            ql[counter] = qlower # store
            qu[counter] = qupper # store
            counter = counter + 1 
        plt.plot(xs, ql, color = 'tab:red', linewidth = 2) # plot lower quantile
        plt.plot(xs, qu, color = 'tab:red', linewidth = 2) # plot upper quantile
        plt.title(title)
        plt.show()
        return(ql, qu)
    
    
def one_dim_uq(num_trials, dist_dict, calc_srq, resolu = 1000):     
    """
    Returns a cdf for the given SRQ
    
    Parameters
    ----------
    num_trials : numeric
        an integer indicating the number of trials to be performed in the
        uncertainty quantification. '
        
    dist_dict : dictionary
        dictionary of distribution. Keys should be the names of the input 
        variables. 
        
    resolu : integer
        specify how many points cdf should be evaluated and plotted at. 
        
    Returns
    -------
    cdf : numeric 1D array
        a array containing the values of the cdf (to be plotted)
    """    
    srqs = np.zeros(num_trials) # empty container to store srq values
    counter = 0 
    for i in range(num_trials):
        inputs = sample_inputs(dist_dict) # sample from various distributions
        result = calc_srq(**inputs) # calculate value of srq from sampled inputs
        srqs[counter] = result # store 
        counter = counter + 1
    
    srq_max = np.max(srqs) # find max and min for plotting purposes
    srq_min = np.min(srqs)
    x = np.linspace(srq_min, srq_max, num = resolu) # x grid over which to evaluate cdf.
    ecdf = sm.distributions.ECDF(srqs) # calculate emperical cdf
    cdf = ecdf(x) # evaluate cdf over x grid
    plot_cdfs(cdf, x, dim = 1) # plot cdf
    return(cdf)   
    
def two_dim_uq(num_inner, num_outer, dist_dict_aleatory, dist_dict_epistemic, 
               calc_srq, title = None):
    """
    Perform 2-D UQ. Return an ensemble of cdfs for the given SRQ
    
    Parameters
    ----------
    num_inner : numeric
        an integer indicating the number of iterations to run on the inner
        (Aleatory) Monte Carlo loop
        
    num_outer : numeric
       an integer indicating the number of iterations to run on the outer
        (Epistemic) Monte Carlo loop 
        
    title : string
        optional title to be included on plot
        
    Returns
    -------
    cdf_list : list of numeric lists
        A list (ensemble) of calculated cdfs
    """    
    cdf_resolu = 1000 # resolution of cdf (how many points to evaluate cdf at)
    srqs_arr = np.zeros((num_outer, num_inner)) # array of all srqs
    count_outer = 0 # counter for outer loop
    for i in range(0, num_outer, 1):
        ep_inputs = sample_inputs(dist_dict = dist_dict_epistemic) # Sample epistemic variables
        srqs = np.zeros(num_inner) # initialize empty array to store srq values
        count_inner = 0 # counter for inner loop
        for j in range(0, num_inner, 1):
            al_inputs = sample_inputs(dist_dict = dist_dict_aleatory) # Sample aleatory variables
            inputs = {**al_inputs, **ep_inputs} # Create input dictionary
            result = calc_srq(**inputs) # Calculate srq value
            srqs[count_inner] = result # store srq value
            count_inner = count_inner + 1 # increment inner counter
        srqs_arr[count_outer] = srqs # store srqs from inner loop in sqrs_arr
        
        count_outer = count_outer + 1 # increment outer loop
    
    cdfs_arr,xs = calc_cdf(srqs_arr, cdf_resolu, num_outer) # calculate emperical cdfs
   
    ql, qu = plot_cdfs(cdfs_arr,xs, dim = 2, title = title)
    return(xs, ql, qu)

def al_ep_toss(a_probs, distns, bounds = None):
    al_dict = {}
    ep_dict = {}
    for variable, prob in a_probs.items():
          result = np.random.binomial(1, prob)  
          if result == 1:
              al_dict[str(variable)] = distns[str(variable)]
          else:
              ep_dict[str(variable)] = distns[str(variable)]
    return(al_dict, ep_dict)                         

def three_dim_uq(num_inner, num_outer, num_toss, a_probs_dict, dist_dict, calc_srq, bound_dict=None, title = None):
    toss_counter = 0
    srq_counter = 0
    srqs_arr = np.zeros((num_outer * num_toss, num_inner)) # array of all srqs 
    
    for toss in range(0, num_toss, 1):
        cdf_resolu = 1000 # resolution of cdf (how many points to evaluate cdf at)
        count_outer = 0 # counter for outer loop
        dist_dict_aleatory, dist_dict_epistemic = al_ep_toss(a_probs_dict, dist_dict)
        print("Toss:" + str(toss_counter))
        print("Aleatory: " + str(list(dist_dict_aleatory.keys())))
        print("Epistemic: " + str(list(dist_dict_epistemic.keys())))
        for i in range(0, num_outer, 1):
            ep_inputs = sample_inputs(dist_dict = dist_dict_epistemic) # Sample epistemic variables
            srqs = np.zeros(num_inner) # initialize empty array to store srq values
            count_inner = 0 # counter for inner loop
            
            for j in range(0, num_inner, 1):
                al_inputs = sample_inputs(dist_dict = dist_dict_aleatory) # Sample aleatory variables
                inputs = {**al_inputs, **ep_inputs} # Create input dictionary
                result = calc_srq(**inputs) # Calculate srq value
                srqs[count_inner] = result # store srq value
                count_inner = count_inner + 1 # increment inner counter
                
            srqs_arr[srq_counter] = srqs # store srqs from inner loop in sqrs_arr
            count_outer = count_outer + 1 # increment outer loop
            srq_counter = srq_counter + 1
        toss_counter = toss_counter + 1
        
    cdfs_arr,xs = calc_cdf(srqs_arr, cdf_resolu, num_outer*num_toss) # calculate emperical cdfs
    ql, qu = plot_cdfs(cdfs_arr,xs, dim = 3, title = title)
    return(xs, ql, qu)

        
#def three_dim_uq(num_inner, num_outer, num_toss, a_probs_dict, dist_dict, calc_srq, bound_dict=None):
#    """
#    a_probs_dict : dictionary
#        dictionary containing the probability of each variable being aleatory    
#    """
#    cdfs = []
#    toss_counter = 0
#    for t in range(0, num_toss, 1):
#        dist_dict_aleatory, dist_dict_epistemic = al_ep_toss(a_probs_dict, dist_dict)
#        
#        for i in range(0, num_outer, 1):
#            ep_inputs = sample_inputs(dist_dict = dist_dict_epistemic)
#            srqs = []
#            for j in range(0, num_inner, 1):
#                al_inputs = sample_inputs(dist_dict = dist_dict_aleatory)
#                inputs = {**al_inputs, **ep_inputs}
#                result = calc_srq(**inputs)
#                srqs.append(result)
#            cdf = calc_cdf(srq_list = srqs)
#            cdfs.append(cdf)
#        print("Toss:" + str(toss_counter))
#        print("Aleatory: " + str(list(dist_dict_aleatory.keys())))
#        print("Epistemic: " + str(list(dist_dict_epistemic.keys())))
#        toss_counter = toss_counter + 1
#    plot_cdfs(cdfs)
#    
#def three_dim_uq_v2(num_inner, num_outer, num_toss, a_probs_dict, dist_dict, calc_srq, bound_dict=None):
#    """
#    a_probs_dict : dictionary
#        dictionary containing the probability of each variable being aleatory    
#    """
#    cdfs = []
##    toss_counter = 0        
#    for i in range(0, num_outer, 1):
#        dist_dict_aleatory, dist_dict_epistemic = al_ep_toss(a_probs_dict, dist_dict)
#        ep_inputs = sample_inputs(dist_dict = dist_dict_epistemic)
#        srqs = []
#        for j in range(0, num_inner, 1):
#            al_inputs = sample_inputs(dist_dict = dist_dict_aleatory)
#            inputs = {**al_inputs, **ep_inputs}
#            result = calc_srq(**inputs)
#            srqs.append(result)
#        cdf = calc_cdf(srq_list = srqs)
#        cdfs.append(cdf)
##    print("Toss:" + str(toss_counter))
##    print("Aleatory: " + str(list(dist_dict_aleatory.keys())))
##    print("Epistemic: " + str(list(dist_dict_epistemic.keys())))
##    toss_counter = toss_counter + 1
#    plot_cdfs(cdfs)

## Unit Tests
def test_1d_uq():
    dist_dict_1d = {'a1': norm(loc = 2, scale = 0.13),
                'a2': norm(loc = 7, scale = 0.55),
                'a3': norm(loc = 50, scale = 0.12),
                'e1':  uniform(loc = 0.5, scale = 0.2),
                'e2': uniform(loc = 1, scale = 0.1),
                'e3': uniform(loc = 100, scale = 10)}
    one_dim_uq(num_trials = 10000, dist_dict = dist_dict_1d, calc_srq = srq1)
     
def test_2d_uq():
    dist_dict_al = {'a1': norm(loc = 2, scale = 0.13),
                    'a2': norm(loc = 7, scale = 0.55),
                    'a3': norm(loc = 50, scale = 0.12)}
    
    dist_dict_ep = {'e1':  uniform(loc = 0.5, scale = 0.2),
                    'e2': uniform(loc = 1, scale = 0.1),
                    'e3': uniform(loc = 100, scale = 10)}

    two_dim_uq(num_inner=700, num_outer=500, 
               dist_dict_aleatory = dist_dict_al, dist_dict_epistemic = dist_dict_ep, calc_srq = srq1)
    
    
def test_3d_uq():
    prob_dict = {'a1': 0.75,
                'a2': 0.8,
                'a3': 0.9,
                'e1': 0.1,
                'e2': 0.15,
                'e3': 0.2}
    
    dist_dict = {'a1': norm(loc = 2, scale = 0.13),
                'a2': norm(loc = 7, scale = 0.55),
                'a3': norm(loc = 50, scale = 0.12),
                'e1':  uniform(loc = 0.5, scale = 0.2),
                'e2': uniform(loc = 1, scale = 0.1),
                'e3': uniform(loc = 100, scale = 10)}
    
    three_dim_uq(num_inner = 1000, num_outer=100, num_toss = 10, 
                 a_probs_dict=prob_dict, dist_dict = dist_dict, calc_srq = srq1,
                 title = "3D UQ - Test")
     
# TEST
#test_1d_uq()
#test_2d_uq()
#test_3d_uq()

###############################################################################
# Tests with 2D UQ
###############################################################################
# original
dist_dict_al = {'a1': norm(loc = 2, scale = 0.13),
                'a2': norm(loc = 7, scale = 0.55),
                'a3': norm(loc = 50, scale = 0.12)}

dist_dict_ep = {'e1':  uniform(loc = 0.5, scale = 0.2),
                'e2': uniform(loc = 1, scale = 0.1),
                'e3': uniform(loc = 100, scale = 10)}

xs2, pl2, pu2 = two_dim_uq(num_inner=700, num_outer=500, 
               dist_dict_aleatory = dist_dict_al, 
               dist_dict_epistemic = dist_dict_ep, 
               calc_srq = srq1,
               title = "Original")
#
## move aleatory to epistemic
#dist_dict_al = {'a1': norm(loc = 2, scale = 0.13),
#                'a2': norm(loc = 7, scale = 0.55)}
#
#dist_dict_ep = {'e1':  uniform(loc = 0.5, scale = 0.2),
#                'e2': uniform(loc = 1, scale = 0.1),
#                'e3': uniform(loc = 100, scale = 10),
#                'a3': norm(loc = 50, scale = 0.12)}
#
#two_dim_uq(num_inner=700, num_outer=500, 
#               dist_dict_aleatory = dist_dict_al, 
#               dist_dict_epistemic = dist_dict_ep, 
#               calc_srq = srq1,
#               title = "Al to Ep")
#      
## move epistemic to aleatory
#dist_dict_al = {'a1': norm(loc = 2, scale = 0.13),
#                'a2': norm(loc = 7, scale = 0.55),
#                'a3': norm(loc = 50, scale = 0.12),
#                'e3': uniform(loc = 100, scale = 10)}
#
#dist_dict_ep = {'e1':  uniform(loc = 0.5, scale = 0.2),
#                'e2': uniform(loc = 1, scale = 0.1)}
#
#two_dim_uq(num_inner=700, num_outer=500, 
#               dist_dict_aleatory = dist_dict_al, 
#               dist_dict_epistemic = dist_dict_ep, 
#               calc_srq = srq1,
#               title = "Ep to Al")

###############################################################################
# Tests with 3D UQ
###############################################################################
prob_dict = {'a1': 0.75,
            'a2': 0.8,
            'a3': 0.9,
            'e1': 0.1,
            'e2': 0.15,
            'e3': 0.2}

dist_dict = {'a1': norm(loc = 2, scale = 0.13),
            'a2': norm(loc = 7, scale = 0.55),
            'a3': norm(loc = 50, scale = 0.12),
            'e1':  uniform(loc = 0.5, scale = 0.2),
            'e2': uniform(loc = 1, scale = 0.1),
            'e3': uniform(loc = 100, scale = 10)}

xs3, pl3, pu3 = three_dim_uq(num_inner = 1000, num_outer=100, num_toss = 10, 
             a_probs_dict=prob_dict, dist_dict = dist_dict, calc_srq = srq1,
             title = "3D UQ - Test")

fig_2d_3d = plt.figure()
p = fig_2d_3d.add_subplot(1,1,1)
p.plot(xs2, pl2, color = 'tab:blue')
p.plot(xs2, pu2, color = 'tab:blue')
p.plot(xs3, pl3, color = 'tab:red')
p.plot(xs3, pu3, color = 'tab:red')
plt.show()









