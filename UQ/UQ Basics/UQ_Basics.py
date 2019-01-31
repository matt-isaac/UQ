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
 
def calc_cdf(srq_list):
    """
    Calculates an emperical cdf to be plotted. Uses values in srq_list to 
    generate the cdf. 
    
    Parameters
    ----------
    srq_list : list
        list of SRQ's (numeric). 
        
    Returns
    -------
    (x,y) : tuple
        x and y values of cdf, ready to be plotted
    """
    
    ecdf = sm.distributions.ECDF(srq_list)
    x = np.linspace(min(srq_list), max(srq_list), num = 1000)
    y = ecdf(x)
        
    return((x, y))
        

def plot_cdfs(cdf_lst):
    """
    Creates plot of cdfs in cdf_list.
    Note: If using for 1D UQ (i.e. if there is only one cdf to be plotted), this
    cdf must still be put into a list. 
    
    Parameters
    ----------
    cdf_list : list of tuples
        list of tuples containing (x, y) of cdf. 
        
    """
#        perc_10_index = [i for i, j in enumerate(y) if abs(j - 0.1) <= 0.002]
#        perc_90_index = [i for i, j in enumerate(y) if abs(j - 0.9) <= 0.002]
#        
#        if isinstance(perc_10_index, list):
#            perc_10_index = perc_10_index[0]
#       
#        if isinstance(perc_90_index, list):
#            perc_90_index = perc_90_index[0]
#        
#        plt.axvline(x = x[perc_10_index], color = 'tab:gray')
#        plt.axvline(x = x[perc_90_index], color = 'tab:gray')
#        plt.axhline(y = y[perc_10_index], color = 'tab:gray')
#        plt.axhline(y = y[perc_90_index], color = 'tab:gray')
    for cdf in cdf_lst:
        plt.plot(cdf[0], cdf[1], color = 'tab:blue')
    plt.show()
    
    
def one_dim_uq(num_trials, dist_dict, calc_srq):     
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
        
    Returns
    -------
    cdf : numeric 1D array
        a array containing the values of the cdf (to be plotted)
    """    
    srqs = []  
    for i in range(num_trials):
        inputs = sample_inputs(dist_dict)
        result = calc_srq(**inputs)
        srqs.append(result)
    
    cdf = calc_cdf(srqs)
    plot_cdfs([cdf])
    return(cdf)
    
def two_dim_uq(num_inner, num_outer, dist_dict_aleatory, dist_dict_epistemic, calc_srq):
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
        
    Returns
    -------
    cdf_list : list of numeric lists
        A list (ensemble) of calculated cdfs
    """    
    cdfs = []
    for i in range(0, num_outer, 1):
        ep_inputs = sample_inputs(dist_dict = dist_dict_epistemic)
        srqs = []
        for j in range(0, num_inner, 1):
            al_inputs = sample_inputs(dist_dict = dist_dict_aleatory)
            inputs = {**al_inputs, **ep_inputs}
            result = calc_srq(**inputs)
            srqs.append(result)
        cdf = calc_cdf(srq_list = srqs)
        cdfs.append(cdf)
    plot_cdfs(cdfs)

## Unit Tests
def test_1d_uq():
    dist_dict_1d = {'a1': norm(loc = 0.5, scale = 0.01),
                'a2': norm(loc = 5, scale = 0.1),
                'a3': norm(loc = 2, scale = 0.01),
                'e1': norm(loc = 2.5, scale = 0.01),
                'e2': norm(loc = 0.5, scale = 0.01),
                'e3': norm(loc = 3, scale = 0.02)}
    
    one_dim_uq(num_trials = 10000, dist_dict = dist_dict_1d, calc_srq = srq1)
     
def test_2d_uq():
    dist_dict_al = {'a1': norm(loc = 0.5, scale = 0.01),
                'a2': norm(loc = 5, scale = 0.1),
                'a3': norm(loc = 2, scale = 0.01)}

    dist_dict_ep = {'e1': norm(loc = 2.5, scale = 0.01),
                'e2': norm(loc = 0.5, scale = 0.01),
                'e3': norm(loc = 3, scale = 0.02)}

    two_dim_uq(num_inner=1000, num_outer=500, 
               dist_dict_aleatory = dist_dict_al, dist_dict_epistemic = dist_dict_ep, calc_srq = srq1)
    
# TEST
test_1d_uq()
test_2d_uq()