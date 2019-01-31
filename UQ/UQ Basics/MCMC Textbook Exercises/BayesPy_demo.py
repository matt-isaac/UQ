# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:34:47 2019

bayespy demo

@author: misaa
"""

import numpy as np
from bayespy.nodes import GaussianARD, Gamma
from bayespy.inference import VB
import bayespy.plot as bpplt

"""
See http://www.bayespy.org/user_guide/quickstart.html for code

The Quick start guide indicates that there are 4 key steps to using BayesPy
for bayesian inference:
    1. Construct the model
    2. Observe some of the variables by providing the data in a proper format
    3. Run variational Bayesian inference
    4. Examine the resulting posterior distribution
    
"""

# Generate data
data = np.random.normal(5, 10, size = (10,))

"""
1. Construct the model
"""
mu = GaussianARD(0, 1e-6)
tau = Gamma(1e-6, 1e-6)
y = GaussianARD(mu, tau, plates = (10,))

"""
2. Observe some of the variables by providing data
"""
y.observe(data)

"""
Run variational Bayesian inference
"""
Q = VB(mu, tau, y)
Q.update(repeat = 20)

bpplt.pyplot.subplot(2, 1, 1)
bpplt.pdf(mu, np.linspace(-10, 20, num=100), color='k', name=r'\mu')
bpplt.pyplot.subplot(2, 1, 2)
bpplt.pdf(tau, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\tau')
bpplt.pyplot.tight_layout()
bpplt.pyplot.show()
