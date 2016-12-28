# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.18 - Logistic model using pymc3
# 1 response (by) and 1 explanatory variables (x1)
#
# used snippets from http://people.duke.edu/~ccc14/sta-663/PyMC3.html

import numpy as np
from scipy.stats import bernoulli, uniform, binom
import pymc3 as pm
import pylab as plt
import pandas

def invlogit(x):
    """ Inverse logit function. 
    
        input: scalar
        output: scalar
    """
    return 1.0 / (1 + np.exp(-x))

# Data
np.random.seed(13979)                # set seed to replicate example
nobs= 5000                           # number of obs in model 

x1 = binom.rvs(1, 0.6, size=nobs)
x2 = uniform.rvs(size=nobs) 

beta0 = 2.0
beta1 = 0.75
beta2 = -5.0

xb = beta0 + beta1 * x1 + beta2 * x2     
exb = 1.0/(1 + np.exp(-xb))            # logit link function

by = binom.rvs(1, exb, size=nobs)

df = pandas.DataFrame({'x1': x1, 'x2': x2, 'by': by})   # re-write data

# Fit
niter = 5000                        # parameters for MCMC

with pm.Model() as model_glm:
    # define priors
    beta0 = pm.Flat('beta0')
    beta1 = pm.Flat('beta1')
    beta2 = pm.Flat('beta2')


    # define likelihood
    p = invlogit(beta0 + beta1 * x1 + beta2 * x2)
    y_obs = pm.Binomial('y_obs', n=np.ones(nobs), p=p, observed=by)

    # inference
    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(niter, step, start, progressbar=True)

# print summary to screen
pm.summary(trace)

# show graphical output
pm.traceplot(trace)
plt.show()
