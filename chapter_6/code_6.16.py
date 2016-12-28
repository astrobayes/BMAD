# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 6.16 - Negative binomial model in Python using pymc3
# 1 response (nby) and 2 explanatory variables (x1, x2)
#
# pymc3 parametrization taken from http://bebi103.caltech.edu/2015/tutorials/r7_pymc3.html

import numpy as np
import pandas
import pylab as plt
import pymc3 as pm
import pystan 
from scipy.stats import uniform, binom, nbinom
import statsmodels.api as sm

# Data
np.random.seed(141)                 # set seed to replicate example
nobs= 2500                          # number of obs in model 

x1 = binom.rvs(1, 0.6, size=nobs)   # categorical explanatory variable
x2 = uniform.rvs(size=nobs)         # real explanatory variable

theta = 0.303
xb = 1 + 2 * x1 - 1.5 * x2          # linear predictor

exb = np.exp(xb)
nby = nbinom.rvs(exb, theta)

df = pandas.DataFrame({'x1': x1, 'x2':x2,'nby': nby})   # re-write data


# Fit
niter = 10000                        # parameters for MCMC

with pm.Model() as model_glm:
    # define priors
    beta0 = pm.Flat('beta0')
    beta1 = pm.Flat('beta1')
    beta2 = pm.Flat('beta2')

    # define likelihood
    linp = beta0 + beta1 * x1 + beta2 * x2
    mu = np.exp(linp)
    mu2 = mu * (1 - theta)/theta      # compensate for difference between
                                      # parametrizations from pymc3 and scipy       
    y_obs = pm.NegativeBinomial('y_obs', mu2, theta, observed=nby)

    # inference
    start = pm.find_MAP()             # Find starting value by optimization
    step = pm.NUTS()
    trace = pm.sample(niter, step, start, progressbar=True)

# print summary to screen
pm.summary(trace)

# show graphical output
pm.traceplot(trace)
plt.show()
