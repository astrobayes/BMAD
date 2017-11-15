# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.13 Bayesian random intercept Poisson model in Python


import numpy as np
import pymc3 as pm

from scipy.stats import norm, uniform, poisson

# Data
np.random.seed(1656)                  # set seed to replicate example
N = 2000                              # number of obs in model 
NGroups = 10

x1 = uniform.rvs(size=N)
x2 = uniform.rvs(size=N)

Groups = np.array([200 * [i] for i in range(NGroups)]).flatten()
a = norm.rvs(loc=0, scale=0.5, size=NGroups)
eta = 1 + 0.2 * x1 - 0.75 * x2 + a[list(Groups)]
mu = np.exp(eta)

y = poisson.rvs(mu, size=N)


with pm.Model() as model: 
    # Define priors
    sigma_a = pm.Uniform('sigma_a', 0, 100)
    beta1 = pm.Normal('beta1', 0, sd=100)
    beta2 = pm.Normal('beta2', 0, sd=100)
    beta3 = pm.Normal('beta3', 0, sd=100)
    
    # priors for random intercept (RI) parameters
    a_param = pm.Normal('a_param',
                         np.repeat(0, NGroups),                   # mean
                         sd=np.repeat(sigma_a, NGroups),          # standard deviation
                         shape=NGroups)                           # number of RI parameters

    eta = beta1 + beta2*x1 + beta3*x2 + a_param[Groups]
    
    # Define likelihood
    y = pm.Poisson('y', mu=np.exp(eta), observed=y)
    
    # Fit
    start = pm.find_MAP()                        # Find starting value by optimization
    step = pm.NUTS(scaling=start)                  # Initiate sampling 
    trace = pm.sample(20000, step, start=start, progressbar=False)     

# Print summary to screen
pm.summary(trace)

