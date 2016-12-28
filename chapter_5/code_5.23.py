# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.23 - Probit model using Python

import numpy as np
from scipy.stats import norm, uniform, bernoulli
import pymc3 as pm
import pylab as plt
import pandas
import theano.tensor as tsr

def probit_phi(x):
    """Probit transformation."""
    mu = 0
    sd = 1
    return 0.5 * (1 + tsr.erf((x - mu) / (sd * tsr.sqrt(2))))

# Data
np.random.seed(135)                                       # set seed to replicate example
nobs = 5000                                               # number of obs in model

x1 = uniform.rvs(size=nobs)
x2 = 2 * uniform.rvs(size=nobs)

beta0 = 2.0                                                # coefficients for linear predictor
beta1 = 0.75
beta2 = -1.25

xb = beta0 + beta1 * x1 + beta2 * x2                       # linear predictor
exb = 1 - norm.sf(xb)                                      # inverse probit link
py = bernoulli.rvs(exb)
df = pandas.DataFrame({'x1': x1, 'x2':x2, 'by': py})       # re-write data

# Fit
niter = 10000                                              # parameters for MCMC

with pm.Model() as model_glm:
    # define priors
    beta0 = pm.Flat('beta0')
    beta1 = pm.Flat('beta1')
    beta2 = pm.Flat('beta2')

    # define likelihood
    theta_p = beta0 + beta1*x1 + beta2 * x2
    theta = probit_phi(theta_p)
    y_obs = pm.Bernoulli('y_obs', p=theta, observed=py)

    # inference
    start = pm.find_MAP()                                  # find starting value by optimization
    step = pm.NUTS()
    trace = pm.sample(niter, step, start, random_seed=135, progressbar=True)

# Print summary to screen
pm.summary(trace)

# Show graphical output
pm.traceplot(trace)
plt.show()
