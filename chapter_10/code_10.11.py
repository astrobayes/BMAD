# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Chapter 8 - Astronomical Applications 
#
# Statistical Model: Beta model in Python using Stan
#
# Astronomy case: Relation between atomic gas fraction
#                 and  stellar mass
#                 taken from Bradford et al., 2015, ApJ 809, id. 146
#
# 1 response (Y - atomic gass fraction) 
# 1 explanatory variable (x - log stellar mass)
#
# Original data from: http://www.astro.yale.edu/jdbradford/research.html

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = '../data/Section_10p5/f_gas.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))

# built atomic gas fraction
y = np.array([data_frame['M_HI'][i]/
              (data_frame['M_HI'][i] + data_frame['M_STAR'][i])
              for i in range(data_frame['M_STAR'].shape[0])])

x = np.array([np.log(item) for item in data_frame['M_STAR']])

# prepare data for Stan
data = {}
data['Y'] = y
data['X'] = sm.add_constant((x.transpose()))
data['nobs'] = data['X'].shape[0]
data['K'] = data['X'].shape[1]

############### Fit
# Stan  model
stan_code="""
data{
    int<lower=0> nobs;                # number of data points
    int<lower=0> K;                   # number of coefficients
    matrix[nobs, K] X;                # stellar mass
    real<lower=0, upper=1> Y[nobs];   # atomic gas fraction
}
parameters{
    vector[K] beta;                   # linear predictor coefficients
    real<lower=0> theta;
}
model{
    vector[nobs] pi;
    real a[nobs];
    real b[nobs];
    
    for (i in 1:nobs){
       pi[i] = inv_logit(X[i] * beta);
       a[i]  = theta * pi[i];
       b[i]  = theta * (1 - pi[i]);
    }

    # priors and likelihood
    for (i in 1:K) beta[i] ~ normal(0, 100);
    theta ~ gamma(0.01, 0.01);

    Y ~ beta(a, b);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=7500, chains=3,
                  warmup=5000, thin=1, n_jobs=3)

############### Output
print(fit)
