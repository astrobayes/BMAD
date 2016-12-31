# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Chapter 8 - Astronomical Applications 
#
# Statistical Model: Hurdle model in Python using Stan
#
# Astronomy case: Relationship between stellar and dark matter halo mass
#                 inspired on 
#                 de Souza et al., 2015, Astronomy & Computing 12, p. 21-32
#
# 1 response variable (Y - stellar mass)
# 1 explanatory variable (X - dark matter halo mass)
#
# Data from: Biffi & Maio, 2013, MNRAS 436 (2), p.1621

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

############### Data
path_to_data = ('../data/Section_10p9/MstarZSFR.csv')

# read data
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
y = np.array([np.arcsinh(10**10*item) for item in data_frame['Mstar']])
x = np.array([np.log10(item) for item in data_frame['Mdm']])

data = {}
data['Y'] = y
data['Xc'] = sm.add_constant(x.transpose())
data['Xb'] = sm.add_constant(x.transpose())
data['Kc'] = data['Xc'].shape[1]
data['Kb'] = data['Xb'].shape[1]
data['N'] = data['Xc'].shape[0]

############### Fit
# Stan  model
stan_code="""
data{
    int<lower=0> N;                # number of data points
    int<lower=0> Kc;               # number of coefficients
    int<lower=0> Kb;
    matrix[N,Kb] Xb;               # dark matter halo mass
    matrix[N,Kc] Xc;
    real<lower=0> Y[N];            # stellar mass
}
parameters{
    vector[Kc] beta;
    vector[Kb] gamma;
    real<lower=0> sigmaLN;
}
model{
    vector[N] mu;
    vector[N] Pi;

    mu = Xc * beta;
    for (i in 1:N) Pi[i] = inv_logit(Xb[i] * gamma);
  
    # priors and likelihood
    for (i in 1:Kc) beta[i] ~ normal(0, 100);
    for (i in 1:Kb) gamma[i] ~ normal(0, 100);
    sigmaLN ~ gamma(0.001, 0.001);

    for (i in 1:N) {
        (Y[i] == 0) ~ bernoulli(Pi[i]);
        if (Y[i] > 0) Y[i] ~ lognormal(mu[i], sigmaLN);
    }
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=15000, chains=3,
                  warmup=5000, thin=1, n_jobs=3)

# Output
print(fit)
