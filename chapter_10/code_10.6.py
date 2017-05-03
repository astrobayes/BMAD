# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
#Code 10.6 Multivariate Gaussian mixed model in Python, using Stan, 
#          for accessing the relationship between luminosity, period, 
#          and color in early-type contact binaries.
#
# Statistical Model: Multivariate Gaussian regression 
#                    
#
# Astronomy case: Relation between period, color and luminosity 
#                 for early type contact binaries
#                 taken from Pawlak, 2016, MNRAS, 457 (4), p.4323-4329
#
# 1 response (obsy - luminosity) 
# 2 explanatory variables (x1 - log period, x2 - color (V-I))
#
# Data from: http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1602.01467

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p3/PLC.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['x1'] = np.array(data_frame['logP'])
data['x2'] = np.array(data_frame['V_I'])
data['y'] = np.array(data_frame['M_V'])
data['nobs'] = len(data['x1'])
data['type'] = np.array([1 if item == data_frame['type'][0] else 0 
                         for item in data_frame['type']])
data['M'] = 3
data['K'] = data['M'] - 1

# Fit
stan_code="""
data{
    int<lower=0> nobs;                # number of data points
    int<lower=1> M;                   # number of linear predicor coefficients
    int<lower=1> K;                   # number of  distinct populations
    vector[nobs] x1;                  # obs log period
    vector[nobs] x2;                  # obs color V-I
    vector[nobs] y;                   # obs luminosity
    int type[nobs];                   # system type (near/genuine contact)
}
parameters{
    matrix[M,K] beta;                 # linear predictor coefficients
    real<lower=0> sigma[K];           # scatter around linear predictor
    real mu0;
    real sigma0;
}
model{
    vector[nobs] mu;                  # linear predictor

    for (i in 1:nobs) {
        if (type[i] == type[1])
            mu[i] = beta[1,2] + beta[2,2] * x1[i] + beta[3,2] * x2[i];
        else mu[i] = beta[1,1] + beta[2,1] * x1[i] + beta[3,1] * x2[i];
    }

    # priors and likelihood
    mu0 ~ normal(0, 100);
    sigma0 ~ gamma(0.001, 0.001);

    for (i in 1:K) {
        sigma[i] ~ gamma(0.001, 0.001);
        for (j in 1:M) beta[j,i] ~ normal(mu0,sigma0);
    }

    for (i in 1:nobs){
        if (type[i] == type[1]) y[i] ~ normal(mu[i], sigma[2]);
        else y[i] ~ normal(mu[i], sigma[1]);
    }
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=5000, chains=3,
                  warmup=2500, thin=1, n_jobs=3)

# Output
nlines = 13                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item) 
