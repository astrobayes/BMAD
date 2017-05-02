# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 10.4 - Gaussian linear mixed model, in Python using Stan, for modeling
#             the relationship between type Ia supernovae host galaxy mass 
#             and Hubble residuals.
#
# Statistical Model: Gaussian mixed model considering errors in variables
#                    in Python using Stan
#
# Astronomy case:  Relation between mass type Ia SNe host galaxy mass 
#                  and Hubble Residuals
#                  from Wolf et al., 2016, ApJ 821 (2), id. 115
#
# 1 response (obsy - Hubble Residuals)  
# 1 explanatory variable (obsx - host galaxy mass)


import numpy as np
import pandas as pd
import pystan 

# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p2/HR.csv'
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['obsx'] = np.array(data_frame['LogMass'])
data['errx'] = np.array(data_frame['e_LogMass'])
data['obsy'] = np.array(data_frame['HR'])
data['erry'] = np.array(data_frame['e_HR'])
data['type'] = np.array([1 if item == 'P' else 0 
                         for item in data_frame['Type']])
data['N'] = len(data['obsx'])
data['K'] = 2                        # number of distinct populations
data['L'] = 2                        # number of coefficients

# Fit
stan_code="""
data{
    int<lower=0> N;                   # number of data points
    int<lower=0> K;                   # number of distinct populations
    int<lower=0> L;                   # number of coefficients
    vector[N] obsx;                   # obs host galaxy mass
    vector<lower=0>[N] errx;          # errors in host mass measurements
    vector[N] obsy;                   # obs Hubble Residual
    vector<lower=0>[N] erry;          # errors in Hubble Residual measurements
    vector[N] type;                   # flag for spec/photo sample
}
parameters{
    matrix[K,L] beta;                 # linear predictor coefficients
    real<lower=0> sigma;              # scatter around true black hole mass
    vector[N] x;                      # true host galaxy mass
    vector[N] y;                      # true Hubble Residuals
    real<lower=0, upper=5> sig0;      # scatter for shared hyperprior on beta
    real mu0;                         # mean for shared hyperprior on beta
}
transformed parameters{
    vector[N] mu;                     # linear predictor

    for (i in 1:N) {
        if (type[i] == type[1]) mu[i] = beta[1,1] + beta[2,1] * x[i];
        else mu[i] = beta[1,2] + beta[2,2] * x[i];
    }
}
model{
    # shared hyperprior
    mu0 ~ normal(0, 1);
    sig0 ~ normal(0, 5);

    for (i in 1:K){
        for (j in 1:L) beta[i,j] ~ normal(mu0, sig0);
    } 

    # priors and likelihood
    obsx ~ normal(x, errx);
    x ~ normal(0, 10);
    y ~ normal(mu, sigma);
    sigma ~ gamma(0.5,0.5);

    obsy ~ normal(y, erry);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=40000, chains=3,
                  warmup=15000, thin=1, n_jobs=3)

# Output
nlines = 10                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item) 
