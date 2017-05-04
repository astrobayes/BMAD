# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 10.25 Negative binomial model (AR1) in Python using Stan, 
#            for assessing the evolution of the number of sunspots 
#            through the years.
#
# Statistical Model: Time series in Python using Stan
#
# Astronomy case: Evolution of the number of sunspots with time
#                 from Lunn et al., 2012, 
#                 The BUGS Book: a Practical Introduction to
#                                Bayesian Analysis, CRC
#
# 1 response (Y - number of sunspots) 
# 1 explanatory variable (x - year)
#
# Data from: http://www.sidc.be/silso/DATA/EISN/EISN_current.csv

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = "https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p10/sunspot.csv"

# read data
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['Y'] = [int(round(item)) for item in data_frame['nspots']]
data['nobs'] = len(data['Y'])
data['K'] = 2

############### Fit
# Stan  model
stan_code="""
data{
    int<lower=0> nobs;                # number of data points
    int<lower=0> K;                   # number of coefficients
    int Y[nobs];                      # nuber of sunspots
}
parameters{
    vector[K] phi;                    # linear predictor coefficients
    real<lower=0> theta;              # noise parameter
}
model{
    vector[nobs] mu;
    
    mu[1] = Y[1];                     # set initial value

    for (t in 2:nobs) mu[t] = exp(phi[1] + phi[2] * Y[t - 1]);

    # priors and likelihood
    theta ~ gamma(0.001, 0.001);
    for (i in 1:K) phi[i] ~ normal(0, 100);

    Y ~ neg_binomial_2(mu, theta);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=7500, chains=3,
                  warmup=5000, thin=1, n_jobs=3)

############### Output
print(fit)
