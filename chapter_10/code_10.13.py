# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Chapter 8 - Astronomical Applications 
#
# Statistical Model: Bernoulli model in Python using Stan
#
# Astronomy case: Relation between fraction of red spiral galaxies
#                 and  galaxy bulge size
#                 taken from Masters et al., 2010, MNRAS,  405 (2), 783-799
#
# 1 response (Y - galaxy type: red/blue spirals) 
# 1 explanatory variable (x - bulge size)
#
# Data from: http://data.galaxyzoo.org/data/redspirals/BlueSpiralsA2.txt
#            http://data.galaxyzoo.org/data/redspirals/RedSpiralsA1.txt

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = '../data/Section_10p6/Red_spirals.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))
x = np.array(data_frame['fracdeV'])

# prepare data for Stan
data = {}
data['X'] = sm.add_constant((x.transpose()))
data['Y'] = np.array(data_frame['type'])
data['nobs'] = data['X'].shape[0]
data['K'] = data['X'].shape[1]

# Fit
# Stan  model
stan_code="""
data{
    int<lower=0> nobs;                # number of data points
    int<lower=0> K;                   # number of coefficients
    matrix[nobs, K] X;                # bulge size
    int Y[nobs];                      # galaxy type: 1 - red, 0 - blue
}
parameters{
    vector[K] beta;                   # linear predictor coefficients
}
model{
    # priors and likelihood
    for (i in 1:K) beta[i] ~ normal(0, 100);

    Y ~ bernoulli_logit(X * beta);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=6000, chains=3,
                  warmup=3000, thin=1, n_jobs=3)

############### Output
print(fit)
