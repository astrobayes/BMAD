# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Code 10.9 Lognormal model in Python using Stan to describe the 
#           initial mass function (IMF)
#
# Statistical Model: Lognormal distribution in Python using Stan
#
# Astronomy case: Modelling the Initial Mass Function behavior
#                 taken from Zaninetti, L., ApJ  765 (2), id. 128
#
# 1 variable (X - stellar mass)
#
#
# Data from: http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/MNRAS/392/1034/table1
#            column M2Myr

import numpy as np
import pandas as pd
import pylab as plt
import pystan 
import statsmodels.api as sm

# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p4/NGC6611.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['X'] = data_frame['Mass']
data['nobs'] = data['X'].shape[0]

# Fit
# Stan  model
stan_code="""
data{
    int<lower=0> nobs;                # number of data points
    vector[nobs] X;                   # stellar mass
}
parameters{
    real mu;                          # mean 
    real<lower=0> sigma;              # scatter
}
model{
    # priors and likelihood
    sigma ~ normal(0, 100);
    mu ~ normal(0, 100);

    X ~ lognormal(mu, sigma);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=5000, chains=3,
                  warmup=2500, thin=1, n_jobs=3)

# Output
print(fit)

# plot chains and posteriors
fit.traceplot()
plt.show()
