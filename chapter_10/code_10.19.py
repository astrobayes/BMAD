# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 10.19 Bernoulli logit model, in Python using Stan, for assessing 
#            the relationship between Seyfert AGN activity and 
#            galactocentric distance
#
# Statistical Model: Bernoulli mixed model in Python using Stan
#
# Astronomy case: Relationship between Seyfert activity and 
#                 cluster centric distance - taken from 
#                 de Souza et al., 2016, MNRAS in  press, 
#                 arXiv:astro-ph/1603.06256
#
# 1 response variable (Y - galaxy class Seyfert - 1/AGN - 0)
# 2 explanatory variable (x1 - M200, x2 - cluster-centric distance)
#
# Data from: Trevisan, Mamon & Khosroshahi, 2017, MNRAS, 464, p.4593-4610
#            https://github.com/COINtoolbox/LOGIT_AGNs/tree/master/data

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p8/Seyfert.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))

x1 = data_frame['logM200']
x2 = data_frame['r_r200']

data = {}
data['Y'] = data_frame['bpt']
data['X'] = sm.add_constant(np.column_stack((x1,x2)))
data['K'] = data['X'].shape[1]
data['N'] = data['X'].shape[0]
data['gal'] = [0 if item == data_frame['zoo'][0] else 1 
                 for item in data_frame['zoo']]
data['P'] = 2


# Fit
stan_code="""
data{
    int<lower=0> N;                # number of data points
    int<lower=0> K;                # number of coefficients
    int<lower=0> P;                # number of populations
    matrix[N,K] X;                 # [logM200, galactocentric distance]
    int<lower=0, upper=1> Y[N];    # Seyfert 1/SF 0
    int<lower=0, upper=1> gal[N];  # elliptical 0/spiral 1
}
parameters{
    matrix[K,P] beta;
    real<lower=0> sigma;
    real mu;
}
model{
    vector[N] pi;

    for (i in 1:N) {
        if (gal[i] == gal[1]) pi[i] = dot_product(col(beta,1),X[i]);
        else pi[i] = dot_product(col(beta,2), X[i]);
    }

    # shared hyperpriors
    sigma ~ gamma(0.001, 0.001);
    mu ~ normal(0, 100);

    # priors and likelihood
    for (i in 1:K) {
        for (j in 1:P) beta[i,j] ~ normal(mu, sigma);
    }

    Y ~ bernoulli_logit(pi);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=60000, chains=3,
                  warmup=30000, thin=10, n_jobs=3)

# Output
print(fit)
