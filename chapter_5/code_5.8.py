# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.8 - Log-gamma model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import pystan

from scipy.stats import uniform, gamma

# Data
np.random.seed(33559)                # set seed to replicate example
nobs= 3000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable
x2 = uniform.rvs(size=nobs)          # second explanatory

beta0 = 1.0                          # intercept
beta1 = 0.66                         # first linear coefficient
beta2 = -1.25                        # second linear coefficient

eta = beta0 + beta1 * x1 + beta2 * x2      # linear predictor, xb
mu = np.exp(eta)
y = gamma.rvs(mu)                          # create y as adjusted

# Fit
mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['x1'] = x1                          # explanatory variable         
mydata['x2'] = x2
mydata['y'] = y                            # response variable

# STAN code
stan_gamma = """
data{
    int<lower=0> N;
    vector[N] x1;
    vector[N] x2;
    vector[N] y;
}
parameters{
    real beta0;
    real beta1;
    real beta2;
    real<lower=0> r;
}
transformed parameters{
    vector[N] eta;
    vector[N] mu;
    vector[N] lambda;

    for (i in 1:N){
        eta[i] = beta0 + beta1 * x1[i] + beta2 * x2[i];
        mu[i] = exp(eta[i]);
        lambda[i] = r/mu[i];
    }
}
model{
    r ~ gamma(0.01, 0.01);
    for (i in 1:N) y[i] ~ gamma(r, lambda[i]);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_gamma, data=mydata, iter=7000, chains=3,
                  warmup=6000, n_jobs=3)

# Output
nlines = 9                                   # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

