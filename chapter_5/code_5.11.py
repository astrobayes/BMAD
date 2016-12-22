# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.11 - Inverse Gaussian model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform, invgauss

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 1000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable

beta0 = 1.0
beta1 = 0.5
l1 = 20

xb = beta0 + beta1 * x1                       # linear predictor, xb              
exb = np.exp(xb)

y = invgauss.rvs(exb/l1, scale=l1)            # create response variable
                                             
# Fit
stan_data = {}                                # build data dictionary
stan_data['Y'] = y                            # response variable
stan_data['x1'] = x1                          # explanatory variable
stan_data['N'] = nobs                         # sample size

# Stan code
stan_code = """
data{
    int<lower=0> N;
    vector[N] Y;
    vector[N] x1;
}
parameters{
    real beta0;
    real beta1;
    real<lower=0> lambda;
}
transformed parameters{
    vector[N] exb;
    vector[N] xb;

    for (i in 1:N) xb[i] = beta0 + beta1 * x1[i];
    for (i in 1:N) exb[i] = exp(xb[i]);
}
model{
    real l1;
    real l2;   
    vector[N] loglike;

    lambda ~ uniform(0.0001, 100);

    for (i in 1:N){
        l1 = 0.5 * (log(lambda) - log(2 * pi() * pow(Y[i], 3)));
        l2 = -lambda*pow(Y[i] - exb[i], 2)/(2 * pow(exb[i], 2) * Y[i]);
        loglike[i] = l1 + l2;
    }

    target += loglike;
}
"""

fit = pystan.stan(model_code=stan_code, data=stan_data, iter=5000, chains=3,
                  warmup=2500, n_jobs=3)

# Output
nlines = 8                                   # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

