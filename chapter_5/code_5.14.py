# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.14 - Beta model in Python using Stan
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform
from scipy.stats import beta as beta_dist

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 2000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable

beta0 = 0.3
beta1 = 1.5
theta = 15

xb = beta0 + beta1 * x1
exb = np.exp(-xb)
p = exb / (1 + exb)

y = beta_dist.rvs(theta * (1 - p), theta * p)           # create y as adjusted

# Fit
mydata = {}                                
mydata['N'] = nobs                         # sample size
mydata['x1'] = x1                          # predictors         
mydata['y'] = y                            # response variable
  
stan_code = """
data{
    int<lower=0> N;
    vector[N] x1;
    vector<lower=0, upper=1>[N] y;
}
parameters{
    real beta0;
    real beta1;
    real<lower=0> theta;
}
model{
    vector[N] eta;
    vector[N] p;
    vector[N] shape1;
    vector[N] shape2;

    for (i in 1:N){
        eta[i] = beta0 + beta1 * x1[i];
        p[i] = inv_logit(eta[i]);
        shape1[i] = theta * p[i];
        shape2[i] = theta * (1 - p[i]);
    }

    y ~ beta(shape1, shape2);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
                  warmup=2500, n_jobs=3)

# Output
print(fit)  

