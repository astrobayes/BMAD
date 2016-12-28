# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.5 - Lognormal model in Python using Stan
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
from scipy.stats import uniform, lognorm
import pystan

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 5000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable

beta0 = 2.0                          # intercept
beta1 = 3.0                          # linear predictor
sigma = 1.0                          # dispersion
xb = beta0 + beta1 * x1              # linear predictor, xb
exb = np.exp(xb)

y = lognorm.rvs(sigma, scale=exb, size=nobs)    # create y as adjusted
                                                # random normal variate  
# Fit
mydata = {}
mydata['N'] = nobs
mydata['x1'] = x1
mydata['y'] = y


stan_lognormal = """
data{
    int<lower=0> N;
    vector[N] x1;
    vector[N] y;
}
parameters{ 
    real beta0;
    real beta1;
    real<lower=0> sigma;
}
transformed parameters{
    vector[N] mu;
    
    for (i in 1:N) mu[i] = beta0 + beta1 * x1[i];
}
model{
    y ~ lognormal(mu, sigma);
}
"""

# fit
fit = pystan.stan(model_code=stan_lognormal, data=mydata, iter=5000, chains=3,
                  verbose=False, n_jobs=3)


############### Output
nlines = 8                          # number of lines in output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

