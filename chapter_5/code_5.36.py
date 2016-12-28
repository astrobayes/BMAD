# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.36 - Beta-binomial model with synthetic data in Python using Stan
# 1 response (y) and 1 explanatory variables (x1)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform, poisson, binom
from scipy.stats import beta as beta_dist


# Data
np.random.seed(33559)                # set seed to replicate example
nobs= 4000                           # number of obs in model 
m = 1 + poisson.rvs(5, size=nobs)
x1 = uniform.rvs(size=nobs)          # random uniform variable
  

beta0 = -2.0
beta1 = -1.5

eta = beta0 + beta1 * x1
sigma = 20

p = np.exp(eta) / (1 + np.exp(eta))
shape1 = sigma * p
shape2 = sigma * (1-p)

 # binomial distribution with p ~ beta
y = binom.rvs(m, beta_dist.rvs(shape1, shape2))          

mydata = {}
mydata['K'] = 2
mydata['X'] = sm.add_constant(np.transpose(x1))
mydata['N'] = nobs
mydata['Y'] = y
mydata['m'] = m

# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    int<lower=0> Y[N];
    int<lower=1> m[N];
}
parameters{
    vector[K] beta;
    real<lower=0> sigma;
}
transformed parameters{
    vector[N] eta;
    vector[N] pi;
    vector[N] shape1;
    vector[N] shape2;
 
    eta = X * beta;
    for (i in 1:N){ 
        pi[i] = inv_logit(eta[i]); 
        shape1[i] = sigma * pi[i];
        shape2[i] = sigma * (1 - pi[i]);
    }
}
model{    

    Y ~ beta_binomial(m, shape1, shape2);  
}
"""

fit = pystan.stan(model_code=stan_code, data=mydata, iter=7000, chains=3,
                  warmup=3500, n_jobs=3)

# Output
nlines = 8

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)
