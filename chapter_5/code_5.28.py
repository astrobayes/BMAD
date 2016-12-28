# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.28 - Binomial model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform, poisson, binom

# Data
np.random.seed(33559)                # set seed to replicate example
nobs= 2000                           # number of obs in model

m = 1 + poisson.rvs(5, size=nobs)
x1 = uniform.rvs(size=nobs)          # random uniform variable
x2 = uniform.rvs(size=nobs)

beta0 = -2.0
beta1 = -1.5
beta2 = 3.0

xb = beta0 + beta1 * x1 + beta2 * x2
exb = np.exp(xb)
p = exb / (1 + exb)
y = binom.rvs(m, p)                   # create y as adjusted

mydata = {}
mydata['K'] = 3
mydata['X'] = sm.add_constant(np.column_stack((x1,x2)))
mydata['N'] = nobs
mydata['Y'] = y
mydata['m'] = m

# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    int Y[N];
    int m[N];
}
parameters{
    vector[K] beta;
}
transformed parameters{
    vector[N] eta;
    vector[N] p;
    eta = X * beta;
    for (i in 1:N) p[i] = inv_logit(eta[i]);
}
model{
    Y ~ binomial(m, p);
}
"""

fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
warmup=3000, n_jobs=3)

# Output
nlines = 8

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)
