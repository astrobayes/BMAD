# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.29 - Adaptation of binomial model data section of Code 5.27 
#             allowing it to handle explicit three-parameter data

# 1 response (y) and 3 explanatory variables (x1, x2, x3)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform, poisson, binom

# Data
np.random.seed(33559)                # set seed to replicate example

y = [6,11,9,13,17,21,8,10,15,19,7,12]
m = [45,54,39,47,29,44,36,57,62,55,66,48]
x1 = [1,1,1,1,1,1,0,0,0,0,0,0]
x2 = [1,1,0,0,1,1,0,0,1,1,0,0]
x3 = [1,0,1,0,1,0,1,0,1,0,1,0]

mydata = {}
mydata['K'] = 4
mydata['X'] = sm.add_constant(np.column_stack((x1,x2,x3)))
mydata['N'] = len(y)
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
nlines = 9

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)
