# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.24 - Probit model in Python using Stan

import numpy as np
import pystan
import statsmodels.api as sm
from scipy.stats import uniform, norm, bernoulli

# Data
np.random.seed(1944)                        # set seed to replicate example
nobs = 2000                                 # number of obs in model
x1 = uniform.rvs(size=nobs)
x2 = 2 * uniform.rvs(size=nobs)

beta0 = 2.0
beta1 = 0.75
beta2 = -1.25

xb = beta0 + beta1 * x1 + beta2 * x2
exb = 1 - norm.sf(xb)                       # inverse probit link
py = bernoulli.rvs(exb)

# Data transformation for illustrative purpouses
K = 3                                       # number of coefficients
X = np.column_stack((x1, x2))
X = sm.add_constant(X)


# Fit
probit_data = {}
probit_data['N'] = nobs
probit_data['K'] = K
probit_data['X'] = X
probit_data['Y'] = py
probit_data['logN'] = np.log(nobs)

probit_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    int Y[N];
    real logN;
}
parameters{
   vector[K] beta;
}
transformed parameters{
    vector[N] xb;
    
    xb = X * beta;
}
model{
   
    for (i in 1:N) Y[i] ~ bernoulli(Phi(xb[i]));      # likelihood
}
generated quantities{
    real LLi[N];
    real AIC;
    real BIC;
    real LogL;
    vector[N] xb2;
    real p[N];

    xb2 = X * beta;
    
    for (i in 1:N){
        p[i] = Phi(xb2[i]);
        LLi[i] = Y[i] * log(p[i]) + (1-Y[i]) * log(1 - p[i]);
    }
   
    LogL = sum(LLi);
    AIC = -2 * LogL + 2 * K;
    BIC = -2 * LogL + logN * K;
}
"""

fit = pystan.stan(model_code=probit_code, data=probit_data, iter=5000, chains=3,
warmup=3000, n_jobs=3)

# Output
lines = list(range(8)) + [2 * nobs + 8, 2 * nobs + 9, 2 * nobs + 10]

output = str(fit).split('\n')

for i in lines:
    print(output[i])
