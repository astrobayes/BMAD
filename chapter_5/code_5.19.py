# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.19 - Logistic model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import pystan
import statsmodels.api as sm

from scipy.stats import uniform, bernoulli

# Data
np.random.seed(13979)                # set seed to replicate example
nobs= 5000                           # number of obs in model 

x1 = bernoulli.rvs(0.6, size=nobs)
x2 = uniform.rvs(size=nobs) 

beta0 = 2.0
beta1 = 0.75
beta2 = -5.0

xb = beta0 + beta1 * x1 + beta2 * x2     
exb = 1.0/(1 + np.exp(-xb))            # logit link function

by = bernoulli.rvs(exb, size=nobs)

mydata = {}
mydata['Y'] = by
mydata['N'] = nobs
mydata['X'] = sm.add_constant(np.column_stack((x1, x2)))
mydata['K'] = 3
mydata['logN'] = np.log(nobs)


# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    int Y[N];
    matrix[N,K] X;
    real logN;
}
parameters{
    vector[K] beta;
}
transformed parameters{
    vector[N] eta;

    eta = X * beta;
}
model{

    Y ~ bernoulli_logit(eta);
}
generated quantities{
    real LLi[N];
    real AIC;
    real BIC;
    real LogL;
    vector[N] etanew;
    real<lower=0, upper=1.0> pnew[N];

    etanew = X * beta;

    for (i in 1:N){ 
        pnew[i] = inv_logit(etanew[i]);
        LLi[i] = bernoulli_lpmf(1|pnew[i]);
    }

    LogL = sum(LLi);
    AIC = -2 * LogL + 2 * K; 
    BIC = -2 * LogL + logN * K; 
}
"""

fit = pystan.stan(model_code=stan_code, data=mydata, iter=10000, chains=3,
                  warmup=5000, n_jobs=1)

# Output
lines = list(range(8)) + [2 * nobs + 8, 2 * nobs + 9, 2 * nobs + 10]
output = str(fit).split('\n')

for i in lines:
    print(output[i])   

