# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.12 - Bayesian lognormal-logit hurdle model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1, x2)
#

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import uniform, bernoulli

# Data
np.random.seed(33559)                                 # set seed to replicate example
nobs= 2000                                            # number of obs in model 

x1 = uniform.rvs(loc=0, scale=2.5, size=nobs)
xc = 0.6 + 1.25 * x1                                  # linear predictor, xb
y = np.random.lognormal(sigma=0.4, mean=np.exp(xc))

xb = -3.0 + 4.5 * x1                                  # construct filter
pi = 1.0/(1.0 + np.exp(-xb))
bern = [bernoulli.rvs(1-pi[i]) for i in range(nobs)]

ly = [y[i]*bern[i] for i in  range(nobs)]             # Add structural zeros


X = np.transpose(x1)
X = sm.add_constant(X)

mydata = {}                                # build data dictionary
mydata['Y'] = ly                           # response variable
mydata['N'] = nobs                         # sample size
mydata['Xb'] = X                           # predictors         
mydata['Xc'] = X
mydata['Kb'] = X.shape[1]                  # number of coefficients
mydata['Kc'] = X.shape[1] 

# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> Kb;
    int<lower=0> Kc;
    matrix[N, Kb] Xb;
    matrix[N, Kc] Xc;
    real<lower=0> Y[N];
}
parameters{
    vector[Kc] beta;
    vector[Kb] gamma;
    real<lower=0> sigmaLN;
}
model{
    vector[N] mu;
    vector[N] Pi;

    mu = exp(Xc * beta);
    for (i in 1:N) Pi[i] = inv_logit(Xb[i] * gamma);

    for (i in 1:N) {
        (Y[i] == 0) ~ bernoulli(Pi[i]);
        if (Y[i] > 0) Y[i] ~ lognormal(mu[i], sigmaLN);
    }
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=7000, chains=3,
                  warmup=4000, n_jobs=3)

# Output
print(fit)  



