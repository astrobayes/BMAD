# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 4.7 - Multivariate normal linear model in Python using STAN
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import uniform, norm

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 5000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable
x2 = uniform.rvs(size=nobs)          # second explanatory

X = np.column_stack((x1,x2))        # create response matrix
X = sm.add_constant(X)              # add intercept
beta = [2.0, 3.0, -2.5]             # create vector of parameters

xb = np.dot(X, beta)                                  # linear predictor, xb
y = np.random.normal(loc=xb, scale=1.0, size=nobs)    # create y as adjusted
                                                      # random normal variate 
# Fit
toy_data = {}                                # build data dictionary
toy_data['nobs'] = nobs                      # sample size
toy_data['x'] = X                            # explanatory variable         
toy_data['y'] = y                            # response variable
toy_data['k'] = toy_data['x'].shape[1]       # number of explanatory variables

# STAN code
stan_code = """
data {
    int<lower=1> k;  
    int<lower=0> nobs;                                 
    matrix[nobs, k] x;                     
    vector[nobs] y;                     
}
parameters {
    matrix[k,1] beta;                                             
    real<lower=0> sigma;               
}
transformed parameters{
    matrix[nobs,1] mu;
    vector[nobs] mu2;

    mu = x * beta;
    mu2 = to_vector(mu);                 # normal distribution 
                                          # does not take matrices as input
}
model {
    for (i in 1:k){                       # Diffuse normal priors for predictors
        beta[i] ~ normal(0.0, 100);
    }
    sigma ~ uniform(0, 100);            # Uniform prior for standard deviation

    y ~ normal(mu2, sigma);               # Likelihood function
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=toy_data, iter=5000, chains=3,
                  n_jobs=3, verbose=False)

# Output
nlines = 9                                   # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   


