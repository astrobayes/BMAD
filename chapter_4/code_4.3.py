# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 4.3 - Normal linear model in Python using STAN
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
import statsmodels.api as sm
import pystan
from scipy.stats import uniform

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 5000                           # number of obs in model 
x1 = uniform.rvs(size=nobs)          # random uniform variable

x1.transpose()                   # create response matrix
X = sm.add_constant(x1)          # add intercept
beta = [2.0, 3.0]                # create vector of parameters

xb = np.dot(X, beta)                                  # linear predictor, xb
y = np.random.normal(loc=xb, scale=1.0, size=nobs)    # create y as adjusted
                                                      # random normal variate 

# Fit
toy_data = {}                  # build data dictionary
toy_data['nobs'] = nobs        # sample size
toy_data['x'] = x1             # explanatory variable
toy_data['y'] = y              # response variable

# STAN code
stan_code = """
data {
    int<lower=0> nobs;                                 
    vector[nobs] x;                       
    vector[nobs] y;                       
}
parameters {
    real beta0;
    real beta1;                                                
    real<lower=0> sigma;               
}
model {
    vector[nobs] mu;

    mu = beta0 + beta1 * x;

    y ~ normal(mu, sigma);             # Likelihood function
}
"""

fit = pystan.stan(model_code=stan_code, data=toy_data, iter=5000, chains=3, verbose=False, n_jobs=3)

# Output
nlines = 8                     # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   


# Plot
import pylab as plt

fit.plot(['beta0', 'beta1', 'sigma'])
plt.tight_layout()
plt.show()
