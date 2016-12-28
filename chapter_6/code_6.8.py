# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.8 - Bayesian Poisson model in Python using Stan
# 1 response (y) and 2 explanatory variables (x1_2, x2)

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import norm, poisson, binom

# Data
np.random.seed(18472)                     # set seed to replicate example
nobs= 750                                  # number of obs in model 

x1_2 = binom.rvs(1, 0.7, size=nobs)
x2 = norm.rvs(loc=0, scale=1.0, size=nobs)

xb = 1 - 1.5 * x1_2  - 3.5 * x2            # linear predictor, xb           
exb = np.exp(xb)
py = poisson.rvs(exb)                      # create y as adjusted

X = sm.add_constant(np.column_stack((x1_2, x2)))

mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = X                            # predictors         
mydata['Y'] = py                           # response variable
mydata['K'] = mydata['X'].shape[1]         # number of coefficients
  
# Fit
stan_code = """
data{
    int N;
    int K;
    matrix[N,K] X;
    int Y[N];
}
parameters{
    vector[K] beta;
}
model{
    Y ~ poisson_log(X * beta);
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
                  warmup=4000, n_jobs=3)

# Output
print(fit) 

