# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.23 - Zero-truncated Poisson model in Python using Stan
# 1 response (y) and 3 explanatory variable (x1, x2, x3)

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import uniform, binom, poisson

def ztpoisson(N, lambda_par):
    """Zero truncated Poisson distribution."""

    temp = poisson.pmf(0, lambda_par)                
    p = [uniform.rvs(loc=item, scale=1-item) for item in temp]
    ztp = [int(poisson.ppf(p[i],lambda_par[i])) for i in range(N)]
  
    return np.array(ztp)


# Data
np.random.seed(123579)                  # set seed to replicate example
nobs= 3000                              # number of obs in model 

x1 = binom.rvs(1, 0.3, size=nobs)
x2 = binom.rvs(1, 0.6, size=nobs)
x3 = uniform.rvs(size=nobs)

xb = 1.0 + 2.0 * x1 - 3.0 * x2 - 1.5 * x3    # linear predictor, xb
exb = np.exp(xb)          

ztpy = ztpoisson(nobs, exb)                 # create y as adjusted

X = np.column_stack((x1,x2,x3))
X = sm.add_constant(X)

# Fit
mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = X                            # predictors         
mydata['Y'] = ztpy                         # response variable
mydata['K'] = X.shape[1]                   # number of coefficients

stan_code = """
data{
    int N;
    int K;
    matrix[N, K] X;
    int Y[N];
}
parameters{
    vector[K] beta;
}
model{
    vector[N] mu;

    mu = exp(X * beta);

    # likelihood
    for (i in 1:N) Y[i] ~ poisson(mu[i]) T[1,];
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
                  warmup=4000, n_jobs=3)

# Output
print(fit) 

