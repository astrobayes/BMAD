# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 6.28 - Negative binomial model with three parameters in Python using Stan. 
#             Synthetic data generated with R package MASS

# 1 response (nby) and 1 explanatory variables (x1)


import numpy as np
import pystan

import statsmodels.api as sm
from rpy2.robjects import r, FloatVector
from scipy.stats import uniform, binom, nbinom, poisson, gamma


def gen_negbin(N, mu1, theta1):
    """Negative binomial distribution."""

    # load R package
    r('library(MASS)')

    # get R functions
    nbinomR = r['rnegbin']

    res = nbinomR(n=N, mu=FloatVector(mu1), theta=FloatVector(theta1))

    return res 

# Data
nobs= 1500                         # number of obs in model 

x1 = uniform.rvs(size=nobs)        # categorical explanatory variable

xb = 2 - 5 * x1                    # linear predictor
exb = np.exp(xb)

theta = 0.5
Q = 1.4

nbpy = gen_negbin(nobs, exb, theta * (exb ** Q))


X = sm.add_constant(np.transpose(x1))      # format data for input

mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = X                             # predictors         
mydata['Y'] = nbpy                          # response variable
mydata['K'] = X.shape[1]
  
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
    real<lower=0> theta;
    real<lower=0, upper=3> Q;
}
transformed parameters{
    vector[N] mu;
    real<lower=0> theta_eff[N];
   
    mu = exp(X * beta);
    for (i in 1:N) {
        theta_eff[i] = theta * pow(mu[i], Q);
    }
}
model{        

    Y ~ neg_binomial_2(mu, theta_eff);
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
                  warmup=2500, n_jobs=3)

# Output
nlines = 9                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

