# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 6.17 - Negative binomial model in Python using Stan
# 1 response (nby) and 2 explanatory variables (x1, x2)
#
# useful link: http://bebi103.caltech.edu/2015/tutorials/r7_pymc3.html

import numpy as np
import pystan 

from scipy.stats import uniform, binom, nbinom
import statsmodels.api as sm

# Data
np.random.seed(141)                 # set seed to replicate example
nobs= 2500                          # number of obs in model 

x1 = binom.rvs(1, 0.6, size=nobs)   # categorical explanatory variable
x2 = uniform.rvs(size=nobs)         # real explanatory variable

theta = 0.303
X = sm.add_constant(np.column_stack((x1, x2)))
beta = [1.0, 2.0, -1.5]
xb = np.dot(X, beta)          # linear predictor

exb = np.exp(xb)
nby = nbinom.rvs(exb, theta)

mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = X                            # predictors         
mydata['Y'] = nby                          # response variable
mydata['K'] = len(beta)
  

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
    real<lower=0, upper=5> alpha;
}
transformed parameters{
    vector[N] mu;
   
    mu = exp(X * beta);
}
model{
    Y ~ neg_binomial(mu, alpha);
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=10000, chains=3,
                  warmup=5000, n_jobs=3)

# Output
nlines = 9                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

