# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.10 - Bayesian log-gamma-logit hurdle model in Python using Stan
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import uniform, gamma, bernoulli

# Data
np.random.seed(33559)                          # set seed to replicate example
nobs= 1000                                     # number of obs in model 

# Generate predictors, design matrix
x1 = uniform.rvs(loc=0, scale=4, size=nobs)
xc = -1 + 0.75 * x1
exc = np.exp(xc)
phi = 0.066
r = 1.0/phi
y = np.random.gamma(shape=exc, scale=r)  

# Construct filter
xb = -2 + 1.5 * x1
pi = 1 / (1 + np.exp(-xb))
bern = bernoulli.rvs(1-pi)

gy = [y[i]*bern[i] for i in  range(nobs)]  # Add structural zeros

X = np.transpose(x1)
X = sm.add_constant(X)

mydata = {}                                # build data dictionary
mydata['Y'] = gy                           # response variable
mydata['N'] = nobs                         # sample size
mydata['Xb'] = X                           # predictors         
mydata['Xc'] = X
mydata['Kb'] = X.shape[1]                  # number of coefficients
mydata['Kc'] = X.shape[1] 

# Fit
stan_code = """
data{
    int N;
    int Kb;
    int Kc;
    matrix[N, Kb] Xb;
    matrix[N, Kc] Xc;
    real<lower=0> Y[N];
}
parameters{
    vector[Kc] beta;
    vector[Kb] gamma;
    real<lower=0> phi;
    
}
model{
    vector[N] mu;
    vector[N] Pi;

    mu = exp(Xc * beta);
    for (i in 1:N) Pi[i] = inv_logit(Xb[i] * gamma);

    for (i in 1:N) {
        (Y[i] == 0) ~ bernoulli(Pi[i]);
        if (Y[i] > 0) Y[i] ~ gamma(mu[i], phi) T[0,];
    }
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=6000, chains=3,
                  warmup=4000, n_jobs=3)

# Output
print(fit) 

