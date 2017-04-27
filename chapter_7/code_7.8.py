# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.8 - Bayesian Poisson-logit hurdle model in Python using Stan
# 1 response (y) and 1 explanatory variable (x1)
#
# discussion of alternative forms at
# https://groups.google.com/forum/#!msg/stan-users/X1dBkLNel4s/o4NGTJgsCAAJ

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import uniform, bernoulli, poisson

def ztp(N, lambda_):
    """Zero truncated Poisson distribution"""
    temp = [poisson.pmf(0, item) for item in lambda_]
    p = [uniform.rvs(loc=item, scale=1-item) for item in temp]
    ztp = [int(poisson.ppf(p[i], lambda_[i])) for i in range(len(p))]

    return np.array(ztp)


# Data
np.random.seed(141)                                   # set seed to replicate example
nobs= 750                                             # number of obs in model 

x1 = uniform.rvs(size=nobs)

xb = 1.0 + 4.0 * x1                                   # linear predictor, xb
exb = np.exp(xb)       
poy = ztp(nobs, exb)


xc = -1.0 + 3.5 * x1                                  # construct filter
pi = 1.0/(1.0 + np.exp(xc))
bern = [bernoulli.rvs(1-pi[i]) for i in range(nobs)]

poy = [poy[i]*bern[i] for i in  range(nobs)]          # Add structural zeros


X = np.transpose(x1)
X = sm.add_constant(X)

# prepare data for Stan
mydata = {}                                # build data dictionary
mydata['Y'] = poy                          # response variable
mydata['N'] = nobs                         # sample size
mydata['Xb'] = X                           # predictors         
mydata['Xc'] = X
mydata['Kb'] = X.shape[1]                  # number of coefficients
mydata['Kc'] = X.shape[1] 

stan_code = """
data{
    int<lower=0> N;
    int<lower=0> Kb;
    int<lower=0> Kc;
    matrix[N, Kb] Xb;
    matrix[N, Kc] Xc;
    int<lower=0> Y[N];
}
parameters{
    vector[Kc] beta;
    vector[Kb] gamma;
    real<lower=0, upper=5.0> r;
    
}
transformed parameters{
    vector[N] mu;
    vector[N] Pi;

    mu = exp(Xc * beta);
    for (i in 1:N) Pi[i] = inv_logit(Xb[i] * gamma);
}
model{
    for (i in 1:N) {
        (Y[i] == 0) ~ bernoulli(1-Pi[i]);
        if (Y[i] > 0) Y[i] ~ poisson(mu[i]) T[1,];
    }
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=7000, chains=3,
                  warmup=4000, n_jobs=3)

############### Output
nlines = 10                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   



