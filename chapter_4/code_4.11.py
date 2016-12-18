# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 4.11 - Normal linear model in Python using Stan and
#             including errors in variables

# 1 response (y) and 1 explanatory variable (x1)


import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import norm

############### Data
np.random.seed(42)                      # set seed to replicate example
nobs = 1000                             # number of obs in model 
sdobsx = 1.25
truex =  norm.rvs(0,2.5, size=nobs)     # normal variable
errx = norm.rvs(0, sdobsx, size=nobs)   # errors
obsx = truex + errx                     # observed

beta0 = -4
beta1 = 7           
sdy = 1.25
sdobsy = 2.5

erry = norm.rvs(0, sdobsy, size=nobs)
truey = norm.rvs(beta0 + beta1*truex, sdy, size=nobs)
obsy = truey + erry

# Fit
toy_data = {}                                # build data dictionary
toy_data['N'] = nobs                         # sample size
toy_data['obsx'] = obsx                      # explanatory variable       
toy_data['errx'] = errx                      # uncertainty in explanatory variable
toy_data['obsy'] = obsy                      # response variable
toy_data['erry'] = erry                      # uncertainty in response variable
toy_data['xmean'] = np.repeat(0, nobs)       # initial guess for true x position


# STAN code
stan_code = """
data {
    int<lower=0> N;                                 
    vector[N] obsx;                     
    vector[N] obsy;     
    vector[N] errx; 
    vector[N] erry;     
    vector[N] xmean;        
}
transformed data{
    vector[N] varx;
    vector[N] vary;

    for (i in 1:N){ 
        varx[i] = fabs(errx[i]);
        vary[i] = fabs(erry[i]);
    }
}
parameters {
    real beta0;
    real beta1;                                             
    real<lower=0> sigma;
    vector[N] x;
    vector[N] y; 
}
transformed parameters{
    vector[N] mu;

    for (i in 1:N){ 
        mu[i] = beta0 + beta1 * x[i];
    }
}
model{
    beta0 ~ normal(0.0, 100);                # Diffuse normal priors for predictors
    beta1 ~ normal(0.0, 100);

    sigma ~ uniform(0.0, 100);                # Uniform prior for standard deviation

    x ~ normal(xmean, 100);
    obsx ~ normal(x, varx);
    y ~ normal(mu, sigma);
    obsy ~ normal(y, vary);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=toy_data, iter=5000, chains=3,
                  n_jobs=3, warmup=2500, verbose=False, thin=1)

# Output
nlines = 8                                   # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   


