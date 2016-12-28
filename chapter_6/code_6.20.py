# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.20 - Generalized Poisson model in Python using Stan
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
import pystan 
import statsmodels.api as sm

from  scipy.misc import factorial
from scipy.stats import uniform, rv_discrete


def sign(delta):
    """Returns a pair of vectors to set sign on 
       generalized Poisson distribution.

       input: delta, scalar
              extra parameter from generalized Poisson

       output: value, sig
               pair of scalars
               value -> absolute value of delta
               sig -> if delta < 0, sig = 0.5
                       else sign > 1.5
    """
    if delta > 0:
        value = delta                      
        sig = 1.5
    else:
        value = abs(delta)
        sig = 0.5

    return value, sig

class gpoisson(rv_discrete):
    """Generalized Poisson distribution."""
   
    def _pmf(self, n, mu, delta, sig):

        if sig < 1.0:
            delta1 = -delta
        else:
            delta1 = delta

        term1 = mu * ((mu + delta1 * n) ** (n - 1))
        term2 = np.exp(-mu- n * delta1) / factorial(n)

        return term1 * term2

# Data
np.random.seed(160)                 # set seed to replicate example
nobs= 1000                              # number of obs in model 

x1 = uniform.rvs(size=nobs)

xb = 1.0 + 3.5 * x1                       # linear predictor, xb
delta = -0.3

exb = np.exp(xb)          

gen_poisson = gpoisson(name="gen_poisson", shapes='mu, delta, sig') 
gpy = [gen_poisson.rvs(exb[i], 
       sign(delta)[0], sign(delta)[1]) for i in range(nobs)]      


mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = sm.add_constant(np.transpose(x1))                            # predictors         
mydata['Y'] = gpy                           # response variable
mydata['K'] = 2                             # number of coefficients


# Fit
stan_code = """
data{
    int N;
    int K;
    matrix[N, K] X;
    int Y[N];
}
parameters{
    vector[K] beta;
    real<lower=-1, upper=1> delta;
}
transformed parameters{
    vector[N] mu;

    mu = exp(X * beta);
}
model{
    vector[N] l1;
    vector[N] l2;
    vector[N] LL;

    delta ~ uniform(-1, 1);

    for (i in 1:N){
        l1[i] = log(mu[i]) + (Y[i] - 1) * log(mu[i] + delta * Y[i]);
        l2[i] = mu[i] + delta * Y[i] + lgamma(Y[i] + 1);
        LL[i] = l1[i] - l2[i];
    }

   target += LL;
}
"""


# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=5000, chains=3,
                  warmup=4000, n_jobs=3)

# Output
nlines = range(9)          # lines in screen output

output = str(fit).split('\n')
for i in nlines:
    print(output[i])   

