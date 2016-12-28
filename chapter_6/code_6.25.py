# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.25 - Zero-truncated negative binomial model in Python using Stan 
# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import pystan 
import statsmodels.api as sm

from scipy.stats import uniform, nbinom, bernoulli

def gen_ztnegbinom(n, mu, size):
    """Zero truncated negative binomial distribution.

        input:  n, int
                number of successes

                mu, float or int
                number of trials

                size, float
                probability of success

        output: ztnb, list of int
                draws from a zero truncated negative binomial distribution
    """

    temp = nbinom.pmf(0, mu, size)
    p = [uniform.rvs(loc=temp[i], scale=1-temp[i]) for i in range(n)]
    ztnb = [int(nbinom.ppf(p[i], mu[i], size)) for i in range(n)]

    return np.array(ztnb)

# Data
np.random.seed(123579)                 # set seed to replicate example
nobs= 2000                              # number of obs in model 

x1 = bernoulli.rvs(0.7, size=nobs)
x2 = uniform.rvs(size=nobs)

xb = 1.0 + 2.0 * x1 - 4.0 * x2             # linear predictor
exb = np.exp(xb)          
alpha = 5

# create y as adjusted 
ztnby = gen_ztnegbinom(nobs, exb, 1.0/alpha)  

X = np.column_stack((x1,x2))
X = sm.add_constant(X)

mydata = {}                                # build data dictionary
mydata['N'] = nobs                         # sample size
mydata['X'] = X                            # predictors         
mydata['Y'] = ztnby                        # response variable
mydata['K'] = X.shape[1]                   # number of coefficients

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
    real<lower=1> alpha;
}
model{
    vector[N] mu;

    # covariates transformation
    mu = exp(X * beta);      
   
    # likelihood
    for (i in 1:N) Y[i] ~ neg_binomial(mu[i], 1.0/(alpha - 1.0)) T[0,];
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

