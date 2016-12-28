# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.5 - Bayesian zero-inflated negative binomial 
#            model in Python using Stan

# 1 response (y) and 2 explanatory variables (x1, x2)

import numpy as np
import pystan 
import statsmodels.api as sm

from rpy2.robjects import r, FloatVector
from scipy.stats import uniform, bernoulli


def gen_zinegbinom(N, mu1, mu2, alpha):
    """Zero inflated negative binomial distribution."""
    
    # load R package
    r('require(VGAM)')
   
    # get R functions
    zinbinomR = r['rzinegbin'] 
    res = zinbinomR(n=N, munb=FloatVector(mu1), size=1.0/alpha,
                    pstr0=FloatVector(mu2))

    return np.array([int(item) for item in res])


# Data
np.random.seed(141)                         # set seed to replicate example
nobs= 7500                                  # number of obs in model 

x1 = uniform.rvs(size=nobs)
x2 = bernoulli.rvs(0.6, size=nobs)
xb = 1.0 + 2.0 * x1 + 1.5 * x2              # linear predictor
xc = 2.0 - 5.0 * x1 + 3.0 * x2

exb = np.exp(xb)       
exc = 1 / (1 + np.exp(-xc))   
alpha = 2

# create y as adjusted 
zinby = gen_zinegbinom(nobs, exb, exc, alpha) 

X = np.column_stack((x1,x2))
X = sm.add_constant(X)

mydata = {}                                # build data dictionary
mydata['Y'] = zinby                        # response variable
mydata['N'] = nobs                         # sample size
mydata['Xb'] = X                           # predictors         
mydata['Xc'] = X
mydata['Kb'] = X.shape[1]                   # number of coefficients
mydata['Kc'] = X.shape[1] 

# Fit
stan_code = """
data{
    int N;
    int Kb;
    int Kc;
    matrix[N, Kb] Xb;
    matrix[N, Kc] Xc;
    int Y[N];
}
parameters{
    vector[Kc] beta;
    vector[Kb] gamma;
    real<lower=0> alpha;
    
}
transformed parameters{
    vector[N] mu;
    vector[N] Pi;

    mu = exp(Xc * beta);
    for (i in 1:N) Pi[i] = inv_logit(Xb[i] * gamma);
}
model{
    vector[N] LL;

    alpha ~ gamma(0.001, 0.001);


    for (i in 1:N) {
        if (Y[i] == 0) {
            LL[i] = log_sum_exp(bernoulli_lpmf(1| Pi[i]), 
                               bernoulli_lpmf(0| Pi[i]) + 
                               neg_binomial_2_lpmf(Y[i]| mu[i], 1/alpha));
        } else {
            LL[i] = bernoulli_lpmf(0| Pi[i]) + 
                               neg_binomial_2_lpmf(Y[i]| mu[i], 1/alpha);
        }
    }
    target += LL;
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=mydata, iter=7000, chains=3,
                  warmup=3500, n_jobs=3)

# Output
nlines = 12                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   

