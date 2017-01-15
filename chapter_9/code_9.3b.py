# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.3b 

import numpy as np
import pystan

from scipy.linalg import toeplitz
from scipy.stats import norm, uniform, nbinom, multivariate_normal, bernoulli

# Data
np.random.seed(1056)

nobs = 500
nvar = 15
rho = 0.6
p = bernoulli.rvs(0.2, size=nvar)

# use these to generate new values of beta
#beta1 = p * norm.rvs(loc=0, scale=5.0, size=nvar)
#beta = np.array([round(item, 2) for item in beta1])

# use same beta values as Code 9.1
beta = np.array([-2.43, 1.60, 0.00, 5.01, 4.12, 0.00, 0.00, 0.00, -0.89, 0.00, -2.31, 0.00, 0.00, 0.00, 0.00]) 

# covariance matrix
d = beta.shape[0]
Sigma = toeplitz(np.insert(np.repeat(rho, d-1), 0, 1))


# multivariate sampling - default mean is zero
M = multivariate_normal.rvs(cov=Sigma, size=nobs)
xb = np.dot(M, beta)

y = [norm.rvs(loc=xb[i], scale=2.0) for i in range(xb.shape[0])]

# fit
mydata = {}
mydata['X'] = M - 1.0
mydata['K'] = mydata['X'].shape[1]
mydata['Y'] = y
mydata['N'] = nobs
mydata['PInd'] = 0.2

stan_model = '''
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    vector[N] Y;
    real PInd;
}
parameters{
    vector[K] beta;
    real<lower=0> sigma; 
    real<lower=0> sdBeta;
}
transformed parameters{
    vector[N] mu;
    real<lower=0> tau;
    real<lower=0> tauBeta;

    mu = X * beta;
    tau = pow(sigma, 2);
    tauBeta = pow(sdBeta, 1);
}
model{
    sdBeta ~ gamma(0.01, 0.01);

    for (i in 1:K) beta[i] ~ double_exponential(0, tauBeta);
 
    sigma ~  gamma(0.01, 0.01);

    Y ~ normal(mu, tau);
}
'''

fit = pystan.stan(model_code=stan_model, data=mydata, iter=5000, chains=3, thin=1,
                  warmup=2500, n_jobs=3)

# Output
nlines = 21                                 # number of lines in screen output

output = str(fit).split('\n')

for item in output[:nlines]:
    print(item)  


