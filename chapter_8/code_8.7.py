# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 8.5b - Simulated random intercept binary logistic data

import numpy as np
import statsmodels.api as sm

from scipy.stats import norm, uniform, bernoulli

# Data
np.random.seed(13531)                 # set seed to replicate example
N = 4000                              # number of obs in model 
NGroups = 20

x1 = uniform.rvs(size=N)
x2 = uniform.rvs(size=N)
Groups = np.array([200 * [i] for i in range(20)]).flatten()

a = norm.rvs(loc=0, scale=0.5, size=NGroups)
eta = 1 + 0.2 * x1 - 0.75 * x2 + a[list(Groups)]
mu = 1.0/(1.0 + np.exp(-eta))
y = bernoulli.rvs(mu, size=N)

X = sm.add_constant(np.column_stack((x1,x2)))
K = X.shape[1]
re = Groups
Nre = NGroups


model_data = {}
model_data['Y'] = y
model_data['X'] = X                              
model_data['K'] = K
model_data['N'] = N
model_data['NGroups'] = NGroups
model_data['re'] = re
model_data['b0'] = np.repeat(0, K) 
model_data['B0'] = np.diag(np.repeat(100, K))
model_data['a0'] = np.repeat(0, Nre)
model_data['A0'] = np.diag(np.repeat(1, Nre))

# Code 8.7b - Bayesian random intercept binary logistic model in Python using Stan

import pystan

# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    int<lower=0> NGroups;
    matrix[N, K] X;
    int<lower=0> Y[N];
    int re[N];
    vector[K] b0;
    matrix[K, K] B0;
    vector[NGroups] a0;
    matrix[NGroups, NGroups] A0;
}
parameters{
    vector[K] beta;
    vector[NGroups] a;
    real<lower=0> sigma_re;
}
transformed parameters{
    vector[N] eta;
    vector[N] p; 
 
    eta = X * beta;
    for (i in 1:N){ 
        p[i] = eta[i] + a[re[i]+1];
    }
}
model{    
    sigma_re ~ cauchy(0, 25);

    beta ~ multi_normal(b0, B0);
    a ~ multi_normal(a0, sigma_re * A0);

    Y ~ bernoulli_logit(p);  
}
"""

fit = pystan.stan(model_code=stan_code, data=model_data, iter=10000, chains=3, thin=10,
                  warmup=6000, n_jobs=3)

# Output
nlines = 29                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)  


