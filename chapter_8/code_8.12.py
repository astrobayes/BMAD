# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.12b Random intercept random slopes poisson model in Python using Stan


import numpy as np
import pystan
import statsmodels.api as sm

from scipy.stats import norm, uniform, poisson

# Data
np.random.seed(1656)                 # set seed to replicate example
N = 5000                             # number of obs in model 
NGroups = 10

x1 = uniform.rvs(size=N)
x2 = np.array([0 if item <= 0.5 else 1 for item in x1])

Groups = np.array([500 * [i] for i in range(NGroups)]).flatten()
a = norm.rvs(loc=0, scale=0.1, size=NGroups)
b = norm.rvs(loc=0, scale=0.35, size=NGroups)
eta = 1 + 4 * x1 - 7 * x2 + a[list(Groups)] + b[list(Groups)] * x1
mu = np.exp(eta)

y = poisson.rvs(mu)


X = sm.add_constant(np.column_stack((x1,x2)))
K = X.shape[1]

model_data = {}
model_data['Y'] = y
model_data['X'] = X                              
model_data['K'] = K
model_data['N'] = N
model_data['NGroups'] = NGroups
model_data['re'] = Groups
model_data['b0'] = np.repeat(0, K) 
model_data['B0'] = np.diag(np.repeat(100, K))
model_data['a0'] = np.repeat(0, NGroups)
model_data['A0'] = np.diag(np.repeat(1, NGroups))

# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    int<lower=0> NGroups;
    matrix[N, K] X;
    int Y[N];
    int re[N];
    vector[K] b0;
    matrix[K, K] B0;
    vector[NGroups] a0;
    matrix[NGroups, NGroups] A0;
}
parameters{
    vector[K] beta;
    vector[NGroups] a;
    vector[NGroups] b;
    real<lower=0> sigma_ri;
    real<lower=0> sigma_rs;
}
transformed parameters{
    vector[N] eta;
    vector[N] mu; 
 
    eta = X * beta;
    for (i in 1:N){ 
        mu[i] = exp(eta[i] + a[re[i]+1] + b[re[i] + 1] * X[i,2]);
    }
}
model{    
    sigma_ri ~ gamma(0.01, 0.01);
    sigma_rs ~ gamma(0.01, 0.01);

    beta ~ multi_normal(b0, B0);
    a ~ multi_normal(a0, sigma_ri * A0);
    b ~ multi_normal(a0, sigma_rs * A0);

    Y ~ poisson(mu);  
}
"""

fit = pystan.stan(model_code=stan_code, data=model_data, iter=4000, chains=3, thin=10,
                  warmup=3000, n_jobs=3)

# Output
nlines = 30                                  # number of lines in screen output

output = str(fit).split('\n')

for item in output[:nlines]:
    print(item)  



