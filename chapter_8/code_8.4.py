# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.4 - Random intercept normal model in Python using Stan

import numpy as np
import statsmodels.api as sm
import pystan

from scipy.stats import norm, uniform

# Data
np.random.seed(1656)                 # set seed to replicate example
N = 4500                           # number of obs in model 
NGroups = 20

x1 = uniform.rvs(size=N)
x2 = uniform.rvs(size=N)
Groups = np.array([225 * [i] for i in range(20)]).flatten()

# use this if you want random values for a
#a = norm.rvs(loc=0, scale=0.5, size=NGroups)

# use  values from code 8.1
a = np.array([0.579, -0.115, -0.125, 0.169, -0.500, -1.429, -1.171, -0.205, 0.193,
0.041, -0.917, -0.353, -1.197, 1.044, 1.084, -0.085, -0.886, -0.352,
-1.398, 0.350])

mu = 1 + 0.2 * x1 - 0.75 * x2 + a[list(Groups)]
y = norm.rvs(loc=mu, scale=2, size=N)

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


# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    int<lower=0> NGroups;
    matrix[N, K] X;
    real Y[N];
    int re[N];
    vector[K] b0;
    matrix[K, K] B0;
    vector[NGroups] a0;
    matrix[NGroups, NGroups] A0;
}
parameters{
    vector[K] beta;
    vector[NGroups] a;
    real<lower=0> sigma_plot;
    real<lower=0> sigma_eps;
}
transformed parameters{
    vector[N] eta;
    vector[N] mu; 
 
    eta = X * beta;
    for (i in 1:N){ 
        mu[i] = eta[i] + a[re[i]+1];
    }
}
model{    
    sigma_plot ~ cauchy(0, 5);
    sigma_eps ~ cauchy(0, 25);

    beta ~ multi_normal(b0, B0);
    a ~ multi_normal(a0, sigma_eps * A0);

    Y ~ normal(mu, sigma_eps);  
}
"""

fit = pystan.stan(model_code=stan_code, data=model_data, iter=10000, chains=3, thin=10,
                  warmup=6000, n_jobs=3)

# Output
nlines = 30                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)  
