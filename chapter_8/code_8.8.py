# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.9b Random intercept binomial logistic model in Python using Stan


import numpy as np
import statsmodels.api as sm
import pystan 

from scipy.stats import norm, uniform, bernoulli

y = [6,11,9,13,17,21,8,10,15,19,7,12,8,5,13,17,5,12,9,10]
m = [45,54,39,47,29,44,36,57,62,55,66,48,49,39,28,35,39,43,50,36]
x1 = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
x2 = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
Groups = range(len(y))

X = sm.add_constant(np.column_stack((x1,x2)))
K = X.shape[1]

model_data = {}
model_data['Y'] = y                             # response
model_data['X'] = X                             # covariates
model_data['K'] = K                             # num. betas
model_data['m'] = m                             # binomial denominator
model_data['N'] = len(y)                        # sample size
model_data['re'] = Groups                       # random effects
model_data['b0'] = np.repeat(0, K) 
model_data['B0'] = np.diag(np.repeat(100, K))
model_data['a0'] = np.repeat(0, len(y))
model_data['A0'] = np.diag(np.repeat(1, len(y)))


# Fit
stan_code = """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    int<lower=0> Y[N];
    int re[N];
    int m[N];
    vector[K] b0;
    matrix[K, K] B0;
    vector[N] a0;
    matrix[N, N] A0;
}
parameters{
    vector[K] beta;
    vector[N] a;
    real<lower=0> sigma;
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
    sigma ~ cauchy(0, 25);

    beta ~ multi_normal(b0, B0);
    a ~ multi_normal(a0, sigma * A0);

    Y ~ binomial_logit(m, p);  
}
"""

fit = pystan.stan(model_code=stan_code, data=model_data, iter=5000, chains=3, thin=10,
                  warmup=4000, n_jobs=3)

# Output
nlines = 29                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)  



