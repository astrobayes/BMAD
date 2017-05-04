# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 10.17 Negative binomial model in Python using Stan, 
#            for modeling the relationship between globular 
#            cluster population and host galaxy visual magnitude 
#
# Statistical Model: Negative Binomial regression in Python using Stan
#
# Astronomy case: Relation between galaxy visual magnitude and its 
#                 globular cluster population
#                 analysis presented in de Souza et al., 2015
#                 MNRAS, 453, Issue 2, p.1928-1940
#
# 1 response (obsy - size of globular cluster population) 
# 1 explanatory variable (obsx - visual magnitude)
#
# Data from: http://www.physics.mcmaster.ca/~harris/GCS_table.txt

import numpy as np
import pandas as pd
import pystan 
import statsmodels.api as sm

# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p7/GCs.csv'

data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['X'] = sm.add_constant(np.array(data_frame['MV_T']))
data['Y'] = np.array(data_frame['N_GC'])
data['N'] = len(data['X'])
data['K'] = 2

# Fit
stan_code="""
data{
    int<lower=0> N;           # number of data points
    int<lower=1> K;           # number of linear predictor coefficients  
    matrix[N,K] X;            # galaxy visual magnitude
    int Y[N];                 # size of globular cluster population
}
parameters{
    vector[K] beta;           # linear predictor coefficients
    real<lower=0> theta;
}
model{
    vector[N] mu;             # linear predictor

    mu = exp(X * beta);

    theta ~ gamma(0.001, 0.001);

    # likelihood
    Y ~ neg_binomial_2(mu, theta);
}
generated quantities{
    real dispersion;
    vector[N] expY;           # mean
    vector[N] varY;           # variance
    vector[N] PRes;
    vector[N] mu2;

    mu2 = exp(X * beta);
    expY = mu2;
 
    for (i in 1:N){ 
        varY[i] = mu2[i] + pow(mu2[i], 2) / theta;
        PRes[i] = pow((Y[i] - expY[i]) / sqrt(varY[i]),2); 
    }

    dispersion = sum(PRes) / (N - (K + 1));
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=10000, chains=3,
                  warmup=5000, thin=1, n_jobs=3)

# Output
nlines = 9                                 # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item) 
