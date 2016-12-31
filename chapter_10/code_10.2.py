# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Chapter 8 - Astronomical Applications 
#
# Statistical Model: Gaussian regression considering errors in variables
#                    in Python using Stan
#
# Astronomy case: Relation between mass of galaxy central supermassive black hole
#                 and its stelar bulge velocity dispersion
#                 taken from Harris, Poole and Harris, 2013, MNRAS, 438 (3), p.2117-2130
#
# 1 response (obsy - mass) and 1 explanatory variable (obsx - velocity dispersion)
#
# Data from: http://www.physics.mcmaster.ca/~harris/GCS_table.txt

import numpy as np
import pandas as pd
import pystan 

path_to_data = '../data/Section_10p1/M_sigma.csv'

# read data
data_frame = dict(pd.read_csv(path_to_data))

# prepare data for Stan
data = {}
data['obsx'] = np.array(data_frame['obsx'])
data['errx'] = np.array(data_frame['errx'])
data['obsy'] = np.array(data_frame['obsy'])
data['erry'] = np.array(data_frame['erry'])
data['N'] = len(data['obsx'])

# Stan Gaussian model with errors
stan_code="""
data{
    int<lower=0> N;                   # number of data points
    vector[N] obsx;                   # obs velocity dispersion
    vector<lower=0>[N] errx;          # errors in velocity dispersion measurements
    vector[N] obsy;                   # obs black hole mass
    vector<lower=0>[N] erry;          # errors in black hole mass measurements
}
parameters{
    real alpha;                       # intercept
    real beta;                        # angular coefficient
    real<lower=0> epsilon;            # scatter around true black hole mass
    vector[N] x;                      # true velocity dispersion
    vector[N] y;                      # true black hole mass
}
model{

    # likelihood  
    obsx ~ normal(x, errx);
    y ~ normal(alpha + beta * x, epsilon);
    obsy ~ normal(y, erry);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, data=data, iter=15000, chains=3,
                  warmup=5000, thin=10, n_jobs=3)

############### Output
nlines = 8                                  # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item) 
