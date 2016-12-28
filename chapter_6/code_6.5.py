# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.5 - Simple Poisson model in Python
# 1 response (py) and 1 explanatory variable (x)

import numpy as np
from scipy.stats import uniform, poisson
import statsmodels.api as sm

# Data
np.random.seed(2016)                 # set seed to replicate example
nobs= 1000                           # number of obs in model 

x = uniform.rvs(size=nobs)

xb = 1 + 2 * x                       # linear predictor, xb           
py = poisson.rvs(np.exp(xb))         # create y as adjusted

X = sm.add_constant(x.transpose())

#build model
myp = sm.GLM(py, X, family=sm.families.Poisson()) 

# find parameter values
res = myp.fit()

# print summary to screen
print(res.summary())
