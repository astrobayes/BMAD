# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 3.2- Ordinary least squares regression in Python without formula
# 1 response (y) and 1 explanatory variables (x1)

import numpy as np
import statsmodels.api as sm
from scipy.stats import uniform, norm

# Data
np.random.seed(1056)                          # set seed to replicate example
nobs= 250                                     # number of obs in model 
x1 = uniform.rvs(size=nobs)                   # random uniform variable

beta0 = 2.0                                   # intercept
beta1 = 3.0                                   # angular coefficient

xb = beta0 + beta1 * x1                       # linear predictor, xb
y = norm.rvs(loc=xb, scale=1.0, size=nobs)    # create y as adjusted
                                              # random normal variate 
# Fit
unity_vec = np.full((nobs,),1, np.float)      # unity vector
X = np.column_stack((unity_vec, x1))          # build data matrix with intercept
results = sm.OLS(y, X).fit()

# Output
print(str(results.summary()))
