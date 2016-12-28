# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.6 - Multivariate Poisson model in Python
# 1 response (py) and 2 explanatory variables (x1_2, x2)

import numpy as np
from scipy.stats import norm, binom, poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Data
np.random.seed(18472)                # set seed to replicate example
nobs= 750                            # number of obs in model 

x1_2 = binom.rvs(1, 0.7, size=nobs)
x2 = norm.rvs(loc=0, scale=1.0, size=nobs)

xb = 1 - 1.5 * x1_2  - 3.5 * x2      # linear predictor, xb           
exb = np.exp(xb)
py = poisson.rvs(exb)         # create y as adjusted

my_data = {}
my_data['x1_2'] = x1_2
my_data['x2'] = x2
my_data['py'] = py

#build model
myp = smf.glm('py ~ x1_2 + x2', data=my_data, family=sm.families.Poisson()) 

# find parameter values
res = myp.fit()

# Output
print(res.summary())
