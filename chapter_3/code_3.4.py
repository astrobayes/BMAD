# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 3.4 - Bayesian normal linear model in Python
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
from pymc3 import  Model, sample, summary, traceplot
from pymc3.glm import GLM
import pylab as plt
import pandas
from scipy.stats import uniform, norm

# Data
np.random.seed(1056)                          # set seed to replicate example
nobs= 250                                     # number of obs in model 
x1 = uniform.rvs(size=nobs)                   # random uniform variable

beta0 = 2.0                                   # intercept
beta1 = 3.0                                   # angular coefficient

xb = beta0 + beta1 * x1                       # linear predictor, xb
y = norm.rvs(loc=xb, scale=1.0, size=nobs)    # create y as adjusted

                                              
# Fit
df = pandas.DataFrame({'x1': x1, 'y': y})     # re-write data

with Model() as model_glm:
    GLM.from_formula('y ~ x1', df)
    trace = sample(5000)

# Output
summary(trace)

# show graphical output
traceplot(trace)
plt.show()
