# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 2.3 - Example of linear regression in Python


import numpy as np
import statsmodels.formula.api as smf

# Data
y = np.array([13,15,9,17,8,5,19,23,10,7,10,6])   # continuous response variable
x1 = np.array([1,1,1,1,1,1,0,0,0,0,0,0])         # binary predictor
x2 = np.array([1,1,1,1,2,2,2,2,3,3,3,3])         # categorical predictor

mydata = {}                                      # create data dictionary
mydata['x1'] = x1
mydata['x2'] = x2                                  
mydata['y'] = y

# Fit using ordinary least squares
results = smf.ols(formula='y ~ x1 + x2', data=mydata).fit()

# Output
print(str(results.summary()))

