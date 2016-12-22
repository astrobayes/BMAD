# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.2 - GLM logistic regression in Python
# 1 response (z) and 1 explanatory variable (x)

import numpy as np
import statsmodels.api as sm

# Data
x = np.array([13, 10, 15, 9, 18, 22, 29, 13, 17, 11, 27, 21, 16, 14, 18, 8])
y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0])

X = np.transpose(x) 
X = sm.add_constant(X)                            # add intercept

# Fit
mu = (y + 0.5) / 2                                # initialize mu
eta = np.log(mu/(1-mu))                           # initialize eta with the Bernoulli link

for i in range(8):
    w = mu * (1 - mu);                            # variance function
    z = eta + (y - mu)/(mu * (1 - mu))            # working response
    mod = sm.WLS(z, X, weights=w).fit()           # weigthed regression
    eta = mod.predict()                           # linear predictor
    mu = 1/(1 + np.exp(-eta))                     # fitted value
    print(mod.params)                             # print iteration log

# Output
print(mod.summary())

# Write data as dictionary
mydata = {}
mydata['x'] = x
mydata['y'] = y

# fit using glm package
import statsmodels.formula.api as smf

mylogit = smf.glm(formula='y ~ x', data=mydata, family=sm.families.Binomial())
res = mylogit.fit()
print(res.summary())
