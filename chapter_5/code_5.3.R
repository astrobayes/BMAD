# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.3 - Synthetic lognormal data generated in R


# Data
set.seed(1056)                                         # set seed to replicate example
nobs = 5000                                            # number of observations in model
x1 <- runif(nobs)                                      # random uniform variable
xb <- 2 + 3*x1                                         # linear predictor, xb
y <- rlnorm(nobs, xb, sdlog=1)                         # create y as random lognormal variate

require(gamlss) 
summary(mylnm <- gamlss(y ~ x1, family=LOGNO)) 

