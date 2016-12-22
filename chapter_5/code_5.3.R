# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.3 - Synthetic lognormal data generated in R


# Data
set.seed(1056)                                         # set seed to replicate example
nobs = 5000                                            # number of observations in model
x1 <- runif(nobs)                                      # random uniform variable
xb <- 2 + 3*x1                                         # linear predictor, xb
y <- rlnorm(nobs, xb, sdlog=1)                         # create y as random lognormal variate

require(gamlss) 
summary(mylnm <- gamlss(y ~ x1, family=LOGNO)) 

