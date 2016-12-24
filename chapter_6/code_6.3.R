# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

#  from Code 6.2 Synthetic Poisson data and model in R: 
#               binary and continuous predictors

set.seed(18472)
nobs <- 750

x1_2 <- rbinom(nobs,size=1,prob=0.7)
x2 <- rnorm(nobs,0,1)
xb <- 1 - 1.5*x1_2 - 3.5*x2

exb <- exp(xb)
py <- rpois(nobs, exb)
pois <- data.frame(py, x1_2, x2)

# Code 6.3 - Bayesian Poisson model using R

library(MCMCpack)

mypoisL <- MCMCpoisson(py ~ x1_2 + x2,
                       burnin = 5000,
                       mcmc = 10000,
                       data = pois)
 
summary(mypoisL)

plot(mypoisL)