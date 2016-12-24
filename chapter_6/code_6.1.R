# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 6.1 - Synthetic data following a Poisson distribution in R

set.seed(2016)
nobs <- 500

x <- runif(nobs)
xb <- 1 + 2*x

py <- rpois(nobs, exp(xb))
summary(myp <- glm(py ~ x, family=poisson))