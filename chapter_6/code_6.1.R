# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.1 - Synthetic data following a Poisson distribution in R

set.seed(2016)
nobs <- 500

x <- runif(nobs)
xb <- 1 + 2*x

py <- rpois(nobs, exp(xb))
summary(myp <- glm(py ~ x, family=poisson))
