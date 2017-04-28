# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

#  Data from Code 6.2

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
