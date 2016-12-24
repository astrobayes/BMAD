# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# from Code 6.9 - Synthetic negative binomial data and model in R

library(MASS)

set.seed(141)
nobs <- 2500

x1 <- rbinom(nobs,size=1, prob=0.6)
x2 <- runif(nobs)
xb <- 1 + 2.0*x1 - 1.5*x2
a <- 3.3

theta <- 0.303                                 # 1/a
exb <- exp(xb)

nby <- rnegbin(n=nobs, mu=exb, theta=theta)
negbml <- data.frame(nby, x1, x2)

# Code 6.10 - Negative binomial model in R using COUNT

library(COUNT)

nb3 <- nbinomial(nby ~ x1 + x2, data=negbml)
summary(nb3)