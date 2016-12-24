# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.20 - Synthetic probit data and model generated in R

set.seed(135)
nobs <- 1:2000

x1 <- runif(nobs)
x2 <- 2*runif(nobs)

xb <- 2 + 0.75 * x1 - 1.25 * x2

exb <- pnorm(xb)                               # probit inverse link
py <- rbinom(nobs, size=1, prob=exb)

probdata <- data.frame(py, x1, x2)

# Code 5.21 - Probit model using R

library(MCMCpack)

mypL <- MCMCprobit(py ~ x1 + x2,
                   burnin = 5000,
                   mcmc = 100000,
                   data = probdata)

summary(mypL)
plot(mypL)