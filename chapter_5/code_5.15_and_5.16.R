# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.15  - Synthetic data from logistic model in R

set.seed(13979)
nobs <- 5000

x1 <- rbinom(nobs, size = 1, 0.6)
x2 <- runif(nobs)
xb <- 2 + 0.75*x1 - 5*x2

exb <- 1/(1+exp(-xb))
by <- rbinom(nobs, size = 1, prob = exb)

logitmod <- data.frame(by, x1, x2)

# Code 5.16 Logistic model using R
library(MCMCpack)
myL <- MCMClogit(by ~ x1 + x2,
                 burnin = 5000,
                 mcmc = 10000,
                 data = logitmod)

summary(myL)


plot(myL)                             # Produces figure
