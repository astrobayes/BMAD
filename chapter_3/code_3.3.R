# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 3.3 Bayesian normal linear model in R


library(MCMCpack)

# Data
nobs = 5000                               # number of obs in model
x1 <- runif(nobs)                         # random uniform variable
beta0 = 2.0                               # intercept
beta1 = 3.0                               # angular coefficient
xb <- beta0 + beta1*x1                    # linear predictor
y <- rnorm(nobs, xb, sd=1)

# Fit
posteriors <- MCMCregress(y ~ x1, thin=1, seed=1056, burnin=1000,
                          mcmc=10000, verbose=1)

# Output
summary(posteriors)

# Plot
plot(posteriors)