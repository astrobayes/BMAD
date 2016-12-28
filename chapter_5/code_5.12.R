# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.12 - Synthetic beta-distributed data generated in R

set.seed(1056)                    # set seed to replicate example
nobs <- 1000                      # number of obs in model
x1 <- runif(nobs)                 # random normal variable
xb <- 0.3+1.5*x1                  # linear predictor, xb
exb <- exp(-xb)                   # prob of 0
p <- exb/(1+exb)                  # prob of 1
theta <- 15
y <- rbeta(nobs,theta*(1-p),theta*p)

summary(y)

hist(y)
