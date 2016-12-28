# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.9 - Synthetic negative binomial data and model in R

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

nb2 <- glm.nb(nby ~ x1 + x2, data=negbml)

summary(nb2)
