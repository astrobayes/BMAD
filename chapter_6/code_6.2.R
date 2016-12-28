# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.2 Synthetic Poisson data and model in R: 
#          binary and continuous predictors

set.seed(18472)
nobs <- 750

x1_2 <- rbinom(nobs,size=1,prob=0.7)
x2 <- rnorm(nobs,0,1)
xb <- 1 - 1.5*x1_2 - 3.5*x2

exb <- exp(xb)
py <- rpois(nobs, exb)
pois <- data.frame(py, x1_2, x2)
poi <- glm(py ~ x1_2 + x2, family=poisson, data=pois)

summary(poi)
