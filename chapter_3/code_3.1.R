# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 3.1 Basic linear model in R.

# Data
set.seed(1056)                      # set seed to replicate example
nobs = 250                          # number of obs in model
x1 <- runif(nobs)                   # random uniform variable
alpha = 2                           # intercept
beta = 3                            # angular coefficient
xb <- alpha + beta* x1              # linear predictor, xb
y <- rnorm(nobs, xb, sd=1)          # create y as adjusted random normal variate

# Fit
summary(mod <- lm(y ~ x1))          # model of the synthetic data.

# Output
ypred <- predict(mod,type="response")                # prediction from the model
plot(x1,y,pch=19,col="red")                          # plot scatter
lines(x1,ypred,col='grey40',lwd=2)                   # plot regression line
segments(x1,fitted(mod),x1,y,lwd=1,col="gray70")     # add the residuals