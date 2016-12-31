# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 9.1 - Synthetic multivariate data in R

require(MASS)

# Data
set.seed(1056)
nobs <- 500                      # number of samples
nvar <- 15                       # number of predictors
rho <- 0.6                       # correlation between predictors

p <- rbinom(nvar,1,0.2)
beta <- round(p*rnorm(nvar,0,5),2)

# Check the coefficients
print(beta,2)

# Covariance matrix
d <- length(beta)
Sigma <- toeplitz(c(1, rep(rho, d - 1)))
Mu <- c(rep(0,d))

# Multivariate sampling
M <- mvrnorm(nobs, mu = Mu, Sigma = Sigma )
xb <- M %*% beta

# Dependent variable
y <- rnorm(nobs, xb, sd = 2)

require(corrplot)

corrplot(cor(M), method="number",type="upper")