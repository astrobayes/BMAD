# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 4.6 -cMultivariate normal linear model in R using JAGS

require(R2jags)

set.seed(1056)                              # set seed to replicate example
nobs = 5000                                 # number of obs in model

x1 <- runif(nobs)                           # random uniform variable
x2 <- runif(nobs)                           # random uniform variable
beta1 = 2.0                                 # intercept
beta2 = 3.0                                 # 1st coefficient
beta3 = -2.5                                # 2nd coefficient

xb <- beta1 + beta2*x1 + beta3*x2           # linear predictor
y <- rnorm(nobs, xb, sd=1)                  # create y as adjusted random normal variate

# Model setup
X <- model.matrix(~ 1 + x1+x2)
K <- ncol(X)

model.data <- list(Y = y,                  # response variable
                   X = X,                  # predictors
                   K = K,                  # number of predictors including the intercept
                   N = nobs                # sample size
                   )

NORM <- "
model{
    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }

    # Uniform prior for standard deviation
    tau <- pow(sigma, -2) # precision
    sigma ~ dunif(0, 100) # standard deviation

    # Likelihood function
    for (i in 1:N){
        Y[i] ~ dnorm(mu[i],tau)
        mu[i] <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}"

inits <- function () {
  list (
    beta = rnorm(K, 0, 0.01))
}

params <- c ("beta", "sigma")

normOfit <- jags(data = model.data,
                  inits = inits,
                  parameters = params,
                  model = textConnection(NORM),
                  n.chains = 3,
                  n.iter = 15000,
                  n.thin = 1,
                  n.burnin = 10000)

print (normOfit, intervals=c(0.025, 0.975), digits=2)