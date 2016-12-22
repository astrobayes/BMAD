# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.4 - Lognormal model in R using JAGS

require(R2jags)

# Data
set.seed(1056)                                         # set seed to replicate example
nobs = 5000                                            # number of observations in model
x1 <- runif(nobs)                                      # random uniform variable
xb <- 2 + 3*x1                                         # linear predictor, xb
y <- rlnorm(nobs, xb, sdlog=1)                         # create y as random lognormal variate

X <- model.matrix(~ 1 + x1)
K <- ncol(X)
model_data <- list(Y = y, X = X, K = K, N = nobs,
                   Zeros = rep(0, nobs))
LNORM <-"
model{
    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }

    # Uniform prior for standard deviation
    tau <- pow(sigma, -2)                              # precision
    sigma ~ dunif(0, 100)                              # standard deviation

    # Likelihood
    for (i in 1:N){
        Y[i] ~ dlnorm(mu[i],tau)
        mu[i] <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}"

inits <- function () { list(beta = rnorm(K, 0, 0.01)) }

params <- c("beta", "sigma")

LN <- jags(data = model_data,
           inits = inits,
           parameters = params,
           model = textConnection(LNORM),
           n.chains = 3,
           n.iter = 5000,
           n.thin = 1,
           n.burnin = 2500)

print(LN, intervals=c(0.025, 0.975), digits=3)

# plot
source("CH-Figures.R")

out <- LN$BUGSoutput
MyBUGSHist(out,c(uNames("beta",K),"sigma"))

MyBUGSChains(out,c(uNames("beta",K),"sigma"))