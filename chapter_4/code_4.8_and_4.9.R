# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 4.8 - Synthetic normal data in R with errors in measurements

require(R2jags)


# Data
set.seed(1056)                   # set seed to replicate example
nobs = 1000                      # number of obs in model

sdobsx <- 1.25
truex <- rnorm(nobs,0,2.5)       # normal variable
errx <- rnorm(nobs, 0, sdobsx)
obsx <- truex + errx

beta1 <- -4
beta2 <- 7
sdy <- 1.25
sdobsy <- 2.5

erry <- rnorm(nobs, 0, sdobsy)
truey <- rnorm(nobs,beta1 + beta2*truex,sdy)
obsy <- truey + erry

# Code 4.9 - Normal linear model in R using JAGS and 
#            ignoring errors in measurements


K <- 2
model.data <- list(obsy = obsy,
                   obsx = obsx,
                   K = K,
                   N = nobs)

NORM <-" model{
    # diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }

    # uniform prior for standard deviation
    tau <- pow(sigma, -2) # precision
    sigma ~ dunif(0, 100) # standard deviation

    # likelihood function
    for (i in 1:N){
        obsy[i] ~ dnorm(mu[i],tau)
        mu[i] <- eta[i]
        eta[i] <- beta[1]+beta[2]*obsx[i]
    }
}"

# initial value
inits <- function () {
  list(
    beta = rnorm(K, 0, 0.01))
}

# Parameters to display and save
params <- c("beta", "sigma")

normefit <- jags(data = model.data,
                 inits = inits,
                 parameters = params,
                 model = textConnection(NORM),
                 n.chains = 3,
                 n.iter = 10000,
                 n.thin = 1,
                 n.burnin = 5000)

print(normefit,intervals=c(0.025, 0.975), digits=3)
