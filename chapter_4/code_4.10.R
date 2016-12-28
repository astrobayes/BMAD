# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 4.10 - Normal linear model in R using JAGS and including errors in variables


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

K <- 2

model.data <- list(obsy = obsy,
                   obsx = obsx,
                   K = K,
                   errx = errx,
                   erry = erry,
                   N = nobs)

NORM_err <-" model{
    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 1e-3) }

    # Uniform prior for standard deviation
    tauy <- pow(sigma, -2)                               # precision
    sigma ~ dunif(0, 100)                                # diffuse prior for standard deviation

    # Diffuse normal priors for true x
    for (i in 1:N){
        x[i] ~ dnorm(0,1e-3)
    }

    # Likelihood
    for (i in 1:N){
        obsy[i] ~ dnorm(y[i],pow(erry[i],-2))
        y[i] ~ dnorm(mu[i],tauy)
        obsx[i] ~ dnorm(x[i],pow(errx[i],-2))
        mu[i] <- beta[1]+beta[2]*x[i]
    }
}"

# Initial values
inits <- function () {
    list(beta = rnorm(K, 0, 0.01))
}

# Parameter to display and save
params <- c("beta", "sigma")

evfit <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model = textConnection(NORM_err),
              n.chains = 3,
              n.iter = 5000,
              n.thin = 1,
              n.burnin = 2500)

print(evfit,intervals=c(0.025, 0.975), digits=3)
