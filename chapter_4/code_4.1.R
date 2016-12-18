# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 4.1 Normal linear model in R using JAGS


require(R2jags)

set.seed(1056)                           # set seed to replicate example
nobs = 5000                              # number of observations in model

x1 <- runif(nobs)                        # random uniform variable
beta0 = 2.0                              # intercept
beta1 = 3.0                              # slope or coefficient
xb <- beta0 + beta1 * x1                 # linear predictor, xb
y <- rnorm(nobs, xb, sd=1)               # create y as adjusted random normal variate

# Construct data dictionary
X <- model.matrix(~ 1 + x1)
K <- ncol(X)
model.data <- list(Y = y,               # Response variable
                   X = X,               # Predictors
                   K = K,               # Number of predictors including the intercept
                   N = nobs             # Sample size
)

# Model set up
NORM <- "model{
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

# Initial values
inits <- function () {
    list(beta = rnorm(K, 0, 0.01))
}

# Parameters to be displayed
    params <- c("beta", "sigma")

# MCMC
normfit <- jags(data = model.data,
                inits = inits,
                parameters = params,
                model = textConnection(NORM),
                n.chains = 3,
                n.iter = 15000,
                n.thin = 1,
                n.burnin = 10000)

print(normfit, intervals = c(0.025, 0.975), digits = 2)


# Plot the chains to assess mixing
source("CH-Figures.R")
out <- normfit$BUGSoutput
MyBUGSChains(out,c(uNames("beta",K),"sigma"))

# Display the histograms
out <- normfit$BUGSoutput
MyBUGSHist(out,c(uNames("beta",K),"sigma"))
