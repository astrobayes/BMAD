# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.5 - Multivariate normal model in R using JAGS for accessing the relationship
#             between period, luminosity, and color in early-type contact binaries

library(R2jags)

# Data
PLC <- read.csv("https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p3/PLC.csv", header = T)

# Prepare data for JAGS
nobs = nrow(PLC)                        # number of data points
x1 <- PLC$logP                          # log period
x2 <- PLC$V_I                           # V-I color
y <- PLC$M_V                            # V magnitude
type <- as.numeric(PLC$type)            # type NC/GC
X <- model.matrix(~ 1 + x1+x2)          # covariate matrix
K <- ncol(X)                            # number of covariates per type

jags_data <- list(Y = y,
                  X = X,
                  K = K,
                  type = type,
                  N = nobs)

# Fit
NORM <-"model{
    # Shared hyperprior
    tau0 ~ dgamma(0.001,0.001)
    mu0 ~ dnorm(0,1e-3)

    # Diffuse normal priors for predictors
    for(j in 1:2){
        for (i in 1:K) {
            beta[i,j] ~ dnorm(mu0, tau0)
        }
    }

    # Uniform prior for standard deviation
    for(i in 1:2) {
        tau[i] <- pow(sigma[i], -2)                   #precision
        sigma[i] ~ dgamma(1e-3, 1e-3)                 #standard deviation
    }

    # Likelihood function
    for (i in 1:N){
        Y[i]~dnorm(mu[i],tau[type[i]])
        mu[i] <- eta[i]
        eta[i] <- beta[1, type[i]] * X[i, 1] + beta[2, type[i]] * X[i, 2] +
        beta[3, type[i]] * X[i, 3]
    }
}"

# Determine initial values
inits <- function () {
  list(beta = matrix(rnorm(6,0, 0.01),ncol=2))
}

# Identify parameters
params <- c("beta", "sigma")

# Fit
jagsfit <- jags(data = jags_data,
                inits = inits,
                parameters = params,
                model = textConnection(NORM),
                n.chains = 3,
                n.iter = 5000,
                n.thin = 1,
                n.burnin = 2500)

## Output
print(jagsfit,justify = "left", digits=2)