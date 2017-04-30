# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.13 - Bayesian lognormal–logit hurdle using JAGS

library(R2jags)
set.seed(33559)

# Sample size
nobs <- 3000

# Generate predictors, design matrix
x1 <- runif(nobs,0,2.5)
xc <- 0.6 + 1.25*x1
y <- rlnorm(nobs, xc, sdlog=0.4)

lndata <- data.frame(y, x1)

# Construct filter
xb <- -3 + 4.5*x1
pi <- 1/(1+exp(-(xb)))
bern <- rbinom(nobs,size=1, prob=pi)

# Add structural zeros
lndata$y <- lndata$y*bern
Xc <- model.matrix(~ 1 + x1,data = lndata)
Xb <- model.matrix(~ 1 + x1,data = lndata)
Kc <- ncol(Xc)
Kb <- ncol(Xb)

JAGS.data <- list(
  Y = lndata$y,                             # response
  Xc = Xc,                                  # covariates
  Xb = Xb,                                  # covariates
  Kc = Kc,                                  # number of betas
  Kb = Kb,                                  # number of gammas
  N = nrow(lndata),                         # sample size
  Zeros = rep(0, nrow(lndata)))

load.module('glm')

sink("ZALN.txt")

cat("
model{
    # Priors for both beta and gamma components
    for (i in 1:Kc) {beta[i] ~ dnorm(0, 0.0001)}
    for (i in 1:Kb) {gamma[i] ~ dnorm(0, 0.0001)}

    # Prior for sigma
    sigmaLN ~ dgamma(1e-3, 1e-3)

    # Likelihood using the zero trick
    C <- 10000
    for (i in 1:N) {
        Zeros[i] ~ dpois(-ll[i] + C)

        # LN log-likelihood
        ln1[i] <- -(log(Y[i]) + log(sigmaLN) + log(sqrt(2 * sigmaLN)))
        ln2[i] <- -0.5 * pow((log(Y[i]) - mu[i]),2)/(sigmaLN * sigmaLN)
        LN[i] <- ln1[i] + ln2[i]

        z[i] <- step(Y[i] - 1e-5)
        l1[i] <- (1 - z[i]) * log(1 - Pi[i])
        l2[i] <- z[i] * ( log(Pi[i]) + LN[i])
        ll[i] <- l1[i] + l2[i]
        mu[i] <- inprod(beta[], Xc[i,])
        logit(Pi[i]) <- inprod(gamma[], Xb[i,])
        }
    }", fill = TRUE)

sink()

# Initial parameter values
inits <- function () {
  list(beta = rnorm(Kc, 0, 0.1),
       gamma = rnorm(Kb, 0, 0.1),
       sigmaLN = runif(1, 0, 10))}

# Parameter values to be displayed in output
params <- c("beta", "gamma", "sigmaLN")

# MCMC sampling
ZALN <- jags(data = JAGS.data,
             inits = inits,
             parameters = params,
             model = "ZALN.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 2500,
             n.iter = 5000)

# Model results
print(ZALN, intervals = c(0.025, 0.975), digits=3)
