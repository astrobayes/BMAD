# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

set.seed(33559)

# Sample size
nobs <- 750

# Generate predictors, design matrix
x1 <- runif(nobs,0,4)
xc <- -1 + 0.75*x1
exc <- exp(xc)

phi <- 0.066
r <- 1/phi
y <- rgamma(nobs,shape=r, rate=r/exc)
LG <- data.frame(y, x1)

# Construct filter
xb <- -2 + 1.5*x1
pi <- 1/(1+exp(-(xb)))
bern <- rbinom(nobs,size=1, prob=pi)

# Add structural zeros
LG$y <- LG$y*bern

# Code 7.11 - Bayesian log-gamma–logit hurdle model in R using JAGS
library(R2jags)

Xc <- model.matrix(~ 1 + x1, data=LG)
Xb <- model.matrix(~ 1 + x1, data=LG)
Kc <- ncol(Xc)
Kb <- ncol(Xb)

model.data <- list(
  Y = LG$y,                              # response
  Xc = Xc,                               # covariates from gamma component
  Xb = Xb,                               # covariates from binary component
  Kc = Kc,                               # number of betas
  Kb = Kb,                               # number of gammas
  N = nrow(LG),                          # sample size
  Zeros = rep(0, nrow(LG)))

load.module('glm')

sink("ZAGGLM.txt")

cat("
model{
    # Priors for both beta and gamma components
    for (i in 1:Kc) {beta[i] ~ dnorm(0, 0.0001)}
    for (i in 1:Kb) {gamma[i] ~ dnorm(0, 0.0001)}

    # Prior for scale parameter, r
    r ~ dgamma(1e-2, 1e-2)

    # Likelihood using the zero trick
    C <- 10000

    for (i in 1:N) {
        Zeros[i] ~ dpois(-ll[i] + C)

        # gamma log-likelihood
        lg1[i] <- - loggam(r) + r * log(r / mu[i])
        lg2[i] <- (r - 1) * log(Y[i]) - (Y[i] * r) / mu[i]
        LG[i] <- lg1[i] + lg2[i]
        z[i] <- step(Y[i] - 0.0001)
        l1[i] <- (1 - z[i]) * log(1 - Pi[i])
        l2[i] <- z[i] * ( log(Pi[i]) +LG[i])
        ll[i] <- l1[i] + l2[i]
        log(mu[i]) <- inprod(beta[], Xc[i,])
        logit(Pi[i]) <- inprod(gamma[], Xb[i,])
    }

    phi <- 1/r
    }", fill = TRUE)

sink()

# Initial parameter values
inits <- function () {
  list(beta = rnorm(Kc, 0, 0.1),
       gamma = rnorm(Kb, 0, 0.1),
       r = runif(1, 0,100) )}

# Parameter values to be displayed in output
params <- c("beta", "gamma", "phi")

# MCMC sampling
ZAG <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = "ZAGGLM.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 2500,
            n.iter = 5000)

# Model results
print(ZAG, intervals = c(0.025, 0.975), digits=3)
