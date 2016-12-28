# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.1 - Bayesian zero-inflated Poisson model in R using JAGS

require(MASS)
require(R2jags)
require(VGAM)

set.seed(141)
nobs <- 1000

x1 <- runif(nobs)
xb <- 1 + 2.0*x1
xc <- 2 - 5.0*x1
exb <- exp(xb)
exc <- 1/(1+exp(-xc))
zipy <- rzipois(n=nobs, lambda=exb, pstr0=exc)
zipdata <- data.frame(zipy,x1)

Xc <- model.matrix(~ 1 + x1, data=zipdata)
Xb <- model.matrix(~ 1 + x1, data=zipdata)
Kc <- ncol(Xc)
Kb <- ncol(Xb)

model.data <- list(Y = zipdata$zipy,                      # response
                   Xc = Xc,                               # covariates
                   Kc = Kc,                               # number of betas
                   Xb = Xb,                               # covariates
                   Kb = Kb,                               # number of gammas
                   N = nobs)

ZIPOIS<-"model{
    # Priors - count and binary components
    for (i in 1:Kc) { beta[i] ~ dnorm(0, 0.0001)}
    for (i in 1:Kb) { gamma[i] ~ dnorm(0, 0.0001)}

    # Likelihood
    for (i in 1:N) {
        W[i] ~ dbern(1 - Pi[i])
        Y[i] ~ dpois(W[i] * mu[i])
        log(mu[i]) <- inprod(beta[], Xc[i,])
        logit(Pi[i]) <- inprod(gamma[], Xb[i,])
    }
}"

W <- zipdata$zipy
W[zipdata$zipy > 0] <- 1

inits <- function() {
      list(beta = rnorm(Kc, 0, 0.1),
           gamma = rnorm(Kb, 0, 0.1),
           W = W)}

params <- c("beta", "gamma")

ZIP <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = textConnection(ZIPOIS),
            n.thin = 1,
            n.chains = 3,
            n.burnin = 4000,
            n.iter = 5000)

print(ZIP, intervals = c(0.025, 0.975), digits=3)
