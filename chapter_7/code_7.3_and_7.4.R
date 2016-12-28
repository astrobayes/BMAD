# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.3 - Zero-inflated negative binomial synthetic data in R

require(MASS)
require(R2jags)
require(VGAM)

set.seed(141)
nobs <- 1000

x1 <- runif(nobs)
x2 <- rbinom(nobs, size=1, 0.6)

xb <- 1 + 2.0*x1 + 1.5*x2
xc <- 2 - 5.0*x1 + 3*x2

exb <- exp(xb)
exc <- 1/(1 + exp(-xc))
alpha <- 2

zinby <- rzinegbin(n=nobs, munb = exb, size=1/alpha, pstr0=exc)
zinbdata <- data.frame(zinby,x1,x2)


# Code 7.4 - Bayesian zero-inflated negative binomial model 
#            using JAGS


Xc <- model.matrix(~ 1 + x1+x2, data=zinbdata)
Xb <- model.matrix(~ 1 + x1+x2, data=zinbdata)

Kc <- ncol(Xc)
Kb <- ncol(Xb)
model.data <- list(Y = zinbdata$zinby,              # response
                   Xc = Xc,                         # covariates
                   Kc = Kc,                         # number of betas
                   Xb = Xb,                         # covariates
                   Kb = Kb,                         # number of gammas
                   N = nrow(zinbdata))

ZINB<-"model{
    # Priors - count and binary components
    for (i in 1:Kc) { beta[i] ~ dnorm(0, 0.0001)}
    for (i in 1:Kb) { gamma[i] ~ dnorm(0, 0.0001)}

    alpha ~ dunif(0.001, 5) 

    # Likelihood
    for (i in 1:N) {
        W[i] ~ dbern(1 - Pi[i])
        Y[i] ~ dnegbin(p[i], 1/alpha)
        p[i] <- 1/(1 + alpha * W[i]*mu[i])
        log(mu[i]) <- inprod(beta[], Xc[i,])
        logit(Pi[i]) <- inprod(gamma[], Xb[i,])
    } }"


W <- zinbdata$zinby
W[zinbdata$zinby > 0] <- 1

inits <- function () {
    list(beta = rnorm(Kc, 0, 0.1),
         gamma = rnorm(Kb, 0, 0.1),
         W = W)}

params <- c("beta", "gamma","alpha")

ZINB <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model = textConnection(ZINB),
             n.thin = 1,
             n.chains = 3,
             n.burnin = 4000,
             n.iter = 5000)

print(ZINB, intervals=c(0.025, 0.975), digits=3)

# Figures for parameter trace plots and histogramss
source("CH-Figures.R")

out <- ZINB$BUGSoutput
MyBUGSHist(out,c(uNames("beta",K),"sigma"))
MyBUGSChains(out,c(uNames("beta",K),"sigma"))
