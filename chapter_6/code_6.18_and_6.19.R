# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.18 - Synthetic data for generalized Poisson

require(MASS)
require(R2jags)

source("https://raw.githubusercontent.com/astrobayes/BMAD/master/auxiliar_functions/rgp.R")    # or wherever you stored the file

set.seed(160)
nobs <- 1000

x1 <- runif(nobs)
xb <- 1 + 3.5*x1
exb <- exp(xb)

delta <- -0.3
gpy <- c()

for(i in 1:nobs){
  gpy[i] <- rgp(1, mu=(1-delta)*exb[i],delta = delta)
}

gpdata <- data.frame(gpy, x1)

# Code 6.19 - Bayesian generalized Poisson using JAGS

X <- model.matrix(~ x1, data = gpdata)
K <- ncol(X)

model.data <- list(Y = gpdata$gpy,                      # response
                   X = X,                               # covariates
                   N = nrow(gpdata),                    # sample size
                   K = K,                               # number of betas
                   Zeros = rep(0, nrow(gpdata)))

sink("GP1reg.txt")

cat("
model{
    # Priors beta
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Prior for delta parameter of GP distribution
    delta ~ dunif(-1, 1)

    C <- 10000
    for (i in 1:N){
        Zeros[i] ~ dpois(Zeros.mean[i])
        Zeros.mean[i] <- -L[i] + C
        l1[i] <- log(mu[i])
        l2[i] <- (Y[i] - 1) * log(mu[i] + delta * Y[i])
        l3[i] <- -mu[i] - delta * Y[i]
        l4[i] <- -loggam(Y[i] + 1)
        L[i] <- l1[i] + l2[i] + l3[i] + l4[i]
        mu[i] <- (1 - delta)*exp(eta[i])
        eta[i] <- inprod(beta[], X[i,])
    }

    # Discrepancy measures: mean, variance, Pearson residuals
    for (i in 1:N){
        ExpY[i] <- mu[i] / (1 - delta)
        VarY[i] <- mu[i] / ((1 - delta)^3)
        Pres[i] <- (Y[i] - ExpY[i]) / sqrt(VarY[i])
    } }
    ",fill = TRUE)

sink()

inits <- function () {
    list(beta = rnorm(ncol(X), 0, 0.1),
        delta = 0)}

params <- c("beta", "delta")

GP1 <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = "GP1reg.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 4000,
            n.iter = 5000)

print(GP1, intervals=c(0.025, 0.975), digits=3)
