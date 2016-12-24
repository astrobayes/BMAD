# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

#  5.15 - Synthetic data from logistic model in R

set.seed(13979)
nobs <- 5000

x1 <- rbinom(nobs, size = 1, 0.6)
x2 <- runif(nobs)
xb <- 2 + 0.75*x1 - 5*x2

exb <- 1/(1+exp(-xb))
by <- rbinom(nobs, size = 1, prob = exb)

logitmod <- data.frame(by, x1, x2)

# Code 5.17 Logistic model in R using JAGS

attach(logitmod)
require(R2jags)

X <- model.matrix(~ x1 + x2,
                   data = logitmod)

K <- ncol(X)
model.data <- list(Y = logitmod$by,
                   N = nrow(logitmod),
                   X = X,
                   K = K,
                   LogN = log(nrow(logitmod)),
                   b0 = rep(0, K),
                   B0 = diag(0.00001, K)
)

sink("LOGIT.txt")

cat("
model{
    # Priors
    beta ~ dmnorm(b0[], B0[,])

    # Likelihood
    for (i in 1:N){
        Y[i] ~ dbern(p[i])
        logit(p[i]) <- eta[i]
       
        # logit(p[i]) <- max(-20,min(20,eta[i])) used to avoid numerical instabilities
        # p[i] <- 1/(1+exp(-eta[i])) can use for logit(p[i]) above
        
        eta[i] <- inprod(beta[], X[i,])
        LLi[i] <- Y[i] * log(p[i]) + (1 - Y[i]) * log(1 - p[i])
    }

    LogL <- sum(LLi[1:N])
    AIC <- -2 * LogL + 2 * K
    BIC <- -2 * LogL + LogN * K
}
",fill = TRUE)

sink()

# Initial parameter values
inits <- function () {
    list(beta = rnorm(K, 0, 0.1)) }

params <- c("beta", "LogL", "AIC", "BIC")

LOGT <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model.file = "LOGIT.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 5000,
             n.iter = 10000)

print(LOGT, intervals=c(0.025, 0.975), digits=3)

# plot
source("CH-Figures.R")

out <- LOGT$BUGSoutput
MyBUGSHist(out,c(uNames("beta",K),"AIC","BIC","LogL"))
MyBUGSChains(out,c(uNames("beta",K),"AIC","BIC","LogL"))