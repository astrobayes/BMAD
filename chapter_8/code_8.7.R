# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.7 - Bayesian random intercept binary logistic model in R using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2, data = logitr)
K <- ncol(X)
re <- as.numeric(logitr$Groups)
Nre <- length(unique(logitr$Groups))

model.data <- list(
  Y = logitr$y,                       # Response
  X = X,                              # Covariates
  K = K,                              # Num. betas
  N = nrow(logitr),                   # Sample size
  re = logitr$Groups,                 # Random effects
  b0 = rep(0,K),
  B0 = diag(0.0001, K),
  a0 = rep(0,Nre),
  A0 = diag(Nre))

sink("GLMM.txt")

cat("
model {
    # Diffuse normal priors for regression parameters
    beta ~ dmnorm(b0[], B0[,])

    # Priors for random effect group
    a ~ dmnorm(a0, tau.re * A0[,])
    num ~ dnorm(0, 0.0016)
    denom ~ dnorm(0, 1)
    sigma.re <- abs(num / denom)
    tau.re <- 1 / (sigma.re * sigma.re)

    # Likelihood
    for (i in 1:N) {
        Y[i] ~ dbern(p[i])
        logit(p[i]) <- max(-20, min(20, eta[i]))
       eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
    }
}",fill = TRUE)

sink()

num ~ dnorm(0, 0.0016)
denom ~ dnorm(0, 1)

sigma.re <- abs(num / denom)
tau.re <- 1 / (sigma.re * sigma.re)

inits <- function () {
  list(beta = rnorm(K, 0, 0.01),
       a = rnorm(Nre, 0, 1),
       num = runif(1, 0, 25),
       denom = runif(1, 0, 1))}

params <- c("beta", "a", "sigma.re", "tau.re")

LRI0 <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model.file = "GLMM.txt",
             n.thin = 10,
             n.chains = 3,
             n.burnin = 5000,
             n.iter = 7000)

print(LRI0, intervals=c(0.025, 0.975), digits=3)