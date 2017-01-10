# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

set.seed(1656)

# Code 8.10 - Random intercept Poisson data in R

N <- 2000                        # 10 groups, each with 200 observations
NGroups <- 10

x1 <- runif(N)
x2 <- runif(N)

Groups <- rep(1:10, each = 200)
a <- rnorm(NGroups, mean = 0, sd = 0.5)

eta <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]
mu <- exp(eta)
y <- rpois(N, lambda = mu)

poir <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups])

# Code 8.11 - Bayesian random intercept Poisson model in R using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2, data=poir)
K <- ncol(X)

re <- as.numeric(poir$Groups)
Nre <- length(unique(poir$Groups))

model.data <- list(
  Y = poir$y,                      # response
  X = X,                           # covariates
  N = nrow(poir),                  # sample size
  re = poir$Groups,                # random effects
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
        Y[i] ~ dpois(mu[i])
        log(mu[i])<- eta[i]
        eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
        }
    }
    ",fill = TRUE)

sink()

inits <- function () {
     list(beta = rnorm(K, 0, 0.01),
          a = rnorm(Nre, 0, 1),
          num = runif(1, 0, 25),
          denom = runif(1, 0, 1))}

# Identify parameters
params <- c("beta", "a", "sigma.re", "tau.re")

# Run MCMC
PRI <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model.file = "GLMM.txt",
            n.thin = 10,
            n.chains = 3,
            n.burnin = 4000,
            n.iter = 5000)

print(PRI, intervals=c(0.025, 0.975), digits=3)