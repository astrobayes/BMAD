# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.1 - Random intercept Gaussian data generated in R

set.seed(1656)

N <- 4500                                                 # 20 groups, each with 200 observations
NGroups <- 20

x1 <- runif(N)
x2 <- runif(N)
Groups <- rep(1:20, each = 225)

a <- rnorm(NGroups, mean = 0, sd = 0.5)
print(a,2)


mu <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]
y <- rnorm(N, mean=mu, sd=2)
normr <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups]
)

# Code 8.2 - Random intercept normal model in R using JAGS

library(R2jags)

# Data
X <- model.matrix(~ x1 + x2, data = normr)
K <- ncol(X)

re <- as.numeric(normr$Groups)
Nre <- length

model.data <- list(
  Y = normr$y,                                            # response
  X = X,                                                  # covariates
  K = K,                                                  # number of betas
  N = nrow(normr),                                        # rows in model
  re = re,                                                # random effect
  b0 = rep(0,K),                                          # parameter priors with initial 0 value
  B0 = diag(0.0001, K),                                   # priors for V-C matrix
  a0 = rep(0,Nre),                                        # priors for scale parameters
  A0 = diag(Nre))                                         # hyperpriors for scale parameters


# Fit
sink("lmm.txt")

cat("
model {
    # Diffuse normal priors for regression parameters
    beta ~ dmnorm(b0[], B0[,])

    # Priors for random intercept groups
    a ~ dmnorm(a0, tau.plot * A0[,])

    # Priors for the two sigmas and taus
    tau.plot <- 1 / (sigma.plot * sigma.plot)
    tau.eps <- 1 / (sigma.eps * sigma.eps)
    
    sigma.plot ~ dunif(0.001, 10)
    sigma.eps ~ dunif(0.001, 10)

    # Likelihood
    for (i in 1:N) {
        Y[i] ~ dnorm(mu[i], tau.eps)
        mu[i] <- eta[i]
        eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
    }
}
",fill = TRUE)

sink()

inits <- function () {
  list(beta = rnorm(K, 0, 0.01),
       a = rnorm(Nre, 0, 1),
       sigma.eps = runif(1, 0.001, 10),
       sigma.plot = runif(1, 0.001, 10)
  )}

params <- c("beta","a", "sigma.plot", "sigma.eps")

NORM0 <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "lmm.txt",
              n.thin = 10,
              n.chains = 3,
              n.burnin = 6000,
              n.iter = 10000)

# Output
print(NORM0, intervals=c(0.025, 0.975), digits=3)
