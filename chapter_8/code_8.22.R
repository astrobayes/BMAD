# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 8.19

library(MASS)

N <- 2000                          # 10 groups, each with 200 observations
NGroups <- 10

x1 <- runif(N)
x2 <- runif(N)

Groups <- rep(1:10, each = 200)
a <- rnorm(NGroups, mean = 0, sd = 0.5)
eta <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]

mu <- exp(eta)
y <- rnegbin(mu, theta=2)

nbri <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups]
)


# Code 8.22 - Bayesian random intercept negative binomial in R using JAGS.

library(R2jags)

X <- model.matrix(~ x1 + x2, data = nbri)
K <- ncol(X)
Nre <- length(unique(nbri$Groups))

model.data <- list(
  Y = nbri$y,                    # response
  X = X,                        # covariates
  K = K,                        # num. betas
  N = nrow(nbri),                # sample size
  re = nbri$Groups,              # random effects
  b0 = rep(0,K),
  B0 = diag(0.0001, K),
  a0 = rep(0,Nre),
  A0 = diag(Nre))

sink("GLMM_NB.txt")

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

    # Prior for alpha
    numS ~ dnorm(0, 0.0016)
    denomS ~ dnorm(0, 1)
    alpha <- abs(numS / denomS)

    # Likelihood
    for (i in 1:N) {
        Y[i] ~ dnegbin(p[i], 1/alpha)
        p[i] <- 1 /( 1 + alpha * mu[i])
        log(mu[i]) <- eta[i]
        eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
        }
    }
    ",fill = TRUE)

sink()

# Define initial values
inits <- function () {
    list(beta = rnorm(K, 0, 0.1),
         a = rnorm(Nre, 0, 1),
         num = rnorm(1, 0, 25),
         denom = rnorm(1, 0, 1),
         numS = rnorm(1, 0, 25) ,
         denomS = rnorm(1, 0, 1))}

# Identify parameters
params <- c("beta", "a", "sigma.re", "tau.re", "alpha")

NBI <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model.file = "GLMM_NB.txt",
            n.thin = 10,
            n.chains = 3,
            n.burnin = 4000,
            n.iter = 5000)

print(NBI, intervals=c(0.025, 0.975), digits=3)