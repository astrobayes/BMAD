# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 6.24 - Zero Truncated Negative binomial with 
#             0-trick using JAGS - direct

require(MASS)
require(R2jags)
require(VGAM)

set.seed(123579)
nobs <- 1000

x1 <- rbinom(nobs,size=1,0.7)
x2 <- runif(nobs)

xb <- 1 + 2*x1 - 4*x2
exb <- exp(xb)
alpha = 5

rztnb <- function(n, mu, size){
  p <- runif(n, dzinegbin(0, munb=mu, size=size),1)
  ztnb <- qzinegbin(p, munb=mu, size=size)
  return(ztnb)
}

ztnby <- rztnb(nobs, exb, size=1/alpha)
ztnb.data <-data.frame(ztnby, x1, x2)

X <- model.matrix(~ x1 + x2, data = ztnb.data)
K <- ncol(X)

model.data <- list(Y = ztnb.data$ztnby,
                   X = X,
                   K = K,                        # number of betas
                   N = nobs,
                   Zeros = rep(0, nobs))         # sample size

ZTNB <- "
model{
    for (i in 1:K) {beta[i] ~ dnorm(0, 1e-4)}

    alpha ~ dgamma(1e-3,1e-3)

    # Likelihood with zero trick
    C <- 10000
    for (i in 1:N) {
        # Log likelihood function using zero trick:
        Zeros[i] ~ dpois(Zeros.mean[i])
        Zeros.mean[i] <- -L[i] + C
        l1[i] <- 1/alpha * log(u[i])
        l2[i] <- Y[i] * log(1 - u[i])
        l3[i] <- loggam(Y[i] + 1/alpha)
        l4[i] <- loggam(1/alpha)
        l5[i] <- loggam(Y[i] + 1)
        l6[i] <- log(1 - (1 + alpha * mu[i])^(-1/alpha))
        L[i] <- l1[i] + l2[i] + l3[i] - l4[i] - l5[i] - l6[i]
        u[i] <- 1/(1 + alpha * mu[i])
        log(mu[i]) <- inprod(X[i,], beta[])
    }
}"

inits <- function () {
     list(beta = rnorm(K, 0, 0.1))}

params <- c("beta","alpha")

ZTNB1 <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model = textConnection(ZTNB),
              n.thin = 1,
              n.chains = 3,
              n.burnin = 2500,
              n.iter = 5000)

print(ZTNB1, intervals=c(0.025, 0.975), digits=3)
