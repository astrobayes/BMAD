# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# from Code 6.9 - Synthetic negative binomial data and model in R

library(MASS)

set.seed(141)
nobs <- 2500

x1 <- rbinom(nobs,size=1, prob=0.6)
x2 <- runif(nobs)
xb <- 1 + 2.0*x1 - 1.5*x2
a <- 3.3

theta <- 0.303                                 # 1/a
exb <- exp(xb)

nby <- rnegbin(n=nobs, mu=exb, theta=theta)
negbml <- data.frame(nby, x1, x2)

# Code 6.15 Change to indirect parameterization in R

library(R2jags)

X <- model.matrix(~ x1 + x2, data=negbml)

K <- ncol(X)
N <- nrow(negbml)

model.data <- list(
  Y = negbml$nby,
  N =N,
  K =K,
  X =X,
  Zeros = rep(0, N)
)

sink("NB0.txt")

cat("
model{
    # Priors regression parameters
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Prior for alpha
    numS ~ dnorm(0, 0.0016)
    denomS ~ dnorm(0, 1)
    alpha <- abs(numS / denomS)

    C <- 10000    
    for (i in 1:N) {
        # Log-likelihood function using zero trick:
        Zeros[i] ~ dpois(Zeros.mean[i])
        Zeros.mean[i] <- -L[i] + C
        l1[i] <- alpha * log(u[i])
        l2[i] <- Y[i] * log(1 - u[i])
        l3[i] <- loggam(Y[i] + alpha)
        l4[i] <- loggam(alpha)
        l5[i] <- loggam(Y[i] + 1)
        L[i] <- l1[i] + l2[i] + l3[i] - l4[i] - l5[i]
        u[i] <- alpha / (alpha + mu[i])
        log(mu[i]) <- max(-20, min(20, eta[i]))
        eta[i] <- inprod(X[i,], beta[])
        }
    }
    ",fill = TRUE)

sink()

inits1 <- function () {
    list(beta = rnorm(K, 0, 0.1),
         numS = rnorm(1, 0, 25) ,
         denomS = rnorm(1, 0, 1)
    ) }

params1 <- c("beta", "alpha")

NB01 <- jags(data = model.data,
             inits = inits1,
             parameters = params1,
             model = "NB0.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 3000,
             n.iter = 5000)

print(NB01, intervals=c(0.025, 0.975), digits=3)
