# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 6.9 
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

# Code 6.13 Negative binomial: direct parameterization 
#                              using JAGS and dnegbin

library(R2jags)

X <- model.matrix(~ x1 + x2, data=negbml)
K <- ncol(X)

model.data <- list(
  Y = negbml$nby,
  X = X,
  K = K,
  N = nrow(negbml))

# Model
sink("NBDGLM.txt")

cat("
model{
# Priors for coefficients
for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

# Prior for alpha
alpha ~ dunif(0.001, 5)

# Likelihood function
for (i in 1:N){
    Y[i] ~ dnegbin(p[i], 1/alpha)        # for indirect, (p[i], alpha)
    p[i] <- 1/(1 + alpha*mu[i])
    log(mu[i]) <- eta[i]
    eta[i] <- inprod(beta[], X[i,])
    }
}
",fill = TRUE)

sink()

inits <- function () {
  list(
    beta = rnorm(K, 0, 0.1),              # regression parameters
    alpha = runif(0.00, 5)
  )
}

params <- c("beta", "alpha")

NB3 <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = "NBDGLM.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 3000,
            n.iter = 5000)

print(NB3, intervals=c(0.025, 0.975), digits=3)

# plot
source("CH-Figures.R")

out <- NB3$BUGSoutput
MyBUGSChains(out,c(uNames("beta",K),"alpha"))

out <- NB3$BUGSoutput
MyBUGSHist(out,c(uNames("beta",K),"alpha"))
