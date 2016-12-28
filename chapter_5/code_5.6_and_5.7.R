# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.6  - Log-gamma synthetic data generated in R

set.seed(33559)
nobs <- 1000
r <- 20                                             # shape
beta1 <- 1
beta2 <- 0.66
beta3 <- -1.25
x1 <- runif(nobs)
x2 <- runif(nobs)
xb <- beta1 + beta2*x1+beta3*x2
exb <- exp(xb)
py <- rgamma(nobs,shape = r, rate= r/exb)
LG <- data.frame(py, x1, x2)


# Code 5.7 - Log-gamma model in R using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2, data =LG)
K <- ncol(X)                                        # number of columns
model.data <- list(Y = LG$py,                       # response
                   X = X,                           # covariates
                   N = nrow(LG),                    # sample size
                   b0 = rep(0,K),
                   B0 = diag(0.0001, K))

sink("LGAMMA.txt")

cat("
model{
    # Diffuse priors for model betas
    beta ~ dmnorm(b0[], B0[,])

    # Diffuse prior for shape parameter
    r ~ dgamma(0.01, 0.01)
    
    # Likelihood
    C <- 10000
    for (i in 1:N){
        Y[i] ~ dgamma(r, lambda[i])
        lambda[i] <- r / mu[i]
        log(mu[i]) <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}
",fill = TRUE)

sink()

inits <- function () {
  list(
    beta = rnorm(K,0,0.01),
    r = 1 )
}

params <- c("beta", "r")

# JAGS MCMC
LGAM <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model.file = "LGAMMA.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 3000,
             n.iter = 5000)

print(LGAM, intervals=c(0.025, 0.975), digits=3)
