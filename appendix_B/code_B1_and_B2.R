# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code B.1 - Data for Poisson with offset

library(MASS)

x1 <- runif(5000)
x2 <- runif(5000)

m <- rep(1:5, each=1000, times=1)*100                         # creates offset as defined
logm <- log(m)                                                # log the offset

xb <- 2 + .75*x1 -1.25*x2 + logm                              # linear predictor w offset
exb <- exp(xb)

py <- rpois(5000, exb)
pdata <- data.frame(py, x1, x2, m)

# Code B.2 - Bayesian Poisson with offset
require(R2jags)

X <- model.matrix(~ x1 + x2, data = pdata)
K <- ncol(X)

model.data <- list(Y = pdata$py,
                   N = nrow(pdata),
                   X = X,
                   K = K,
                   m = pdata$m)                                # list offset

sink("PRATE.txt")

cat("
model{
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Likelihood
    for (i in 1:N){
        Y[i] ~ dpois(mu[i])
        log(mu[i]) <- inprod(beta[], X[i,]) + log(m[i]) # offset added
    }
}
",fill = TRUE)

sink()

# Initial parameter values
inits <- function () {
  list(
    beta = rnorm(K, 0, 0.1)
  ) }

params <- c("beta") # posterior parameters to display

poisR <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "PRATE.txt",
              n.thin = 3,
              n.chains = 3,
              n.burnin = 3000,
              n.iter = 5000)

print(poisR, intervals=c(0.025, 0.975), digits=3)

poio <-glm(py ~ x1 + x2 + offset(log(m)), family=poisson, data=pdata)

summary(poio)