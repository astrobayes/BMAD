# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.25 - Synthetic data from a binomial model in R

set.seed(33559)
nobs = 2000

m = 1 + rpois(nobs,5)
x1 = runif(nobs)
x2 = runif(nobs)

xb <- -2 - 1.5 * x1 + 3 * x2

exb <- exp(xb)
p <- exb/(1 + exb)                      # prob of p=0
y <- rbinom(nobs,prob=p,size=m)

bindata=data.frame(y=y,m=m,x1,x2)

# Code 5.26 - Binomial model in R using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2, data = bindata)
K <- ncol(X)

model.data <- list(Y = bindata$y,
                   N = nrow(bindata),
                   X = X,
                   K = K,
                   m = bindata$m)

sink("GLOGIT.txt")

cat("
model{
    # Priors
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Likelihood
    for (i in 1:N){
    Y[i] ~ dbin(p[i],m[i])
    logit(p[i]) <- eta[i]
    eta[i] <- inprod(beta[], X[i,])
    }
}
",fill = TRUE)

sink()

inits <- function () {list(beta = rnorm(K, 0, 0.1)) }

params <- c("beta")

BINL <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model.file = "GLOGIT.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 3000,
             n.iter = 5000)

print(BINL, intervals=c(0.025, 0.975), digits=3)