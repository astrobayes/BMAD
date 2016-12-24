# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.31 - Simulated beta–binomial data in R

# Required packages
require(R2jags)
require(boot)
require(VGAM)

# Simulation
set.seed(33559)
nobs = 2500

m = 1 + rpois(nobs,5)
x1 = runif(nobs)
beta1 <- -2
beta2 <- -1.5
eta <- beta1 + beta2 * x1
sigma <- 20

p <- inv.logit(eta)

shape1 = sigma*p
shape2 = sigma*(1 - p)

y <- rbetabinom.ab(n=nobs, size=m, shape1=shape1, shape2=shape2)

bindata = data.frame(y=y,m=m,x1)

# Code 5.32 - Beta–binomial synthetic model in R using JAGS

X <- model.matrix(~ x1, data = bindata)
K <- ncol(X)

model.data <- list(Y = bindata$y,
                   N = nrow(bindata),
                   X = X,
                   K = K,
                   m = m )

sink("GLOGIT.txt")

cat("
model{
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Prior for theta
    sigma ~ dgamma(0.01,0.01)

    for (i in 1:N){
        Y[i] ~ dbin(p[i],m[i])
        p[i] ~ dbeta(shape1[i],shape2[i])

        shape1[i] <- sigma*pi[i]
        shape2[i] <- sigma*(1-pi[i])
        logit(pi[i]) <- eta[i]
        eta[i] <- inprod(beta[],X[i,])
    }
}
",fill = TRUE)

sink()

inits <- function () {list(beta = rnorm(K, 0, 0.1)) }

params <- c("beta", "sigma")

BBIN <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model.file = "GLOGIT.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 3000,
             n.iter = 5000)

print(BBIN, intervals=c(0.025, 0.975), digits=3)