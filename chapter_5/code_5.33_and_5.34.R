# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.33 - Explicitly given beta–binomial data in R

y <- c(6,11,9,13,17,21,8,10,15,19,7,12)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48)
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)
x2 <- c(1,1,0,0,1,1,0,0,1,1,0,0)
x3 <- c(1,0,1,0,1,0,1,0,1,0,1,0)

bindata <-data.frame(y,m,x1,x2,x3)

# Code 5.34 Beta–binomial model (in R using JAGS) 
#  for explicitly given data and the zero trick

library(R2jags)

X <- model.matrix(~ x1 + x2 + x3, data = bindata)

K <- ncol(X)

model.data <- list(Y = bindata$y,
                   N = nrow(bindata),
                   X =X,
                   K =K,
                   m = m,
                   Zeros = rep(0, nrow(bindata))
)

sink("BBL.txt")

cat("
model{
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Prior for sigma
    sigma ~ dunif(0, 100)

    C <- 10000
    for (i in 1:N){
        eros[i] ~ dpois(Zeros.mean[i])
        Zeros.mean[i] <- -LL[i] + C

        #mu[i] <- 1/(1+exp(-eta[i])) # can use for logit(mu[i]) below
        logit(mu[i]) <- max(-20, min(20, eta[i]))
        L1[i] <- loggam(m[i]+1) - loggam(Y[i]+1) - loggam(m[i]-Y[i]+1)
        L2[i] <- loggam(1/sigma) + loggam(Y[i]+mu[i]/sigma)
        L3[i] <- loggam(m[i] - Y[i]+(1-mu[i])/sigma) - loggam(m[i]+1/sigma)
        L4[i] <- loggam(mu[i]/sigma) + loggam((1-mu[i])/sigma)
        LL[i] <- L1[i] + L2[i] + L3[i] - L4[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}
",fill = TRUE)

sink()

inits <- function () {list(beta = rnorm(K, 0, 0.1)) }

params <- c("beta", "sigma")

BBIN0 <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "BBL.txt",
              n.thin = 3,
              n.chains = 3,
              n.burnin = 10000,
              n.iter = 15000)

print(BBIN0, intervals=c(0.025, 0.975), digits=3)