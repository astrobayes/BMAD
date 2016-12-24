# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.27 - Real data for a grouped binomial model

library(R2jags)

y <- c(6,11,9,13,17,21,8,10,15,19,7,12)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48)
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)
x2 <- c(1,1,0,0,1,1,0,0,1,1,0,0)
x3 <- c(1,0,1,0,1,0,1,0,1,0,1,0)

bindata1 <-data.frame(y,m,x1,x2,x3)

X <- model.matrix(~ x1 + x2 + x3, data = bindata1)
K <- ncol(X)

model.data <- list(Y = bindata1$y,
                    N = nrow(bindata1),
                    X = X,
                    K = K,
                    m = bindata1$m)

sink("GLOGIT.txt")

cat("
model{
# Priors
# Diffuse normal priors betas
for (i in 1:K) {beta[i] ~ dnorm(0, 0.0001)}
    
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