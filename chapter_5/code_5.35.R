# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 5.33

y <- c(6,11,9,13,17,21,8,10,15,19,7,12)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48)
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)
x2 <- c(1,1,0,0,1,1,0,0,1,1,0,0)
x3 <- c(1,0,1,0,1,0,1,0,1,0,1,0)

bindata <-data.frame(y,m,x1,x2,x3)

X <- model.matrix(~ x1 + x2 + x3, data = bindata)

K <- ncol(X)

# Code 5.35 - Beta–binomial model with inverse link in R using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2 + x3, data = bindata)
K <- ncol(X)

glogit.data <- list(Y = bindata$y,
                    N = nrow(bindata),
                    X = X,
                    K = K,
                    m = m)

sink("BBI.txt")

cat("
    model{
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}

    # Prior for sigma
    sigma ~ dgamma(0.01,0.01)

    for (i in 1:N){
        Y[i] ~ dbin(p[i],m[i])
        p[i] ~ dbeta(shape1[i],shape2[i])

        shape1[i]<-sigma*pi[i]
        shape2[i]<-sigma*(1-pi[i])

        logit(pi[i]) <- eta[i]
        eta[i]<-inprod(beta[],X[i,])
        }
    }",fill = TRUE)

sink()

# Determine initial values
inits <- function () {list(beta = rnorm(K, 0, 0.1))}

# Identify parameters
params <- c("beta", "sigma")

BB1 <- jags(data = glogit.data,
            inits = inits,
            parameters = params,
            model.file = "BBI.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 6000,
            n.iter = 10000)

print(BB1, intervals=c(0.025, 0.975), digits=3)
