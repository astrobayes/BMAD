# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

#  from Code 6.2 Synthetic Poisson data and model in R: 
#               binary and continuous predictors

set.seed(18472)
nobs <- 750

x1_2 <- rbinom(nobs,size=1,prob=0.7)
x2 <- rnorm(nobs,0,1)
xb <- 1 - 1.5*x1_2 - 3.5*x2

exb <- exp(xb)
py <- rpois(nobs, exb)
pois <- data.frame(py, x1_2, x2)

# Code 6.4  - Bayesian Poisson model in R using JAGS

require(R2jags)

X <- model.matrix(~ x1_2 + x2, data = pois)
K <- ncol(X)

model.data <- list(Y = pois$py,
                   X = X,
                   K = K,               # number of betas
                   N = nrow(pois))      # sample size

sink("Poi.txt")

cat("
    model{
    for (i in 1:K) {beta[i] ~ dnorm(0, 0.0001)}

    for (i in 1:N) {
        Y[i] ~ dpois(mu[i])
        log(mu[i]) <- inprod(beta[], X[i,])
        }
    }",fill = TRUE)

sink()

inits <- function () {list(beta = rnorm(K, 0, 0.1))}

params <- c("beta")

POI <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = "Poi.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 4000,
            n.iter = 5000)

print(POI, intervals=c(0.025, 0.975), digits=3)
