# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 4.2 Normal linear model in R using JAGS and the zero trick

require(R2jags)

set.seed(1056)                            # set seed to replicate example
nobs = 5000                               # number of obs in model
x1 <- runif(nobs)                         # predictor, random uniform variable
beta0 = 2.0                               # intercept
beta1 = 3.0                               # predictor
xb <- beta0 + beta1 * x1                  # linear predictor, xb
y <- rnorm(nobs, xb, sd=1)                # y as an adjusted random normal variate

# Model setup
X <- model.matrix(~ 1 + x1)
K <- ncol(X)
model.data <- list(Y = y,                 # Response variable
                   X = X,                 # Predictors
                   K = K,                 # Number of predictors including the intercept
                   N = nobs,              # Sample size
                   Zeros = rep(0, nobs)   # Zero trick
)

NORM0 <- "
    model{

    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }

    # Uniform prior for standard deviation
    tau <- pow(sigma, -2)                 # precision
    sigma ~ dunif(0, 100)                 # standard deviation

    # Likelihood function
    C <- 10000

    for (i in 1:N){
        Zeros[i] ~ dpois(Zeros.mean[i])
        Zeros.mean[i] <- -L[i] + C
        l1[i] <- -0.5 * log(2*3.1416) - 0.5 * log(sigma)
        l2[i] <- -0.5 * pow(Y[i] - mu[i],2)/sigma
        L[i] <- l1[i] + l2[i]
        mu[i] <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}"

inits <- function () {
                      list(beta = rnorm(K, 0, 0.01))
}

params <- c("beta", "sigma")

norm0fit <- jags(data = model.data,
                 inits = inits,
                 parameters = params,
                 model = textConnection(NORM0),
                 n.chains = 3,
                 n.iter = 15000,
                 n.thin = 1,
                 n.burnin = 10000)

print(norm0fit,intervals=c(0.025, 0.975), digits=2)