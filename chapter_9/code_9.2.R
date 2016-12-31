# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 9.1 - Synthetic multivariate data in R

require(MASS)

# Data
set.seed(1056)
nobs <- 500                      # number of samples
nvar <- 15                       # number of predictors
rho <- 0.6                       # correlation between predictors

p <- rbinom(nvar,1,0.2)
beta <- round(p*rnorm(nvar,0,5),2)

# Check the coefficients
print(beta,2)

# Covariance matrix
d <- length(beta)
Sigma <- toeplitz(c(1, rep(rho, d - 1)))
Mu <- c(rep(0,d))

# Multivariate sampling
M <- mvrnorm(nobs, mu = Mu, Sigma = Sigma )
xb <- M %*% beta

# Dependent variable
y <- rnorm(nobs, xb, sd = 2)

# Code 9.2 Normal model applied to the multivariate synthetic data from Code 9.1

require(R2jags)

# Prepare data for JAGS
X <- model.matrix(~ M-1)
K <- ncol(X)
jags_data <- list(Y = y,
                  X = X,
                  K = K,
                  N = nobs)
# Fit
NORM <-" model{
    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }

    # Uniform prior for standard deviation
    tau <- pow(sigma, -2)                       # precision
    sigma ~ dgamma(1e-3, 1e-3)                  # standard deviation

    # Likelihood function
    for (i in 1:N){
    Y[i]~dnorm(mu[i],tau)
    mu[i] <- eta[i]
    eta[i] <- inprod(beta[], X[i,])
    }
}"
# Determine initial values
inits <- function () {
  list(beta = rnorm(K, 0, 0.01))
}
# Parameters to display and save
params <- c("beta", "sigma")

# MCMC
NORM_fit <- jags(data = jags_data,
                 inits = inits,
                 parameters = params,
                 model = textConnection(NORM),
                 n.chains = 3,
                 n.iter = 5000,
                 n.thin = 1,
                 n.burnin = 2500)

require(mcmcplots)

caterplot(NORM_fit,"beta",denstrip = T,
          greek = T,style = "plain", horizontal = F,
          reorder = F,cex.labels = 1.25,col="gray35")

caterpoints(beta, horizontal=F, pch="x", col="cyan")

