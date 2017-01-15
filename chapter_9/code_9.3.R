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
# beta <- round(p*rnorm(nvar,0,5),2)
beta = c(-2.43, 1.60, 0.00, 5.01, 4.12, 0.00, 0.00, 0.00, -0.89, 0.00, -2.31, 0.00, 0.00, 0.00, 0.00)

# Check the coefficients
#print(beta,2)

# Covariance matrix
d <- length(beta)
Sigma <- toeplitz(c(1, rep(rho, d - 1)))
Mu <- c(rep(0,d))

# Multivariate sampling
M <- mvrnorm(nobs, mu = Mu, Sigma = Sigma )
xb <- M %*% beta

# Dependent variable
y <- rnorm(nobs, xb, sd = 2)

# Code 9.3 K and M model applied to multivariate synthetic data from Code 9.1

require(R2jags)

# Prepare data for JAGS
X <- model.matrix(~ M-1)
K <- ncol(X)
jags_data <- list(Y = y,
                  X = X,
                  K = K,
                  N = nobs)

NORM_Bin <-" model{
    # Diffuse normal priors for predictors
    tauBeta <- pow(sdBeta,-2);
    sdBeta ~ dgamma(0.01,0.01)

    PInd <- 0.2
    for (i in 1:K){
        Ind[i] ~ dbern(PInd)
        betaT[i] ~ dnorm(0,tauBeta)
        beta[i] <- Ind[i]*betaT[i]
    }
    # Uniform prior for standard deviation
    tau <- pow(sigma, -2) # precision
    sigma ~ dgamma(0.01,0.01) # standard deviation

    # Likelihood function
    for (i in 1:N){
        Y[i] ~ dnorm(mu[i],tau)
        mu[i] <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}"

# Determine initial values
inits <- function () {
  list(betaT = rnorm(K, 0, 0.01))
}

# Parameters to display and save
params <- c("beta", "sigma")

# MCMC
NORM_fit <- jags(data = jags_data,
                 inits = inits,
                 parameters = params,
                 model = textConnection(NORM_Bin),
                 n.chains = 3,
                 n.iter = 5000,
                 n.thin = 1,
                 n.burnin = 2500)

print(NORM_fit, intervals=c(0.025, 0.975), digits=3)

require(mcmcplots)

caterplot(NORM_fit,"beta",denstrip = T,
          greek = T,style = "plain", horizontal = F,
          reorder = F,cex.labels = 1.25,col="gray35")

caterpoints(beta, horizontal=F, pch="x", col="cyan")

