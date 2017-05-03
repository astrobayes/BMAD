# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.12 - Bernoulli model in R using JAGS, for accessing the relationship between bulge
#              size and the fraction of red spirals

require(R2jags)
# Data
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p6/Red_spirals.csv'

# Read data
Red <- read.csv(path_to_data,header=T)

# Prepare data to JAGS
N <- nrow(Red)
x <- Red$fracdeV
y <- Red$type

# Construct data dictionary
X <- model.matrix(~ x,
                   data = Red)

K <- ncol(X)

logit_data <- list(Y = y, # response variable
                   X = X, # predictors
                   N = N, # sample size
                   K = K # number of columns
)

# Fit
LOGIT <-"model{
    # Diffuse normal priors
    for(i in 1:K){
        beta[i] ~ dnorm(0, 1e-4)
    }

    # Likelihood function
    for (i in 1:N){
        Y[i] ~ dbern(p[i])
        logit(p[i]) <- eta[i]
        eta[i] <- inprod(beta[], X[i,])
    }
}"

# Define initial values
inits <- function () {
  list(beta = rnorm(ncol(X), 0, 0.1))
}

# Identify parameters
params <- c("beta")

# Fit
LOGIT_fit <- jags(data = logit_data,
                  inits = inits,
                  parameters = params,
                  model = textConnection(LOGIT),
                  n.thin = 1,
                  n.chains = 3,
                  n.burnin = 3000,
                  n.iter = 6000)

# Output
print(LOGIT_fit,intervals=c(0.025, 0.975),justify = "left", digits=2)