# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.13 - Beta model in R using JAGS

require(R2jags)

# Data
set.seed(1056) # set seed to replicate example
nobs<-3000 # number of obs in model
x1 <- runif(nobs) # random normal variable

# generate toy data
xb <- 0.3+1.5*x1
exb <- exp(-xb)
p <- exb/(1+exb)
theta <- 15
y <- rbeta(nobs,theta*(1-p),theta*p)

# construct data dictionary
X <- model.matrix(~1+x1)
K <- ncol(X)

model.data <- list(Y = y, # response variable
                   X = X, # predictor matrix
                   K = K, # number of predictors including the intercept
                   N = nobs # sample size
)
Beta<-"model{
    # Diffuse normal priors for predictors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }
    
    # Gamma prior for precision parameter
    theta ~ dgamma(0.01,0.01)

    # Likelihood function
    for(i in 1:N){
        Y[i] ~ dbeta(shape1[i],shape2[i])
        shape1[i] <- theta*pi[i]
        shape2[i] <- theta*(1-pi[i])
        logit(pi[i]) <- eta[i]
        eta[i] <- inprod(beta[],X[i,])
    }
}"

# A function to generate initial values for mcmc
inits <- function () { list(beta = rnorm(ncol(X), 0, 0.1)) }

# Parameters to be displayed in output
params <- c("beta","theta")

BETAfit<- jags(data = model.data ,
               inits = inits,
               parameters = params, 
               model = textConnection(Beta),
               n.thin = 1,
               n.chains = 3,
               n.burnin = 2500,
               n.iter = 5000)

print(BETAfit,justify="left", digits=2)
