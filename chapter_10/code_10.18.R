# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication


# Code 10.18 - Bernoulli logit model, in R using JAGS, for accessing the relationship between
#              Seyfert AGN activity and galactocentric distance

library(R2jags)

# Data
data<-read.csv("https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p8/Seyfert.csv",header=T)

# identify data elements
X <- model.matrix( ~ logM200 + r_r200, data = data)
K <- ncol(X) # number of predictors
y <- data$bpt # response variable
n <- length(y) # sample size
gal <- as.numeric(data$zoo) # galaxy type

# Prepare data for JAGS
jags_data <- list(Y = y,
                  N = n,
                  X = X,
                  gal = gal)

# Fit
jags_model<-"model{
    # Shared hyperpriors for beta
    tau ~ dgamma(1e-3,1e-3) # precision
    mu ~ dnorm(0,1e-3) # mean

    # Diffuse prior for beta
    for(j in 1:2){
        for(k in 1:3){
            beta[k,j] ~ dnorm(mu,tau)
        }
    }

    # Likelihood
    for(i in 1:N){
        Y[i] ~ dbern(pi[i])
        logit(pi[i]) <- eta[i]
        eta[i] <- beta[1,gal[i]]*X[i,1]+
        beta[2,gal[i]]*X[i,2]+
        beta[3,gal[i]]*X[i,3]
    }
}"

# Identify parameters to monitor
params <- c("beta")

# Generate initial values
inits <- function () {
    list(beta = matrix(rnorm(6,0, 0.01),ncol=2))
}

# Run mcmc
jags_fit <- jags(data= jags_data,
                 inits = inits,
                 parameters = params,
                 model.file = textConnection(jags_model),
                 n.chains = 3,
                 n.thin = 10,
                 n.iter = 5*10^4,
                 n.burnin = 2*10^4
)

# Output
print(jags_fit,intervals=c(0.025, 0.975), digits=3)