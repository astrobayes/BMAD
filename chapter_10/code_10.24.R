# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication


# Data from  Code 10.22 

require(R2jags)
require(jagstools)

# Data
sunspot <- read.csv("https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p10/sunspot.csv",header = T, sep=",")

# Prepare data to JAGS
y <- round(sunspot[,2])
t <- seq(1700,2015,1)
N <- length(y)

sun_data <- list(Y = y,                  # Response variable
                 N = N)                  # Sample size

# Code 10.24- Negative binomial model (AR1) for assessing the evolution of the number of
#              sunspots through the years.

# Fit
AR1_NB<-"model{
    for(i in 1:2){
        phi[i] ~ dnorm(0,1e-2)
    }

    theta ~ dgamma(0.001,0.001)
    mu[1] <- Y[1]

    # Likelihood function
    for (t in 2:N) {
        Y[t] ~ dnegbin(p[t],theta)
        p[t] <- theta/(theta+mu[t])
        log(mu[t]) <- phi[1] + phi[2]*Y[t-1]
    }

    for (t in 1:N){
        Yx[t] ~ dnegbin(px[t],theta)
        px[t] <- theta/(theta+mu[t])
    }
}"

# Identify parameters
# Include Yx in the list bellow only if interested in prediction
params <- c("phi","theta")

# Generate initial values for mcmc
inits <- function () {
  list(phi = rnorm(2, 0, 0.1))
}

# Run mcmc
jagsfit <- jags(data = sun_data, 
                inits = inits, 
                parameters = params, 
                model = textConnection(AR1_NB),
                n.thin = 1, 
                n.chains=3,
                n.burnin=5000,
                n.iter = 10000)

# Output
print(jagsfit,intervals=c(0.025, 0.975),justify = "left", digits=2)