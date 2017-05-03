# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.10 - Beta model in R using JAGS, for accessing the relationship between the baryon
#              fraction in atomic gas and galaxy stellar mass.

require(R2jags)
# Data
path_to_data = "https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p5/f_gas.csv"

# Read data
Fgas0 <-read.csv(path_to_data,header=T)

# Estimate F_gas
Fgas0$fgas <- Fgas0$M_HI/(Fgas0$M_HI+Fgas0$M_STAR)

# Prepare data to JAGS
N = nrow(Fgas0)
y <- Fgas0$fgas
x <- log(Fgas0$M_STAR,10)
X <- model.matrix(~ 1 + x)
K <- ncol(X)
beta_data <- list(Y = y,
                  X = X,
                  K = K,
                  N = N)

# Fit
Beta <-"model{
    # Diffuse normal priors for predictors
    for(i in 1:K){
        beta[i] ~ dnorm(0, 1e-4)
    }
    
    # Diffuse prior for theta
    theta~dgamma(0.01,0.01)

    # Likelihood function
    for (i in 1:N){
        Y[i] ~ dbeta(a[i],b[i])
        a[i] <- theta*pi[i]
        b[i] <- theta*(1-pi[i])
        logit(pi[i]) <- eta[i]
        eta[i] <- inprod(beta[],X[i,])
    }
}"

# Define initial values
inits <- function () {
  list(beta = rnorm(2, 0, 0.1),
       theta = runif(1,0,100))
}

# Identify parameters
params <- c("beta","theta")

Beta_fit <- jags(data = beta_data,
                 inits = inits,
                 parameters = params,
                 model = textConnection(Beta),
                 n.thin = 1,
                 n.chains = 3,
                 n.burnin = 5000,
                 n.iter = 7500)

# Output
print(Beta_fit,intervals=c(0.025, 0.975),justify = "left", digits=2)