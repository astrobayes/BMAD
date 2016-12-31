# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.3 - Gaussian linear mixed model (in R using JAGS) for modeling the relationship
#             between type Ia supernovae host galaxy mass and Hubble residuals


library(R2jags)

# Data
path_to_data = "../data/Section_10p2/HR.csv"
dat <- read.csv(path_to_data, header = T)

# Prepare data to JAGS
nobs = nrow(dat)
obsx1 <- dat$LogMass
errx1 <- dat$e_LogMass
obsy <- dat$HR
erry <- dat$e_HR
type <- as.numeric(dat$Type) # convert class to numeric flag 1 or 2
jags_data <- list(
  obsx1 = obsx1,
  obsy = obsy,
  errx1 = errx1,
  erry = erry,
  K = 2,
  N = nobs,
  type = type)

# Fit
NORM_errors <-" model{
tau0 ~ dunif(1e-1,5)
mu0 ~ dnorm(0,1)
# Diffuse normal priors for predictors
for (j in 1:2){
for (i in 1:K) {
beta[i,j] ~  dnorm(mu0, tau0)
}
}
# Gamma prior for standard deviation
tau ~ dgamma(1e-3, 1e-3) # precision
sigma <- 1 / sqrt(tau) # standard deviation
# Diffuse normal priors for true x
for (i in 1:N){
x1[i] ~ dnorm(0,1e-3)
}
# Likelihood function
for (i in 1:N){
obsy[i] ~ dnorm(y[i],pow(erry[i],-2))
y[i] ~ dnorm(mu[i],tau)
obsx1[i] ~ dnorm(x1[i],pow(errx1[i],-2))
mu[i] <- beta[1,type[i]] + beta[2,type[i]] * x1[i]
}
}"
inits <- function () {
list(beta = matrix(rnorm(4, 0, 0.01),ncol = 2))
}
params0 <- c("beta", "sigma")
# Run MCMC
NORM <- jags(
data = jags_data,
inits = inits,
parameters = params0,
model = textConnection(NORM_errors),
n.chains = 3,
n.iter = 40000,
n.thin = 1,
n.burnin = 15000)

# Output
print(NORM,justify = "left", digits=3)