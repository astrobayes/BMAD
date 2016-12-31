# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.14 - Poisson model, in R using JAGS, for modeling the relation between globular
#              clusters population and host galaxy visual magnitude

require(R2jags)
require(jagstools)
# Data
path_to_data = "../data/Section_10p7/GCs.csv"
# Read data
GC_dat = read.csv(file=path_to_data,header = T,dec=".")
# Prepare data to JAGS
N <- nrow(GC_dat)
x <- GC_dat$MV_T
y <- GC_dat$N_GC
X <- model.matrix(~ x, data=GC_dat)
K = ncol(X)
JAGS_data <- list(
  Y = y,
  X = X,
  N = N,
  K = K)
# Fit
model.pois <- "model{
# Diffuse normal priors betas
for (i in 1:K) { beta[i] ~ dnorm(0, 1e-5)}
for (i in 1:N){
# Likelihood
eta[i]<-inprod(beta[], X[i,])
mu[i] <- exp(eta[i])
Y[i]~dpois(mu[i])
# Discrepancy
expY[i] <- mu[i] # mean
varY[i] <- mu[i] # variance
PRes[i] <- ((Y[i] - expY[i])/sqrt(varY[i]))^2
}
Dispersion <- sum(PRes)/(N-2)
}"
# Define initial values
inits <- function () {
list(beta = rnorm(K, 0, 0.1))
}
# Identify parameters
params <- c("beta","Dispersion")
# Start JAGS
pois_fit <- jags(data = JAGS_data ,
inits = inits,
parameters = params,
model = textConnection(model.pois),
n.thin = 1,
n.chains = 3,
n.burnin = 3500,
n.iter = 7000)
# Output
print(pois_fit , intervals=c(0.025, 0.975), digits=3)

require(lattice)
source("../auxiliar_functions/CH-Figures.R")
out <- pois_fit$BUGSoutput
MyBUGSHist(out,c("Dispersion",uNames("beta",K)))