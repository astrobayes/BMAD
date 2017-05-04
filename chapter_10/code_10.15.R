# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Data from Code 10.14
require(R2jags)
require(jagstools)

# Data
path_to_data = "https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p7/GCs.csv"

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



# Code 10.15 - Negative binomial model, in R using JAGS, for modeling the relationship
#              between globular cluster population and host galaxy visual magnitude

# Fit
model.NB <- "model{
    # Diffuse normal priors betas
    for (i in 1:K) { beta[i] ~ dnorm(0, 1e-5)}

    # Prior for theta
    theta ~ dgamma(1e-3,1e-3)

    for (i in 1:N){
        eta[i] <- inprod(beta[], X[i,])
        mu[i] <- exp(eta[i])
        p[i] <- theta/(theta+mu[i])
        Y[i] ~ dnegbin(p[i],theta)

        # Discrepancy
        expY[i] <- mu[i] # mean
        varY[i] <- mu[i] + pow(mu[i],2)/theta # variance
        PRes[i] <- ((Y[i] - expY[i])/sqrt(varY[i]))^2
    } 

    Dispersion <- sum(PRes)/(N-3)
}"

# Define initial values
inits <- function () {
  list(beta = rnorm(K, 0, 0.1))
}

# Identify parameters
params <- c("beta","theta","Dispersion")

# Start JAGS
NB_fit <- jags(data = JAGS_data ,
               inits = inits,
               parameters = params,
               model = textConnection(model.NB),
               n.thin = 1,
               n.chains = 3,
               n.burnin = 3500,
               n.iter = 7000)

# Output
# Plot posteriors
MyBUGSHist(out,c("Dispersion",uNames("beta",K),"theta"))

# Dump results on screen
print(NB_fit, intervals=c(0.025, 0.975), digits=3)