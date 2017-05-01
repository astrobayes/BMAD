# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication


# Code 8.9 - Random intercept binomial logistic data in R

y <- c(6,11,9,13,17,21,8,10,15,19,7,12,8,5,13,17,5,12,9,10)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48,49,39,28,35,39,43,50,36)
x1 <- c(1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0)
x2 <- c(1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0)
Groups <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
logitr <- data.frame(y,m,x1,x2,Groups)

# Code 8.10 - Random intercept binomial logistic model in using JAGS

library(R2jags)

X <- model.matrix(~ x1 + x2, data = logitr)
K <- ncol(X)

re <- length(unique(logitr$Groups))
Nre <- length(unique(Groups))

model.data <- list(
  Y = logitr$y,                     # response
  X = X,                            # covariates
  m = m,                            # binomial denominator
  N = nrow(logitr),                 # sample size
  re = logitr$Groups,               # random effects
  b0 = rep(0,K),
  B0 = diag(0.0001, K),
  a0 = rep(0,Nre),
  A0 = diag(Nre))

sink("GLMM.txt")

cat("
model{
# Diffuse normal priors for regression parameters
beta ~ dmnorm(b0[], B0[,])

# Priors for random effect group
a ~ dmnorm(a0, tau * A0[,])
num ~ dnorm(0, 0.0016)
denom ~ dnorm(0, 1)
sigma <- abs(num / denom)
tau <- 1 / (sigma * sigma)

# Likelihood function
for (i in 1:N){
    Y[i] ~ dbin(p[i], m[i])
    logit(p[i]) <- eta[i]
    eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
    }
}",fill = TRUE)

sink()

inits <- function () {
  list(
    beta = rnorm(K, 0, 0.1),
    a = rnorm(Nre, 0, 0.1),
    num = rnorm(1, 0, 25),
    denom = rnorm(1, 0, 1))}

params <- c("beta", "a", "sigma")

LOGIT0 <- jags(data = model.data,
               inits = inits,
               parameters = params,
               model.file = "GLMM.txt",
               n.thin = 10,
               n.chains = 3,
               n.burnin = 4000,
               n.iter = 5000)

print(LOGIT0, intervals=c(0.025, 0.975), digits=3)