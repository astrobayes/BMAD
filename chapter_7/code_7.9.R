# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication


# Data from code 7.6
rztp <- function(N, lambda){
  p <- runif(N, dpois(0, lambda),1)
  ztp <- qpois(p, lambda)
  return(ztp)
}


nobs <- 1000
x1 <- runif(nobs,-0.5,2.5)
xb <- 0.75 + 1.5*x1

exb <- exp(xb)
poy <- rztp(nobs, exb)
pdata <- data.frame(poy, x1)

xc <- -3 + 4.5*x1

pi <- 1/(1+exp((xc)))
bern <- rbinom(nobs,size =1, prob=1-pi)

pdata$poy <- pdata$poy*bern


# Code 7.9 - Zero-altered negative binomial (ZANB) or 
#           NB hurdle model in R using JAGS

require(R2jags)

Xc <- model.matrix(~ 1 + x1, data = pdata)
Xb <- model.matrix(~ 1 + x1, data = pdata)
Kc <- ncol(Xc)
Kb <- ncol(Xb)

model.data <- list(
  Y = pdata$poy,
  Xc = Xc,
  Xb = Xb,
  Kc = Kc,                                                        # number of betas − count
  Kb = Kb,                                                        # number of gammas − binary
  N = nrow(pdata),
  Zeros = rep(0, nrow(pdata)))

sink("NBH.txt")

cat("
    model{
    # Priors beta and gamma
    for (i in 1:Kc) {beta[i]  ~  dnorm(0, 0.0001)}
    for (i in 1:Kb) {gamma[i]  ~  dnorm(0, 0.0001)}
    
    # Prior for alpha
    alpha ~ dunif(0.001, 5)
    
    # Likelihood using zero trick
    C <- 10000
    
    for (i in 1:N) {
    Zeros[i]  ~  dpois(-ll[i] + C)
    LogTruncNB[i] <- 1/alpha * log(u[i])  +
                     Y[i] * log(1 - u[i]) + loggam(Y[i] + 1/alpha) -
                     loggam(1/alpha) - loggam(Y[i] + 1) -
                     log(1 - (1 + alpha * mu[i])^(-1/alpha))
    
    z[i] <- step(Y[i] - 0.0001)
    l1[i] <- (1 - z[i]) * log(1 - Pi[i])
    l2[i] <- z[i] * (log(Pi[i]) + LogTruncNB[i])
    ll[i] <- l1[i] + l2[i]
    u[i] <- 1/(1 + alpha * mu[i])
    log(mu[i]) <- inprod(beta[], Xc[i,])
    logit(Pi[i]) <- inprod(gamma[], Xb[i,])
    }
    }", fill = TRUE)
 
sink()

inits <- function () {
  list(beta = rnorm(Kc, 0, 0.1),
       gamma = rnorm(Kb, 0, 0.1),
       numS = rnorm(1, 0, 25),
       denomS = rnorm(1, 0, 1))}

params <- c("beta", "gamma", "alpha")

ZANB <- jags(data = model.data,
             inits = inits,
             parameters = params,
             model = "NBH.txt",
             n.thin = 1,
             n.chains = 3,
             n.burnin = 4000,
             n.iter = 6000)

print(ZANB, intervals=c(0.025, 0.975), digits=3)