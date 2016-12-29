# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 7.6 - Synthetic data for Poisson–logit hurdle zero-altered models

rztp <- function(N, lambda){
  p <- runif(N, dpois(0, lambda),1)
  ztp <- qpois(p, lambda)
  return(ztp)
}

# Sample size
nobs <- 2000

# Generate predictors, design matrix
x1 <- runif(nobs,-0.5,2.5)
xb <- 0.75 + 1.5*x1

# Construct Poisson responses
exb <- exp(xb)
poy <- rztp(nobs, exb)
pdata <- data.frame(poy, x1)

# Generate predictors for binary part
xc <- -3 + 4.5*x1

# Construct filter
pi <- 1/(1+exp((xc)))
bern <- rbinom(nobs,size =1, prob=1-pi)

# Add structural zeros
pdata$poy <- pdata$poy*bern

# Code 7.7 - Bayesian Poisson–logit hurdle zero-altered models

require(R2jags)

Xc <- model.matrix(~ 1 + x1, data=pdata)
Xb <- model.matrix(~ 1 + x1, data=pdata)
Kc <- ncol(Xc)
Kb <- ncol(Xb)

model.data <- list(
  Y = pdata$poy,                                    # response
  Xc = Xc,                                          # covariates
  Xb = Xb,                                          # covariates
  Kc = Kc,                                          # number of betas
  Kb = Kb,                                          # number of gammas
  N = nrow(pdata),                                  # sample size
  Zeros = rep(0, nrow(pdata)))

sink("HPL.txt")

cat("
model{
    # Priors beta and gamma
    for (i in 1:Kc) {beta[i] ~ dnorm(0, 0.0001)}
    for (i in 1:Kb) {gamma[i] ~ dnorm(0, 0.0001)}

    # Likelihood using zero trick
    C <- 10000

    for (i in 1:N) {
        Zeros[i] ~ dpois(-ll[i] + C)
        LogTruncPois[i] <- log(Pi[i]) + Y[i] * log(mu[i]) - mu[i] -(log(1 - exp(-mu[i])) + loggam(Y[i] + 1) )
        z[i] <- step(Y[i] - 0.0001)
        l1[i] <- (1 - z[i]) * log(1 - Pi[i])
        l2[i] <- z[i] * (log(Pi[i]) + LogTruncPois[i])
        ll[i] <- l1[i] + l2[i]
        log(mu[i]) <- inprod(beta[], Xc[i,])
        logit(Pi[i]) <- inprod(gamma[], Xb[i,])
     }
}", fill = TRUE)

sink()

inits <- function () {
  list(beta = rnorm(Kc, 0, 0.1),
       gamma = rnorm(Kb, 0, 0.1))}

params <- c("beta", "gamma")

ZAP <- jags(data = model.data,
            inits = inits,
            parameters = params,
            model = "HPL.txt",
            n.thin = 1,
            n.chains = 3,
            n.burnin = 2500,
            n.iter = 5000)

print(ZAP, intervals=c(0.025, 0.975), digits=3)
