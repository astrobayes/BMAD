# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.20 - Synthetic probit data and model generated in R

set.seed(135)
nobs <- 1:2000

x1 <- runif(nobs)
x2 <- 2*runif(nobs)

xb <- 2 + 0.75 * x1 - 1.25 * x2

exb <- pnorm(xb)                               # probit inverse link
py <- rbinom(nobs, size=1, prob=exb)

probdata <- data.frame(py, x1, x2)


# Code 5.22 - Probit model in R using JAGS

attach(probdata)
require(R2jags)

set.seed(1944)
X <- model.matrix(~ x1 + x2,
                  data = probdata)
K <- ncol(X)
model.data <- list(Y = probdata$py,
                   N = nrow(probdata),
                   X =X,
                   K =K,
                   LogN = log(nrow(probdata)),
                   b0 = rep(0, K),
                   B0 = diag(0.00001, K)
)
sink("PROBIT.txt")
cat("
model{
    # Priors
    beta ~ dmnorm(b0[], B0[,])

    # Likelihood
    for (i in 1:N){
        Y[i] ~ dbern(p[i])
        probit(p[i]) <- max(-20, min(20, eta[i]))
        eta[i] <- inprod(beta[], X[i,])
        LLi[i] <- Y[i] * log(p[i]) +
        (1 - Y[i]) * log(1 - p[i])
    }

    LogL <- sum(LLi[1:N])

    # Information criteria
    AIC <- -2 * LogL + 2 * K
    BIC <- -2 * LogL + LogN * K
    }
    ",fill = TRUE)

sink()

# Initial parameter values
inits <- function () {
    list(beta = rnorm(K, 0, 0.1)) }

# parameters and statistics to display
params <- c("beta", "LogL", "AIC", "BIC")

PROBT <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "PROBIT.txt",
              n.thin = 1,
              n.chains = 3,
              n.burnin = 5000,
              n.iter = 10000)

print(PROBT, intervals=c(0.025, 0.975), digits=3)
