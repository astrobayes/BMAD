# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.16 - Random-intercept–random-slopes Poisson data in R

N <- 5000                        # 10 groups, each with 500 observations
NGroups <- 10

x1 <- runif(N)
x2 <- ifelse( x1<=0.500, 0, NA)
x2 <- ifelse(x1> 0.500, 1, x2)

Groups <- rep(1:10, each = 500)

a <- rnorm(NGroups, mean = 0, sd = 0.1)
b <- rnorm(NGroups, mean = 0, sd = 0.35)

eta <- 2 + 4 * x1 - 7 * x2 + a[Groups] + b[Groups]*x1
mu <- exp(eta)
y <- rpois(N, lambda = mu)

pric <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups)


# Code 8.17 - Random-intercept–random-slopes Poisson model in R using JAGS.

library(R2jags)

X <- model.matrix(~ x1 + x2, data = pric)
K <- ncol(X)

re <- as.numeric(pric$Groups)
NGroups <- length(unique(pric$Groups))

model.data <- list(Y = pric$y,
                   X = X,
                   N = nrow(pric),
                   b0 = rep(0, K),
                   B0 = diag(0.0001, K),
                   re = re,
                   a0 = rep(0, NGroups),
                   A0 = diag(1, NGroups))

sink("RICGLMM.txt")

cat("
    model{
    #Priors
    beta ~ dmnorm(b0[], B0[,])
    a ~ dmnorm(a0[], tau.ri * A0[,])
    b ~ dmnorm(a0[], tau.rs * A0[,])

    tau.ri ~ dgamma( 0.01, 0.01 )
    tau.rs ~ dgamma( 0.01, 0.01 )
    sigma.ri <- pow(tau.ri,-0.5)
    sigma.rs <- pow(tau.rs,-0.5)

    # Likelihood
    for (i in 1:N){
        Y[i] ~ dpois(mu[i])
        log(mu[i])<- eta[i]
        eta[i] <- inprod(beta[], X[i,]) + a[re[i]] + b[re[i]] * X[i,2]
        }
    }
    ",fill = TRUE)

sink()

# Initial values
inits <- function () {
    list(
        beta = rnorm(K, 0, 0.01),
        a = rnorm(NGroups, 0, 0.1),
        b = rnorm(NGroups, 0, 0.1))}

# Identify parameters
params <- c("beta", "sigma.ri", "sigma.rs","a","b")

# Run MCMC
PRIRS <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "RICGLMM.txt",
              n.thin = 10,
              n.chains = 3,
              n.burnin = 3000,
              n.iter = 4000)

print(PRIRS, intervals=c(0.025, 0.975), digits=3)