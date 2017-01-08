library(MASS)
require(VGAM)
set.seed(42)
nobs <- 500
x1 <- rbinom(nobs,size=1,0.5)
x2 <- rbinom(nobs,size=1,0.5)
m <- rep(1:5, each=100, times=1)*100 # creates offset as defined
logm <- log(m)
theta <- 20
xb <- 2+0.75*x1 -1.25*x2+logm
exb <- exp(xb)
nby <- rnegbin(n=nobs, mu=exb, theta = theta)
nbdata <- data.frame(nby,m,x1,x2)


require(R2jags)
X <- model.matrix(~ x1 + x2, data = nbdata)
K <- ncol(X)
model.data <- list(
  Y = nbdata$nby,
  X = X,
  K = K,
  N = nrow(nbdata),
  m = nbdata$m)
sink("NBOFF.txt")
cat("
model{
# Priors for betas
for (i in 1:K) { beta[i] ~ dnorm(0, 0.01)}
# Prior for alpha
alpha ~ dgamma(0.01, 0.01)
# Likelihood function
for (i in 1:N){
Y[i] ~ dnegbin(p[i], 1/alpha)
p[i] <- 1.0/(1.0 + alpha*mu[i])
log(mu[i]) <- inprod(beta[], X[i,])+log(m[i])
}
}
",fill = TRUE)
sink()
inits <- function () {list(
    beta = rnorm(K, 0, 0.1),
    alpha = runif(0.01, 1)
)}

params <- c("beta", "alpha")
NBofd <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model = "NBOFF.txt",
              n.thin = 3,
              n.chains = 3,
              n.burnin = 10000,
              n.iter = 15000)
print(NBofd, intervals=c(0.025, 0.975), digits=3)