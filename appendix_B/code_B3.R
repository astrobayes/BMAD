y <- c(6,11,9,13,17,21,8,10,15,19,7,12)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48)
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)
x2 <- c(1,1,0,0,1,1,0,0,1,1,0,0)
x3 <- c(1,0,1,0,1,0,1,0,1,0,1,0)
ratep <-data.frame(y,m,x1,x2,x3)

require(R2jags)
X <- model.matrix(~ x1 + x2+x3, data = ratep)
K <- ncol(X)
model.data <- list(Y = ratep$y,
                   N = nrow(ratep),
                   X = X,
                   K = K,
                   m = ratep$m)
sink("PRATE.txt")
cat("
model{
# Diffuse normal priors betas
for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001)}
# Likelihood
for (i in 1:N){
Y[i] ~ dpois(mu[i])
log(mu[i]) <- inprod(beta[], X[i,]) + log(m[i])
}
}
",fill = TRUE)
sink()
# Initial parameter values
inits <- function () {
  list(
    beta = rnorm(K, 0, 0.1)
  ) }
params <- c("beta")
poisR <- jags(data = model.data,
              inits = inits,
              parameters = params,
              model.file = "PRATE.txt",
              n.thin = 3,
              n.chains = 3,
              n.burnin = 8000,
              n.iter = 12000)
print(poisR, intervals=c(0.025, 0.975), digits=3)

poir <-glm(y ~ x1 + x2 + x3 + offset(log(m)), family=poisson, data=ratep)
summary(poir)