# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.5 - Simulated random intercept binary logistic data

set.seed=13531

N <- 4000                                     # 20 groups, each with 200 observations
NGroups <- 20

x1 <- runif(N)
x2 <- runif(N)

Groups <- rep(1:20, each = 200)

a <- rnorm(NGroups, mean = 0, sd = 0.5)
eta <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]

mu <- 1/(1+exp(-eta))
y <- rbinom(N, prob=mu, size=1)

logitr <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups]
)

# Code 8.6 - Random intercept binary model in R

library(MCMCglmm)

BayLogitRI <- MCMCglmm(y ~ x1 + x2, random= ~Groups,
                       family="ordinal", data=logitr,
                       verbose=FALSE, burnin=10000, nitt=20000)

summary(BayLogitRI)