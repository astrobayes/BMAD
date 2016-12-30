# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.14 - Random Intercept negative binomial data in R

library(MASS)

N <- 2000                          # 10 groups, each with 200 observations
NGroups <- 10

x1 <- runif(N)
x2 <- runif(N)

Groups <- rep(1:10, each = 200)
a <- rnorm(NGroups, mean = 0, sd = 0.5)
eta <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]

mu <- exp(eta)
y <- rnegbin(mu, theta=2)

nbri <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups]
)

# Code 8.15 - Random intercept negative binomial mixed model in R

library(gamlss.mx)
nbrani <- gamlssNP(y ~ x1 + x2,
                   data = nbri,
                   random = ~ 1 | Groups,
                   family = NBI,
                   mixture = "gq", k = 20)

summary(nbrani)