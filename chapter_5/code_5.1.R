# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2017, Cambridge Univ. Press

# Code 5.1 - GLM logistic regression in R

# Data
x <- c(13,10,15,9,18,22,29,13,17,11,27,21,16,14,18,8)
y <- c(1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0)

# Fit
mu <- (y + 0.5)/2                              # initialize mu
eta <- log(mu/(1-mu))                          # initialize eta with the Bernoulli link

for (i in 1:8) {
  w <- mu*(1-mu)                               # variance function
  z <- eta + (y - mu)/(mu*(1-mu))              # working response
  mod <- lm(z ~ x, weights=w)                  # weighted regression
  eta <- mod$fit                               # linear predictor
  mu <- 1/(1+exp(-eta))                        # fitted value
  cat(i, coef(mod), "\n")                      # displayed iteration log

}

# Output
summary(mod)

# logistic regression using glm package
mylogit <- glm(y ~ x, family=binomial)
summary(mylogit)