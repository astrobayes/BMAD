# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 8.3 - Code for clustered normal data

# Data
set.seed(1656)

N <- 4500                     
NGroups <- 20

x1 <- runif(N)
x2 <- runif(N)

Groups <- rep(1:20, each = 225)             #20 groups, each with 225 observations

# determine this by hand in order to get similar results to those obtained using JAGS
a <-  c(0.579, -0.115, -0.125,  0.169, -0.500, -1.429, -1.171, -0.205,  0.193, 0.041,
        -0.917, -0.353, -1.197,  1.044,  1.084, -0.085, -0.886, -0.352, -1.398,  0.350)

mu <- 1 + 0.2 * x1 - 0.75 * x2 + a[Groups]
y  <- rnorm(N, mean=mu, sd=2)

# prepare data for Stan
normr <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  Groups = Groups,
  RE = a[Groups]
)

# Code 8.4 - Random intercept normal model in R using Stan

library(rstan)

X <- model.matrix(~ x1 + x2, data = normr)
K <- ncol(X)
re <- as.numeric(normr$Groups)
Nre <- length(unique(normr$Groups))

model.data <- list(     
  Y = normr$y,            # response
  X = X,                  # covariates
  K = K,                  # number of betas
  N = nrow(normr),        # rows in model
  re = re,                # random effect
  Nre = Nre)              # hyperpriors for scale parameters

############### Fit
stan_model = "
data{
  int<lower=0> N;                            # number of data points
  int<lower=0> K;                            # number of covariates
  vector[N] y;                               # response variable
  matrix[N,K] X;                             # matrix of covariates
  int re[N];                                 # random effect/group identification
  int Nre;                                   # number of groups
}
parameters{
  vector[K] beta;                            # regression parameters
  vector[Nre] a;                             # groups coefficients
  real<lower=0, upper=10> sigma_plot;        # scatter for group coefficients
  real<lower=0, upper=10> sigma_eps;         # scatter around linear predictor
}
model{
  vector[N] mu;
  
  for (i in 1:N) mu[i] = dot_product(X[i], beta) + a[re[i]];
  
  # diffuse priors for regression parameters
  beta ~ normal(0, 1);
  
  # priors for random intercept group
  a ~ normal(0, sigma_plot);
  
  # likelihood
  y ~ normal(mu, sigma_eps);
}
"


# run MCMC
fit <- stan(model_code = stan_model,
            data = model.data,
            seed = 42,
            chains = 3,
            iter =10000,
            cores= 3,
            warmup=600
)

# Output 
print(fit,pars=c("beta", "a", "sigma_plot", "sigma_eps"),intervals=c(0.025, 0.975), digits=3)

