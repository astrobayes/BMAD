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
summary(mynb0 <- glm.nb(nby ~ x1 + x2 + offset(log(m)), data=nbdata))