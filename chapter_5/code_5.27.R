# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 5.27 - Real data for a grouped binomial model

library(R2jags)

y <- c(6,11,9,13,17,21,8,10,15,19,7,12)
m <- c(45,54,39,47,29,44,36,57,62,55,66,48)
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)
x2 <- c(1,1,0,0,1,1,0,0,1,1,0,0)
x3 <- c(1,0,1,0,1,0,1,0,1,0,1,0)

bindata1 <-data.frame(y,m,x1,x2,x3)