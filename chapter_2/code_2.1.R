# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 2.1 - Example of linear regression in R


# Data
y <- c(13,15,9,17,8,5,19,23,10,7,10,6)         # continuous response variable
x1 <- c(1,1,1,1,1,1,0,0,0,0,0,0)               # binary predictor
x2 <- c( 1,1,1,1,2,2,2,2,3,3,3,3)              # categorical predictor

# Fit
mymodel <- lm(y~x1 + x2)                       # linear regression of y on x1 and x2

# Output
summary(mymodel)                               # summary display
par(mfrow=c(2, 2))                             # create a 2 by 2 window
plot(mymodel)                                  # display of fitted vs. residuals plot, normal QQ plot
                                               # scale-location plot and residuals vs. leverage plot
