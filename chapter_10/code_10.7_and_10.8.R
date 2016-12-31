# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 10.7 Lognormal model in R using JAGS to describe the initial mass function (IMF)

library(R2jags)
# Data
path_to_data = "../data/Section_10p4/NGC6611.csv"
# Read data
IMF<-read.table(path_to_data,header = T)
N <-nrow(IMF)
x <- IMF$Mass
# prepare data to JAGS
jags_data <- list(x = x,
                  N = N # sample size
)
# Fit
LNORM <-" model{
# Uniform prior for standard deviation
tau <- pow(sigma, -2) # precision
sigma ~ dunif(0, 100) # standard deviation
mu ~ dnorm(0,1e-3)
# Likelihood function
for (i in 1:N){
x[i] ~ dlnorm(mu,tau)
}
}"
# Identify parameters
params <- c("mu", "sigma")
# Run mcmc
LN <- jags(
data = jags_data,
parameters = params,
model = textConnection(LNORM),
n.chains = 3,
n.iter = 5000,
n.thin = 1,
n.burnin = 2500)
# Output
print(LN, justify = "left", intervals=c(0.025,0.975), digits=2)

# Code 10.8 Plotting routine, in R, for Figure 10.6

require(jagstools)
require(ggplot2)
# Create new data for prediction
M = 750
xx = seq(from = 0.75*min(x),
         to = 1.05*max(x),
         length.out = M)
# Extract results
mx <- jagsresults(x=LN, params=c('mu'))
sigmax <- jagsresults(x=LN, params=c('sigma'))
# Estimate values for the Lognormal PDF
ymean <- dlnorm(xx,meanlog=mx[,"50%"],sdlog=sigmax[,"50%"])
ylwr1 <- dlnorm(xx,meanlog=mx[,"25%"],sdlog=sigmax[,"25%"])
ylwr2 <- dlnorm(xx,meanlog=mx[,"2.5%"],sdlog=sigmax[,"2.5%"])
yupr1 <- dlnorm(xx,meanlog=mx[,"75%"],sdlog=sigmax[,"75%"])
yupr2 <- dlnorm(xx,meanlog=mx[,"97.5%"],sdlog=sigmax[,"97.5%"])
# Create a data.frame for ggplot2
gdata <- data.frame(x=xx, mean = ymean,lwr1=ylwr1 ,lwr2=ylwr2,upr1=yupr1,upr2=yupr2)
ggplot(gdata,aes(x=xx))+
  geom_histogram(data=IMF,aes(x=Mass,y = ..density..),
                 colour="red",fill="gray99",size=1,binwidth = 0.075,
                 linetype="dashed")+
  geom_ribbon(aes(x=xx,ymin=lwr1, ymax=upr1,y=NULL),
              alpha=0.45, fill=c("#00526D"),show.legend=FALSE) +
  geom_ribbon(aes(x=xx,ymin=lwr2, ymax=upr2,y=NULL),
              alpha=0.35, fill = c("#00A3DB"),show.legend=FALSE) +
  geom_line(aes(x=xx,y=mean),colour="gray25",
            linetype="dashed",size=0.75,
            show.legend=FALSE)+
  ylab("Density")+
  xlab(expression(M/M['\u0298']))+
  theme_bw() +
  theme(legend.background = element_rect(fill = "white"),
        legend.key = element_rect(fill = "white",color = "white"),
        plot.background = element_rect(fill = "white"),
        legend.position = "top",
        axis.title.y = element_text(vjust = 0.1,margin = margin(0,10,0,0)),
        axis.title.x = element_text(vjust = -0.25),
        text = element_text(size = 25))