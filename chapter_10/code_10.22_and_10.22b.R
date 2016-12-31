# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication


# Code 10.22 - Normal autoregressive model AR(1) for accessing the evolution of the number
#              of sunspots through the years

require(R2jags)
require(jagstools)
# Data
# Read data
sunspot <- read.csv("../data/Section_10p10/sunspot.csv",header = T, sep=",")
# Prepare data to JAGS
y <- round(sunspot[,2])
t <- seq(1700,2015,1)
N <- length(y)
sun_data <- list(Y = y, # Response variable
                 N = N) # Sample size
# Fit
AR1_NORM<-"model{
# Priors
sd2 ~ dgamma(1e-3,1e-3)
tau <- 1/sd2
sd <- sqrt(sd2)
for(i in 1:2){
phi[i] ~dnorm(0,1e-2)
}
mu[1] <- Y[1]
# Likelihood function
for (t in 2:N) {
Y[t] ~ dnorm(mu[t],tau)
mu[t] <- phi[1] + phi[2] * Y[t-1]
}
# Prediction
for (t in 1:N){
Yx[t]~dnorm(mu[t],tau)
}
}"
# Generate initial values for mcmc
inits <- function () {
list(phi = rnorm(2, 0, 0.1))
}
# Identify parameters
# Include Yx only if you intend to generate plots
params <- c("sd", "phi", "Yx")
# Run mcmc
jagsfit <- jags(data = sun_data,
inits = inits,
parameters = params,
model = textConnection(AR1_NORM),
n.thin = 1,
n.chains = 3,
n.burnin = 3000,
n.iter = 5000)

# Output
print(jagsfit,intervals = c(0.025, 0.975),justify = "left", digits=2)

require(ggplot2)
require(jagstools)
# Format data for ggplot
sun <- data.frame(x=t,y=y) # original data
yx <- jagsresults(x=jagsfit, params=c("Yx")) # fitted values
gdata <- data.frame(x = t, mean = yx[,"mean"],lwr1=yx[,"25%"],
                    lwr2=yx[,"2.5%"],upr1=yx[,"75%"],upr2=yx[,"97.5%"])
# Plot
ggplot(sun,aes(x=t,y=y))+
  geom_ribbon(data=gdata,aes(x=t,ymin=lwr1, ymax=upr1,y=NULL),
              alpha=0.95, fill=c("#969696"),show.legend=FALSE)+
  geom_ribbon(data=gdata,aes(x=t,ymin=lwr2, ymax=upr2,y=NULL),
              alpha=0.75, fill=c("#d9d9d9"),show.legend=FALSE)+
  geom_line(data=gdata,aes(x=t,y=mean),colour="black",
            linetype="solid",size=0.5,show.legend=FALSE)+
  geom_point(colour="orange2",size=1.5,alpha=0.75)+
  theme_bw() + xlab("Year") + ylab("Sunspots") +
  theme(legend.background = element_rect(fill="white"),
        legend.key = element_rect(fill = "white",color = "white"),
        plot.background = element_rect(fill = "white"),
        legend.position="top",
        axis.title.y = element_text(vjust = 0.1,margin=margin(0,10,0,0)),
        axis.title.x = element_text(vjust = -0.25),
        text = element_text(size = 25,family="serif"))+
  geom_hline(aes(yintercept=0),linetype="dashed",colour="gray45",size=1.25)