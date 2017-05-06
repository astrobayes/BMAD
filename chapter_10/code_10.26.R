# From Bayesian Models for Astrophysical Data 
# by Hilbe, de Souza & Ishida, 2016, Cambridge Univ. Press
#
# Code 10.26 Bayesian normal model for cosmological parameter 
#            inference from type Ia supernova data in R using Stan.
#
# Statistical Model: Gaussian regression in R using Stan
#                    example using ODE
#
# Astronomy case: Cosmological parameters inference from 
#                 type Ia supernovae data 
#
# Data: JLA sample, Betoule et al., 2014  
# http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html
#
# 1 response (obsy - observed magnitude)
# 5 explanatory variable (redshift - redshift,
#                         ObsMag   - apparent magnitude,
#                         x1       - stretch,
#                         color    - color,
#                         hmass    - host mass)


library(rstan)

# Preparation

# set initial conditions
z0 = 0                          # initial redshift
E0 = 0                          # integral(1/E) at z0

# physical constants
c = 3e5                         # speed of light
H0 = 70                         # Hubble constant

# Data
data <- read.table("https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p11/jla_lcparams.txt",header=T)

# remove repeated redshift 
data2<-data[!duplicated(data$zcmb),]

# prepare data for Stan
nobs         <- nrow(data2)                 # number of SNe
index        <- order(data2$zcmb)           # sort according to redshift
ObsMag       <- data2$mb[index]             # apparent magnitude
redshift     <- data2$zcmb[index]           # redshift
color        <- data2$color[index]          # color
x1           <- data2$x1[index]             # stretch
hmass        <- data2$m3rdvar[index]        # host mass

stan_data  <- list(nobs = nobs,
                   E0 = array(E0,dim=1),
                   z0 = z0,
                   c = c,
                   H0 = H0,
                   obs_mag = ObsMag,    
                   redshift = redshift, 
                   x1 = x1, 
                   color = color,
                   hmass = hmass)

# Fit
stan_model="
functions {
     /** 
     * ODE for the inverse Hubble parameter. 
     * System State E is 1 dimensional.  
     * The system has 2 parameters theta = (om, w)
     * 
     * where 
     * 
     *   om:       dark matter energy density 
     *   w:        dark energy equation of state parameter
     *
     * The system redshift derivative is 
     * 
     * d.E[1] / d.z  =  
     *  1.0/sqrt(om * pow(1+z,3) + (1-om) * (1+z)^(3 * (1+w)))
     * 
     * @param z redshift at which derivatives are evaluated. 
     * @param E system state at which derivatives are evaluated. 
     * @param params parameters for system. 
     * @param x_r real constants for system (empty). 
     * @param x_i integer constants for system (empty). 
     */ 
     real[] Ez(real z,
               real[] H,
               real[] params,
               real[] x_r,
               int[] x_i) {
           real dEdz[1];

           dEdz[1] = 1.0/sqrt(params[1]*(1+z)^3
                     +(1-params[1])*(1+z)^(3*(1+params[2])));

           return dEdz;
    } 
}
data {
    int<lower=1> nobs;              // number of data points
    real E0[1];                     // integral(1/H) at z=0                           
    real z0;                        // initial redshift, 0
    real c;                         // speed of light
    real H0;                        // hubble parameter
    vector[nobs] obs_mag;           // observed magnitude at B max
    real x1[nobs];                  // stretch
    real color[nobs];               // color 
    real redshift[nobs];            // redshift
    real hmass[nobs];               // host mass
}
transformed data {
      real x_r[0];                  // required by ODE (empty)
      int x_i[0]; 
}
parameters{
      real<lower=0, upper=1> om;    // dark matter energy density
      real alpha;                   // stretch coefficient   
      real beta;                    // color coefficient
      real Mint;                    // intrinsic magnitude
      real deltaM;
      real<lower=0> sigint;         // magnitude dispersion
      real<lower=-2, upper=0> w;    // dark matter equation of state parameter
}
transformed parameters{
      real DC[nobs,1];                        // co-moving distance 
      real pars[2];                           // ODE input = (om, w)
      vector[nobs] mag;                       // apparent magnitude
      real dl[nobs];                          // luminosity distance
      real DH;                                // Hubble distance = c/H0
 
      DH = (c/H0);

      pars[1] = om;
      pars[2] = w;

      # Integral of 1/E(z) 
      DC = integrate_ode_rk45(Ez, E0, z0, redshift, pars,  x_r, x_i);

      for (i in 1:nobs) {
            dl[i] = DH * (1 + redshift[i]) * DC[i, 1];
            if (hmass[i] < 10) mag[i] = 25 + 5 * log10(dl[i]) + Mint - alpha * x1[i] + beta * color[i];
            else mag[i] = 25 + 5 * log10(dl[i]) + Mint + deltaM - alpha * x1[i] + beta * color[i];
      }
}
model {

      # priors and likelihood
      sigint ~ gamma(0.001, 0.001);
      Mint ~ normal(-20, 5.);
      beta ~ normal(0, 10);
      alpha ~ normal(0, 1);
      deltaM ~ normal(0, 1);
      obs_mag ~ normal(mag, sigint);   
}
"

# run MCMC
fit <- stan(model_code = stan_model,
                data = stan_data,
                seed = 42,
                chains = 3,
                iter =15000,
                cores= 3,
                warmup=7500
)

# Output 
print(fit,pars=c("om", "Mint","w","alpha","beta","deltaM","sigint"),intervals=c(0.025, 0.975), digits=3)
