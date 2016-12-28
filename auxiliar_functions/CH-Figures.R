require(lattice)
#library(RColorBrewer)
#jBuPuFun <- colorRampPalette(brewer.pal(n = 9, "BuPu"))
MyBUGSChains <- function(xx, vars){
#Small function to make an xyplot of the iterations per chain,
#for each variable 
  x <- xx$sims.array
  idchain.All <- NULL
  x1.All <- NULL
  ChainLength.All <- NULL
  id.All <- NULL

  NumBerChains <- ncol(x[,,vars[1]])
 
  for (i in vars){
	x1          <- as.vector(x[,,i])
	id          <- rep(rep(i, length = nrow(x[,,i])),NumBerChains)
	idchain     <- rep(1:NumBerChains, each = nrow(x[,,i]))
    ChainLength <- rep(1: nrow(x[,,i]), NumBerChains)

    x1.All <- c(x1.All, x1)
    ChainLength.All <- c(ChainLength.All, ChainLength)
    id.All <- c(id.All, id)
    idchain.All <- c(idchain.All, idchain)
   }

  Z <- xyplot(x1.All ~ ChainLength.All | factor(id.All) ,
       type = "l",
       strip = strip.custom(bg = 'white',
       par.strip.text = list(cex = 1.2)),
       scales = list(x = list(relation = "same", draw = TRUE),
                     y = list(relation = "free", draw = TRUE)),
       groups = idchain.All,  col = 1:NumBerChains,
       xlab = list(label = "MCMC iterations", cex = 1.5),
       ylab = list(label = "Sampled values", cex = 1.5))
  print(Z)
}

#######################


MyBUGSHist <- function(xx, vars, col = "#1f78b4"){
#Small function to make an histogram of the ACF per chain,
#for each variable 
  x <- xx$sims.matrix
  AllParams <- NULL
 
  for (i in vars){
  	#print(i)
    #Extract data from variable i
	Paramsi <- x[,i]
	  	print(length(Paramsi))

    AllParams <- c(AllParams, Paramsi)	
    }

   AllID <- rep(vars, each = nrow(x))

Z <- histogram( ~ AllParams | factor(AllID),
           strip = strip.custom(bg = 'white',
           par.strip.text = list(cex = 1.2)),
           type = "count" ,
           nint = 50,
           xlab = list(label = "Posterior distribution", cex = 1.5),
           col = col, 
           ylab = list(label = "Frequencies", cex = 1.5),
           scales = list(alternating = FALSE, 
                         x = list(relation = "free"),
                         y = list(relation = "free")),
           breaks=NULL,              
           panel = function(x, ...) {
             panel.histogram(x, ...)
             panel.abline(v = 0, lwd = 3, col =2)
             CI <- quantile(x, probs = c(0.025,0.975))
             panel.arrows (CI[1],-2, CI[2],-2, col = 2, lwd= 7, length=0)
             })
  print(Z)
}

##################################################

MyBUGSOutput <- function(xx,vars){
	 x <- xx$sims.matrix
    OUT <- matrix(nrow = length(vars), ncol=4) 
    j<-1
	for(i in vars){
	  xi <- x[,i]	
   	  OUT[j,3:4] <- quantile(xi, probs = c(0.025, 0.975))
   	  OUT[j,1] <- mean(xi)
   	  OUT[j,2] <- sd(xi)
   	  j <- j + 1
	}
	colnames(OUT) <- c("mean", "se", "2.5%", "97.5%")
	rownames(OUT) <- vars
	OUT
}

###############################

uNames <- function(k,Q){
  #Function to make a string of variables names of the form:
  #c("u[1]","u[2]", etc, "u[50]")	
  #Q=50 knots were used	
  String<-NULL
  for (j in 1:Q){String <- c(String, paste(k,"[",j,"]",sep = ""))}
  String
}







