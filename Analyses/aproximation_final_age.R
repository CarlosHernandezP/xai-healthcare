df<-read.csv("age_SEER.csv")
y <- df$total[1:89]
y90 <- df$total[90]
x <- 1:89
t.Max <- 110
y<-c(y,mean(y[1:10]))
x<-c(x,t.Max)
t <-1:t.Max
tfinal <- 90:t.Max

plot(x,y)

library(mgcv)
gam0 <- gam(y~s(x,bs="tp"),data=data.frame(x=x,y=y))
gam1 <- gam(log(y)~s(x,bs="cr"),data=data.frame(x=x,y=y))
plot(gam1,residuals = TRUE, pch=20)
pred.t <- predict(gam1, newdata = data.frame(x=t))
pred.tfinal <- predict(gam1,newdata = data.frame(x=tfinal))

plot(x,y,xlim=c(0,t.Max))
lines(t,exp(pred.t),col=2)

plot(gam0,residuals = TRUE, pch=20, xlim=c(0,t.Max), shift = mean(y))
abline(h=0)
lines(t,exp(pred.t),col=2)

########

loess0 <- loess(y~x,data=data.frame(x=x,y=y), span = 0.35)
loess1 <- loess(log(y)~x,data=data.frame(x=x,y=y), span = 0.35)
plot(x,log(y),xlim=c(0,t.Max))
lines(loess1$x, loess1$fitted, pch=20)
pred.t <- predict(loess1, newdata = data.frame(x=t))
pred.tfinal <- predict(loess1,newdata = data.frame(x=tfinal))

plot(x,y,xlim=c(0,t.Max))
lines(t,exp(pred.t),col=2)

plot(x,y,xlim=c(0,t.Max))
lines(loess0$x, loess0$fitted, pch=20, xlim=c(0,t.Max))
abline(h=0)
lines(t,exp(pred.t),col=2)

#####

( exp.y.tfinal <- sum(tfinal*exp(pred.tfinal))/sum(exp(pred.tfinal)) )
# t.Max <- 110, gam
# [1] 94.49056
#
# t.Max <- 110, loess
# [1] 94.57833 
#
# t.Max <- 105, gam
# [1] 93.44265
#
# t.Max <- 105, loess
# [1] 93.47748 
#

#### Conclusión:
# exp.age.final <- 94.5