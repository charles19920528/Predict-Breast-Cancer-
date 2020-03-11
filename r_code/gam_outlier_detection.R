library(gam)
library(MASS)
library(lawstat)
cancer_dt <- read.csv("~/Documents/courses/sta_223/project/dataR2.csv")
cancer_dt$Classification = as.numeric(cancer_dt$Classification == 2)
# 17, 38, 39
# cancer_dt = cancer_dt[-c(87, 78, 88, 85, 86), ]
# cancer_dt = cancer_dt[-c(17), ]
rownames(cancer_dt) = as.character(1:nrow(cancer_dt))

####################
# Helper functions #
####################
residual_plot = function(bwtfit){
  res.P = residuals(bwtfit, type="pearson")
  res.D = residuals(bwtfit, type="deviance") #or residuals(fit), by default
  
  par(mfrow=c(1,2))
  plot(bwtfit$fitted.values, res.P, pch=16, cex=0.6, ylab='Pearson Residuals', xlab='Fitted Values')
  lines(smooth.spline(bwtfit$fitted.values, res.P, spar=0.95), col=2)
  abline(h=0, lty=2, col='grey')
  plot(bwtfit$fitted.values, res.D, pch=16, cex=0.6, ylab='Deviance Residuals', xlab='Fitted Values')
  lines(smooth.spline(bwtfit$fitted.values, res.D, spar=0.9), col=2)
  abline(h=0, lty=2, col='grey')
  
  residual_list = list("pearson" = res.P, "deviance" = res.D)
  return(residual_list)
}

residual_boxplot = function(bwtfit) {
  res.P = residuals(bwtfit, type="pearson")
  res.D = residuals(bwtfit, type="deviance") #or residuals(fit), by default
  boxplot(cbind(res.P, res.D), names = c("Pearson", "Deviance"))
}

plot_leverage = function(bwtfit, data_frame, number_of_points_to_mark) {
  leverage = hatvalues(bwtfit)
  
  W = diag(bwtfit$weights)
  X = model.matrix(bwtfit)
  Hat = sqrt(W) %*% X %*% solve(t(X) %*% W %*% X) %*% t(X) %*% sqrt(W)
  all(abs(leverage - diag(Hat)) < 1e-15)
  
  plot(names(leverage), leverage, xlab="Index", type="h")
  points(names(leverage), leverage, pch=16, cex=0.6)
  
  susPts <- as.numeric(names(sort(leverage, decreasing=TRUE)[1:number_of_points_to_mark]))
  text(susPts, leverage[susPts], susPts, adj=c(-0.1,-0.1), cex=0.7, col=4)
  
  p <- length(coef(bwtfit))
  n <- nrow(data_frame)
  abline(h=2*p/n,col=2,lwd=2,lty=2)
  leverage_points <- which(leverage>2*p/n)
  
  return(leverage_points)
}

plot_cooks = function(bwtfit, leverage_points, number_of_points_to_mark) {
  cooks = cooks.distance(bwtfit)
  
  plot(cooks, ylab="Cook's Distance", pch=16, cex=0.6)
  points(leverage_points, cooks[leverage_points], pch=17, cex=0.8, col=2)
  susPts <- as.numeric(names(sort(cooks[leverage_points], decreasing=TRUE)[1:number_of_points_to_mark]))
  text(susPts, cooks[susPts], susPts, adj=c(-0.1,-0.1), cex=0.7, col=4)
  
  return(cooks)
}

########################
# Exploratory analysis #
########################
pairs(cancer_dt[ -10])

# HOMA is the interaction glucose between insulin.
plot(cancer_dt$Glucose * cancer_dt$Insulin,cancer_dt$HOMA)

# Leptin and bmi has association. # The association is not too strong
summary(lm(cancer_dt$BMI ~ cancer_dt$Leptin + I(cancer_dt$Leptin^2) )) 

# GAM
gamgam1 = gam::gam(Classification ~ s(Age, 4) + s(BMI,4)+ s(Glucose, 4)+ 
                      s(MCP.1, 4) + s(Insulin, 4) +
                    HOMA + s(Leptin, 4) + s(Adiponectin, 4)+ s(Resistin, 4),
                    family=binomial(), data=cancer_dt)
par(mfrow = c(3, 3))
plot(gamgam1)
summary(gamgam1)
residual_plot(gamgam1)


inference_model_full = glm(Classification ~ Age + I(Age^2) + BMI  + Glucose 
                           + MCP.1 + I(MCP.1^2)  + Insulin + HOMA 
                           + Leptin + I(Leptin^2) + I(Leptin^3) 
                           + Adiponectin + I(Adiponectin^2) + I(Adiponectin^3) +  
                             Resistin + I(Resistin^2), 
                           data = cancer_dt, family = binomial())


# Check fit & outliers.
residual_plot(inference_model_full)
par(mfrow = c(1, 2))
leverage_point = plot_leverage(inference_model_full, cancer_dt, 10)
cooks_dist = plot_cooks(inference_model_full, leverage_point, 10)

