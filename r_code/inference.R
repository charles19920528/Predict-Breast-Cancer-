library(gam)
library(MASS)
library(lawstat)
library(boot)

cancer_dt <- read.csv("~/Documents/courses/sta_223/project/dataR2.csv")
cancer_dt$Classification = as.numeric(cancer_dt$Classification == 2)

cancer_dt = cancer_dt[-c(17, 38, 51), ]
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
  lines(smooth.spline(bwtfit$fitted.values, res.D, spar=0.95), col=2)
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

bootstrap_ci_one_slope = function(estimate_vet, number_of_trials, alpha) {
  interval = sort(estimate_vet)[round(c((number_of_trials+1)*alpha/2, 
                                        (number_of_trials+1)*(1-alpha/2)))]
  return(interval)
}

bootstrap_ci = function(boot_object, number_of_trials, alpha) {
  ci = t(apply(boot_object$t, 2, function(x) {
    bootstrap_ci_one_slope(estimate_vet = x, number_of_trials = number_of_trials,
                           alpha = alpha)
    }))
  width = ci[, 2] - ci[, 1]
  return(cbind(ci, width))
}

normal_ci = function(glm_fit, alpha) {
  ci = confint(glm_fit, level = alpha)
  width = ci[, 2] - ci[, 1]
  return(cbind(ci, width))
}

#######
# GAM #
#######
# GAM
gam_model = gam::gam(Classification ~ s(Age, 2) + s(BMI,3)+ s(Glucose, 2)+ 
                       s(MCP.1, 3) + s(Insulin, 2) + BMI:Age+
                       HOMA + s(Leptin, 2) + s(Adiponectin, 2)+ s(Resistin, 2),
                     family=binomial(), data=cancer_dt)
par(mfrow = c(3, 3))
plot(gam_model)
summary(gam_model)
residual_plot(gam_model)

inference_model_full = glm(Classification ~ Age + I(Age^2) + BMI + I(BMI^2)  + Glucose 
                           + MCP.1 + I(MCP.1^2)  + Insulin + HOMA 
                           + Leptin + I(Leptin^2) + Adiponectin + 
                             I(Adiponectin^2)  + Resistin,  
                           data = cancer_dt, family = binomial())

stepAIC(inference_model_full)
inference_model_aic = glm(formula = Classification ~ Age + I(Age^2) + BMI + I(BMI^2) + 
                            Glucose + HOMA + I(Adiponectin^2) + Resistin, family = binomial(), 
                          data = cancer_dt)
residual_plot(inference_model_aic)



stepAIC(inference_model_full, k = log(nrow(cancer_dt)))
inference_model_bic = glm(formula = Classification ~ Age + I(Age^2) + I(BMI^2) + Glucose + 
                            Resistin, family = binomial(), data = cancer_dt)

# Check fit of bic model
residual_plot(inference_model_bic)
par(mfrow=c(1, 2))
leverage_bic = plot_leverage(inference_model_bic, cancer_dt, 5)
cook_bic = plot_cooks(inference_model_bic, leverage_bic, 5)
round(normal_ci(inference_model_bic, 0.998), 4)

summary(inference_model_bic)

# Confidence interval for beta1 / (2 * beta2)
beta_12_vet = coef(inference_model_bic)[2:3]
ratio = beta_12_vet[1] / (-2 * beta_12_vet[2])
cov_mat = vcov(inference_model_bic)[2:3, 2:3]
gradient_vet = matrix(c(- 1 / (2 * beta_12_vet[2]), beta_12_vet[1] / (2 * beta_12_vet[2]^2)),
                      nrow = 2)
ratio_var = t(gradient_vet) %*% cov_mat %*% gradient_vet
ratio - qnorm(0.005) * sqrt(ratio_var)
ratio + qnorm(0.005) * sqrt(ratio_var)

# Boostrap
set.seed(1)
number_of_trials = 10000

slope_bic = function(data, ind){
  res <- glm(formula = Classification ~ Age + I(Age^2) + I(BMI^2) + Glucose + 
               Resistin, family = binomial(), data = data[ind,])
  coef(res)
}

boot_bic = boot(data = cancer_dt, statistic = slope_bic, R = number_of_trials)
boot_bic_ci = bootstrap_ci(boot_object = boot_bic, 
                           number_of_trials = number_of_trials,
                           alpha = 0.01/5)
# Confidence intervals
round(boot_bic_ci, 4)

ratio_estimate_vet = - boot_bic$t[, 2] / (2 * boot_bic$t[, 3] )
quantile(ratio_estimate_vet, probs = c(0.01, 0.99))

# E
par(mfrow=c(1, 1))
plot(boot_bic$t[, 2], boot_bic$t[, 3])

par(mfrow = c(2, 2))
apply(boot_bic$t[, 2:3], 2,hist)




