import sklearn.linear_model as lm
import numpy as np
from sklearn import svm
import classification_functions as cf

###############################
# vanilla logistic regression #
###############################
vanilla_logistic_model = lm.LogisticRegression(solver="lbfgs", tol=1e-5, max_iter=100000, random_state=500)
vanilla_logistic_param_grid = {"penalty": ["none"]}

baseline_result_tuple = cf.model_fitting(model=vanilla_logistic_model, param_grid=vanilla_logistic_param_grid,
                                         dt_fun=cf.svm_dt)

aic_result_tuple = cf.model_fitting(model=vanilla_logistic_model, param_grid=vanilla_logistic_param_grid,
                                   dt_fun=cf.aic_logisitc_dt)

bic_result_tuple = cf.model_fitting(model=vanilla_logistic_model, param_grid=vanilla_logistic_param_grid,
                                   dt_fun=cf.bic_logisitc_dt)


##########################################
# Logistic regression with regularzation #
##########################################
l2_logistic_model = lm.LogisticRegression(penalty="l2", solver="saga", max_iter=100000, random_state=0, tol=1e-5)

logistic_param_grid = {'C': np.logspace(-2, 1, 100)}
l2_baseline_logistic_model = cf.model_fitting(model=l2_logistic_model, param_grid=logistic_param_grid,
                                              dt_fun=cf.svm_dt)
cf.plot_error(l2_baseline_logistic_model, "l2_baseline")

logistic_param_grid = {'C': np.logspace(1, 3, 100)}
l2_aic_result_tuple =  cf.model_fitting(model=l2_logistic_model, param_grid=logistic_param_grid,
                                        dt_fun=cf.aic_logisitc_dt)
cf.plot_error(l2_aic_result_tuple, "l2_aic")

logistic_param_grid = {'C': np.logspace(-1, 1, 100)}
l2_bic_result_tuple = cf.model_fitting(model=l2_logistic_model, param_grid=logistic_param_grid,
                                       dt_fun=cf.bic_logisitc_dt)
cf.plot_error(l2_bic_result_tuple, "l2_bic")


l1_logistic_model = lm.LogisticRegression(penalty="l1", solver="saga", max_iter=100000, random_state=0, tol=1e-5)

logistic_param_grid = {'C': np.logspace(-1, 2, 20)}
l1_result_tuple = cf.model_fitting(model=l1_logistic_model, param_grid=logistic_param_grid, dt_fun=cf.lp_logisitc_dt)
cf.plot_error(l1_result_tuple, "l1_full")

# Find out the coefficient of l1 model
l1_result_tuple[5].best_estimator_.coef_

# L2 full model
logistic_param_grid = {'C': np.logspace(-1, 2, 100)}
l2_full_result_tuple = cf.model_fitting(model=l2_logistic_model, param_grid=logistic_param_grid,
                                        dt_fun=cf.lp_logisitc_dt)
cf.plot_error(l2_full_result_tuple, "l2_full")

##############
# Kernel SVM #
##############
# Create model and cv
svm_model = svm.SVC(kernel='poly')
svm_param_grid = {'C': np.logspace(1, 2, num=100), 'degree': [3]}

svm_poly_result_tuple = cf.model_fitting(model=svm_model, param_grid=svm_param_grid, dt_fun=cf.svm_dt)
cf.plot_error(svm_poly_result_tuple, "svm_poly")



