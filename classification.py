import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

cancer_train_dt = pd.read_csv("data/cancer_train_dt.csv")
cancer_test_dt = pd.read_csv("data/cancer_test_dt.csv")

predictor_train_dt = cancer_train_dt.drop("Classification", axis=1)
predictor_test_dt = cancer_test_dt.drop("Classification", axis=1)

train_label_vet = cancer_train_dt["Classification"]
test_label_vet = cancer_test_dt["Classification"]

###############################
# Vannila logistic regression #
###############################
def logisitc_dt(dt):
    vanilla_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
                                     "bmi": dt["BMI"], "glucose": dt["Glucose"],
                                     "mcp": dt["MCP.1"], "mcp_squared": dt["MCP.1"].map(np.square),
                                     "mcp_cubic": dt["MCP.1"].map(lambda x: x ** 3),
                                     "leptin": dt["Leptin"],
                                     "leptin_squared": dt["Leptin"].map(np.square),
                                     "leptin_cubic": dt["Leptin"].map(lambda x: x ** 3),
                                     "resistin": dt["Resistin"], "resitin_squared": dt["Resistin"].map(np.square)}

    vanilla_predictor_dt = pd.DataFrame(vanilla_logisitc_feature_dict)

    return vanilla_predictor_dt

# Create X mat for logistic regression
logistic_train_predictor_dt = logisitc_dt(cancer_train_dt)
logistic_test_predictor_dt = logisitc_dt(predictor_test_dt)

vanilar_logistic_scaler = StandardScaler()
vanilar_logistic_scaler.fit(logistic_train_predictor_dt)

logistic_train_predictor_dt = vanilar_logistic_scaler.transform(logistic_train_predictor_dt)
logistic_test_predictor_dt = vanilar_logistic_scaler.transform(logistic_test_predictor_dt)


vanilla_logistic_model = lm.LogisticRegression(penalty="none", solver="lbfgs",
                                               tol=1e-5, max_iter=100000).fit(logistic_train_predictor_dt,
                                                                   train_label_vet)
vanilla_logistic_model.score(logistic_train_predictor_dt, train_label_vet)

# Test performance
vanilla_logistic_model.score(logistic_test_predictor_dt, test_label_vet)


#######################################
# Logistic regression with L1 penalty #
#######################################
# def l1_logisitc_dt(dt):
#     l1_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
#                                      "bmi": dt["BMI"], "glucose": dt["Glucose"],
#                                      "mcp": dt["MCP.1"], "mcp_squared": dt["MCP.1"].map(np.square),
#                                      "mcp_cubic": dt["MCP.1"].map(lambda x: x ** 3),
#                                      "insulin": dt["Insulin"], "homa": dt["HOMA"],
#                                      "leptin": dt["Leptin"], "leptin_squared": dt["Leptin"].map(np.square),
#                                      "leptin_cubic": dt["Leptin"].map(lambda x: x ** 3),
#                                      "adiponectin": dt["Adiponectin"],
#                                      "adiponectin_squared": dt["Adiponectin"].map(np.square),
#                                      "adiponectin_cubic": dt["Adiponectin"].map(lambda x: x ** 3),
#                                      "resistin": dt["Resistin"], "resitin_squared": dt["Resistin"].map(np.square)}
#
#
#     l1_predictor_dt = pd.DataFrame(l1_logisitc_feature_dict)
#
#     return l1_predictor_dt


# Create model and CV
l1_logistic_model = lm.LogisticRegression(penalty="l1", solver="saga", max_iter=100000, random_state=0, tol=1e-5)
# param_grid = {'C': np.logspace(-1, 3, num=100, endpoint=True)}
logistic_param_grid = {'C': np.linspace(10, 60,num=60)}
l1_logistic_search = GridSearchCV(l1_logistic_model, logistic_param_grid, cv=10, return_train_score=True)
l1_best_model = l1_logistic_search.fit(logistic_train_predictor_dt, train_label_vet)

fig = plt.figure()
plt.scatter(l1_best_model.param_grid["C"], l1_logistic_search.cv_results_['mean_test_score'])
plt.scatter(l1_best_model.param_grid["C"], l1_logistic_search.cv_results_['mean_train_score'])
plt.xlabel("1 / \u03BB")
plt.ylabel("Accuracy")
fig.savefig("figure/lasso.png")
l1_best_model.score(logistic_test_predictor_dt, test_label_vet)


##############
# Kernel SVM #
##############
# Create X mat for svm
svm_scaler = StandardScaler()
svm_scaler.fit(predictor_train_dt)
svm_train_predictor_dt = svm_scaler.transform(predictor_train_dt)
svm_test_predictor_dt = svm_scaler.transform(predictor_test_dt)

# Create model and cv
svm_model = svm.SVC(kernel='poly')
svm_param_grid = {'C': np.linspace(0.0001, 20, num=100), 'degree': np.arange(1, 5)}

svm_search = GridSearchCV(svm_model, svm_param_grid, cv=10)
svm_best_model = svm_search.fit(svm_train_predictor_dt, train_label_vet)


svm_best_model.score(svm_train_predictor_dt, train_label_vet)
svm_best_model.score(svm_test_predictor_dt, test_label_vet)
svm_best_model.predict(svm_test_predictor_dt) == test_label_vet

import sklearn.metrics as sm
sm.confusion_matrix(test_label_vet, svm_best_model.predict(svm_test_predictor_dt))


#################
# Random forest #
#################
random_forest_model = RandomForestClassifier(random_state=0)

n_estimators = [100]
max_depth = [int(x) for x in np.arange(2, 6)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

forest_param_grid = {'n_estimators': [100],
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

forest_search = GridSearchCV(random_forest_model, forest_param_grid, cv=10)
forest_search.fit(logistic_train_predictor_dt, train_label_vet)

forest_search.score(logistic_train_predictor_dt, train_label_vet)
forest_search.cv_results_["mean_test_score"]

forest_search.score(logistic_test_predictor_dt, test_label_vet)
