import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn.metrics as sm

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


def test_result(model, true_label_vet, predictor_dt):
    test_missclassification_rate = 1 - model.score(predictor_dt, true_label_vet)
    confusion_dt = pd.DataFrame(
        sm.confusion_matrix(true_label_vet, model.predict(predictor_dt), labels=[1, 2]),
                            index=["true: Healthy", "true: Cancer"],
                            columns=["pred: Healthy", "predCancer"]
    )
    print(f"The misclassification rate on the {test_missclassification_rate}")
    print(confusion_dt)
#    print(sm.ConfusionMatrixDisplay(confusion_mat, display_labels=np.unique(true_label_vet)))


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
1 - vanilla_logistic_model.score(logistic_train_predictor_dt, train_label_vet)

# Test performance
test_result(vanilla_logistic_model, test_label_vet, logistic_test_predictor_dt)

#######################################
# Logistic regression with l2 penalty #
#######################################
def l2_logisitc_dt(dt):
    l2_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
                                     "bmi": dt["BMI"], "glucose": dt["Glucose"],
                                     "mcp": dt["MCP.1"], "mcp_squared": dt["MCP.1"].map(np.square),
                                     "mcp_cubic": dt["MCP.1"].map(lambda x: x ** 3),
                                     "insulin": dt["Insulin"], "homa": dt["HOMA"],
                                     "leptin": dt["Leptin"], "leptin_squared": dt["Leptin"].map(np.square),
                                     "leptin_cubic": dt["Leptin"].map(lambda x: x ** 3),
                                     "adiponectin": dt["Adiponectin"],
                                     "adiponectin_squared": dt["Adiponectin"].map(np.square),
                                     "adiponectin_cubic": dt["Adiponectin"].map(lambda x: x ** 3),
                                     "resistin": dt["Resistin"], "resitin_squared": dt["Resistin"].map(np.square)}


    l2_predictor_dt = pd.DataFrame(l2_logisitc_feature_dict)

    return l2_predictor_dt

l2_logistic_train_predictor_dt = l2_logisitc_dt(predictor_train_dt)
l2_logistic_test_predictor_dt = l2_logisitc_dt(predictor_test_dt)

l2_scaler = StandardScaler()
l2_scaler.fit(l2_logistic_train_predictor_dt)
l2_logistic_train_predictor_dt = l2_scaler.transform(l2_logistic_train_predictor_dt)
l2_logistic_test_predictor_dt = l2_scaler.transform(l2_logistic_test_predictor_dt)


# Create model and CV
l2_logistic_model = lm.LogisticRegression(penalty="l1", solver="saga", max_iter=100000, random_state=0, tol=1e-5)
logistic_param_grid = {'C': np.linspace(1, 20, num=40)}
# logistic_param_grid = {'C': np.logspace(-2, 2, 20)}
l2_logistic_search = GridSearchCV(l2_logistic_model, logistic_param_grid, cv=10, return_train_score=True)
l2_best_model = l2_logistic_search.fit(l2_logistic_train_predictor_dt, train_label_vet)


fig = plt.figure()
plt.scatter(l2_best_model.param_grid["C"],  1 - l2_logistic_search.cv_results_['mean_test_score'], label="Training")
plt.scatter(l2_best_model.param_grid["C"], 1 - l2_logistic_search.cv_results_['mean_train_score'], label="CV")
plt.xlabel("1 / \u03BB")
plt.ylabel("Misclassification Rate")
plt.legend()
fig.savefig("figure/lasso.png")

1 - np.max(l2_logistic_search.cv_results_["mean_test_score"])
test_result(l2_best_model, test_label_vet, l2_logistic_test_predictor_dt)


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
svm_param_grid = {'C': np.linspace(0.001, 10, num=50), 'degree': np.arange(1, 6)}

svm_search = GridSearchCV(svm_model, svm_param_grid, cv=10)
svm_best_model = svm_search.fit(svm_train_predictor_dt, train_label_vet)


1 - np.max(svm_search.cv_results_["mean_test_score"])
1 - svm_best_model.score(svm_train_predictor_dt, train_label_vet)

test_result(svm_best_model, test_label_vet, svm_test_predictor_dt)

