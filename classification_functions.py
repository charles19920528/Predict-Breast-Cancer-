import numpy as np
import pandas as pd
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

cancer_train_dt = pd.read_csv("data/cancer_train_dt.csv")
cancer_test_dt = pd.read_csv("data/cancer_test_dt.csv")

predictor_train_dt = cancer_train_dt.drop("Classification", axis=1)
predictor_test_dt = cancer_test_dt.drop("Classification", axis=1)

train_label_vet = cancer_train_dt["Classification"]
test_label_vet = cancer_test_dt["Classification"]


def aic_logisitc_dt(dt):
    aic_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
                                 "bmi": dt["BMI"], "bmi_squared": dt["BMI"].map(np.square),
                                 "glucose": dt["Glucose"], "homa": dt['HOMA'],
                                 "resistin": dt["Resistin"], "adiponectin_squared": dt["Adiponectin"].map(np.square)}

    aic_predictor_dt = pd.DataFrame(aic_logisitc_feature_dict)

    return aic_predictor_dt


def bic_logisitc_dt(dt):
    bic_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
                                 "bmi_squared": dt["BMI"].map(np.square), "glucose": dt["Glucose"],
                                 "resistin": dt["Resistin"]}

    bic_predictor_dt = pd.DataFrame(bic_logisitc_feature_dict)

    return bic_predictor_dt

def lp_logisitc_dt(dt):
    lp_logisitc_feature_dict = {"age": dt["Age"], "age_square": dt["Age"].map(np.square),
                                "bmi": dt["BMI"], "bmi_squared": dt["BMI"].map(np.square),
                                "glucose": dt["Glucose"], "mcp": dt["MCP.1"],
                                "mcp_squared": dt["MCP.1"].map(np.square),
                                "insulin": dt["Insulin"], "homa": dt["HOMA"],
                                "leptin": dt["Leptin"], "leptin_squared": dt["Leptin"].map(np.square),
                                "adiponectin": dt["Adiponectin"],
                                "adiponectin_squared": dt["Adiponectin"].map(np.square),
                                "resistin": dt["Resistin"], "resitin_squared": dt["Resistin"].map(np.square)}

    lp_predictor_dt = pd.DataFrame(lp_logisitc_feature_dict)

    return lp_predictor_dt

def svm_dt(dt):
    svm_feature_dict = {"age": dt["Age"],
                                "bmi": dt["BMI"],
                                "glucose": dt["Glucose"], "mcp": dt["MCP.1"],
                                "insulin": dt["Insulin"], "homa": dt["HOMA"],
                                "leptin": dt["Leptin"],
                                "adiponectin": dt["Adiponectin"],
                                "resistin": dt["Resistin"]}

    svm_predictor_dt = pd.DataFrame(svm_feature_dict)

    return svm_predictor_dt


def test_result(model, true_label_vet, predictor_dt):
    test_missclassification_rate = 1 - model.score(predictor_dt, true_label_vet)
    confusion_dt = pd.DataFrame(
        sm.confusion_matrix(true_label_vet, model.predict(predictor_dt), labels=[1, 2]),
                            index=["true: Healthy", "true: Cancer"],
                            columns=["pred: Healthy", "predCancer"]
    )
    print(f"The misclassification rate on the {test_missclassification_rate}")
    print(confusion_dt)


def model_fitting(model, param_grid, dt_fun, train_dt=cancer_train_dt, test_dt=cancer_test_dt,
                  train_label_vet=train_label_vet, test_label_vet=test_label_vet, k_fold=LeaveOneOut()):
    train_predictor_dt = dt_fun(train_dt)
    test_predictor_dt = dt_fun(test_dt)

    scaler = StandardScaler().fit(train_predictor_dt)
    train_predictor_dt = scaler.transform(train_predictor_dt)
    test_predictor_dt = scaler.transform(test_predictor_dt)

    grid_cv_instance = GridSearchCV(model, param_grid=param_grid, cv=k_fold, return_train_score=True)
    grid_cv_instance.fit(train_predictor_dt, train_label_vet)

    best_test_score_index = np.argmax(grid_cv_instance.cv_results_["mean_test_score"])
    training_score_to_report = 1 - grid_cv_instance.cv_results_["mean_train_score"][best_test_score_index]
    best_cv_score = 1 - np.max(grid_cv_instance.cv_results_["mean_test_score"])

    test_missclassification_rate = 1 - grid_cv_instance.score(test_predictor_dt, test_label_vet)
    confusion_dt = pd.DataFrame(
        sm.confusion_matrix(test_label_vet, grid_cv_instance.predict(test_predictor_dt), labels=[1, 2]),
        index=["true: Healthy", "true: Cancer"],
        columns=["pred: Healthy", "predCancer"]
    )

    precision = confusion_dt.iloc[1, 1] / np.sum(confusion_dt.iloc[:, 1])
    recall = confusion_dt.iloc[1, 1] / np.sum(confusion_dt.iloc[1, :])

    print(f"Mean train error is {training_score_to_report}")
    print(f"Mean cv_error is {best_cv_score}")
    print(f"The misclassification rate on the test data {test_missclassification_rate}")
    print(f"Precision is {precision}")
    print(f"Recall is {recall}")

    return training_score_to_report, best_cv_score, test_missclassification_rate, precision, recall, \
           grid_cv_instance, confusion_dt

def plot_error(model_fitting_result_tuple, figure_name):
    fig = plt.figure()
    plt.scatter(model_fitting_result_tuple[5].param_grid["C"],
                1 - model_fitting_result_tuple[5].cv_results_['mean_test_score'], label="Training")
    plt.scatter(model_fitting_result_tuple[5].param_grid["C"],
                1 - model_fitting_result_tuple[5].cv_results_['mean_train_score'], label="CV")

    plt.xlabel("1 / \u03BB")
    plt.ylabel("Misclassification Rate")
    plt.legend()
    fig.savefig(f"figure/{figure_name}.png")