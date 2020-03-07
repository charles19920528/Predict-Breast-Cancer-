import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

cancer_dt = pd.read_csv("data/cancer_train_dt.csv")
predictor_dt = cancer_dt.drop("Classification", axis=1)

scaler = StandardScaler()
scaler.fit(predictor_dt)
scaled_predictor_dt = scaler.transform(predictor_dt)

#######
# PCA #
#######
pca_instance = PCA()
pca_instance.fit(scaled_predictor_dt)
print(pca_instance.explained_variance_ratio_)
plt.plot(pca_instance.explained_variance_ratio_)

principal_component_mat = pca_instance.transform(scaled_predictor_dt)

# Dimension is roughly 6
sum(pca_instance.explained_variance_ratio_[0:5])

# Visualize first two principle component
patient_boolean_vet = cancer_dt['Classification'] == 2
color_vet = ["red", "blue"]
plt.figure()
non_patient_plot = plt.scatter(x=principal_component_mat[- patient_boolean_vet, 0],
                               y=principal_component_mat[- patient_boolean_vet, 1], c="navy")
patient_plot = plt.scatter(x=principal_component_mat[patient_boolean_vet, 0],
                           y=principal_component_mat[patient_boolean_vet, 1], c="orange")
plt.legend((non_patient_plot, patient_plot), ("Healthy", "Cancer"), loc="upper left")

# Potential outliers
largest_5_indices_1 = np.argpartition(-principal_component_mat[:, 0], 5)[: 5]
largest_5_indices_2 = np.argpartition(-principal_component_mat[:, 1], 5)[: 5]

principal_component_mat[largest_5_indices_1, 0:2]
principal_component_mat[largest_5_indices_2, 0:2]

# 87, 78, 88, 85, 86 are outliers.
principal_component_mat[[87, 78, 88, 85, 86], 0:2]