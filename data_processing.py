import pandas as pd
from sklearn.model_selection import train_test_split

cancer_dt = pd.read_csv("data/dataR2.csv")
cancer_train_dt, cancer_test_dt = train_test_split(cancer_dt, test_size=0.2, random_state=10)
cancer_train_dt.to_csv("data/cancer_train_dt.csv", index=False)
cancer_test_dt.to_csv("data/cancer_test_dt.csv", index=False)

