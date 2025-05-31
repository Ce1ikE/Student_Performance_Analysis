import pandas as pd
from main_config import (
    train_data_cleaned_no_outliers_path,
    test_data_cleaned_no_outliers_path,
)

df_test = pd.read_csv(test_data_cleaned_no_outliers_path)
df_train = pd.read_csv(train_data_cleaned_no_outliers_path)

Y_TRAIN = df_train["Exam_Score"]
X_TRAIN = df_train.drop(columns=["Exam_Score"]).copy()

Y_TEST = df_test["Exam_Score"]
X_TEST = df_test.drop(columns=["Exam_Score"]).copy()

